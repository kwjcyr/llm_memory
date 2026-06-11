"""
Effective Memory → Long-term Memory 流转器

职责：
1. 读取 effective_memories（JSONL 格式）
2. 批量调用 LLM 判断每条 effective 应该执行 ADD / UPDATE / DELETE / NONE
3. 执行对应操作写入 LongTermMemoryDatabase
4. 保留原始 time_range 信息（对 QA 时间推理至关重要）

Prompt 模板从 config/skill/longterm_ops.md 加载。
基于 organizeParam 的四操作模式（ADD/UPDATE/DELETE/NONE）。
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ── 路径配置（当前在 longterm 包下） ──────────────────────────────
BASE_DIR = str(Path(__file__).resolve().parents[3])  # longterm → memory → src → 项目根
sys.path.insert(0, BASE_DIR)

from src.call_llm.call_llm_chat import call_llm as _call_llm_unified

from src.storage.sqlite.longterm_memory_db import LongTermMemoryDatabase, LongTermMemory

PROMPT_MD_PATH   = os.path.join(BASE_DIR, 'config', 'skill', 'longterm_ops.md')
EFFECTIVE_MEMORIES_PATH = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')


# ── Prompt 加载 ────────────────────────────────────────────────────────────

def load_prompt_template(md_path: str = PROMPT_MD_PATH) -> str:
    """从 .md 文件加载 Prompt 模板"""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()


# ── LLM 调用（使用统一 call_llm） ──────────────────────────────────────────────────

def _call_llm(prompt: str, model: str = None,
               temperature: float = None, max_tokens: int = None) -> Optional[str]:
    """调用 LLM，返回文本内容。"""
    try:
        return _call_llm_unified(prompt, model=model, temperature=temperature,
                                   max_tokens=max_tokens)
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return None


# ── LLM 决策解析（新格式：{"memory": [{"id": ..., "value": ..., "event": ...}]}） ──

def parse_llm_decision(response: str) -> List[Dict[str, Any]]:
    """
    解析 LLM 返回的操作决策 JSON。

    新格式（organizeParam 风格）：
    {
      "memory": [
        {"id": 0, "value": "...", "event": "NONE"},
        {"id": 1, "value": "...", "event": "ADD"},
        {"id": 2, "value": "...", "event": "UPDATE", "old_memory": "..."},
        ...
      ]
    }

    返回：memory 数组中的操作列表
    """
    # 尝试提取 JSON 对象
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            result = json.loads(json_match.group())
            # 新格式：{"memory": [...]}
            if 'memory' in result and isinstance(result['memory'], list):
                return result['memory']
            # 旧格式兼容：{"operation": ..., "target_memory_id": ...}
            else:
                op = result.get('operation', 'NOOP').upper()
                target_id = result.get('target_memory_id')
                reason = result.get('reason', '')
                return [{'id': 0, 'value': reason, 'event': op}]
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON 解析失败: {e}")

    # 回退：从文本推断操作
    text = response.upper()
    if 'ADD' in text or 'INSERT' in text:
        return [{'id': -1, 'value': response[:200], 'event': 'ADD'}]
    elif 'UPDATE' in text:
        return [{'id': 0, 'value': response[:200], 'event': 'UPDATE'}]
    elif 'DELETE' in text:
        return [{'id': 0, 'value': response[:200], 'event': 'DELETE'}]
    else:
        return [{'id': 0, 'value': '', 'event': 'NONE'}]


# ── 构建批量 Prompt ────────────────────────────────────────────────────────────

def build_batch_prompt(
    existing_memories: List[LongTermMemory],
    new_effective_memories: List[Dict[str, Any]],
    prompt_template: str
) -> str:
    """
    构建批量处理的 Prompt，一次性传入所有已有记忆和新的 effective memories。

    这比逐条处理更高效，且能让 LLM 看到全局上下文做出更好的去重/合并决策。
    """
    # 构建已有记忆列表
    existing_lines = []
    for i, mem in enumerate(existing_memories):
        mem_dict = {
            'id': i,
            'topic': mem.topic,
            'summary': mem.summary,
            'facts': mem.facts[:5] if mem.facts else [],
            'tags': mem.tags if mem.tags else [],
            'time_range': getattr(mem, 'time_range', None),
        }
        existing_lines.append(json.dumps(mem_dict, ensure_ascii=False))

    existing_str = '\n'.join(existing_lines) if existing_lines else "(no existing long-term memories)"

    # 构建新的 effective memories 列表
    new_lines = []
    for i, em in enumerate(new_effective_memories):
        em_dict = {
            'effective_idx': i,
            'topic': em.get('topic', ''),
            'summary': em.get('summary', ''),
            'facts': em.get('facts', [])[:8],  # 保留更多 facts 用于决策
            'tags': em.get('tags', []),
            'memory_type': em.get('memory_type', ''),
            'confidence': em.get('confidence'),
            'time_range': em.get('time_range'),  # ⏰ 保留时间信息！
            'source_memory_ids': em.get('source_memory_ids', [])[:5],
        }
        new_lines.append(json.dumps(em_dict, ensure_ascii=False))

    new_str = '\n'.join(new_lines)

    # 替换模板变量
    prompt = prompt_template.replace('{{{existing_memories}}}', existing_str)
    prompt = prompt.replace('{{{new_memory}}}', new_str)

    return prompt


# ── 执行批量决策结果 ──────────────────────────────────────────────────────────

def execute_batch_decisions(
    decisions: List[Dict[str, Any]],
    new_effective_memories: List[Dict[str, Any]],
    existing_memories: List[LongTermMemory],
    db: LongTermMemoryDatabase,
    user_id: str,
    verbose: bool = True
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    执行 LLM 返回的批量决策（ADD/UPDATE/DELETE/NONE）。

    Returns (stats, results)
    """
    stats = {'ADD': 0, 'UPDATE': 0, 'DELETE': 0, 'NONE': 0, 'ERROR': 0}
    results = []

    # 构建 id → existing_memory 映射
    id_to_existing = {}
    for i, mem in enumerate(existing_memories):
        id_to_existing[i] = mem

    for decision in decisions:
        event = decision.get('event', 'NONE').upper()
        value = decision.get('value', '')
        mem_id = decision.get('id', -1)

        if event == 'ADD':
            # 新增：优先使用原始 effective memory 数据，保留 time_range 等元信息
            try:
                # 🔑 关键改进：通过关键词匹配找到对应的原始 effective memory
                matched_em = None
                value_lower = value.lower() if isinstance(value, str) else ''

                # 方法 1：精确 topic 匹配
                for em in new_effective_memories:
                    if em.get('topic', '') == value or em.get('topic', '') in value or value in em.get('topic', ''):
                        matched_em = em
                        break

                # 方法 2：关键词重叠度匹配（如果方法1失败）
                if not matched_em and value_lower:
                    best_overlap = 0
                    for em in new_effective_memories:
                        # 计算 value 和 effective topic/summary 的关键词重叠
                        em_text = (em.get('topic', '') + ' ' + em.get('summary', '')).lower()
                        value_words = set(value_lower.split())
                        em_words = set(em_text.split())
                        overlap = len(value_words & em_words)
                        if overlap > best_overlap and overlap >= 3:  # 至少 3 个词重叠
                            best_overlap = overlap
                            matched_em = em

                # 方法 3：按顺序分配（最后手段）
                if not matched_em:
                    used_indices = {r.get('eff_idx') for r in results if 'eff_idx' in r}
                    for idx, em in enumerate(new_effective_memories):
                        if idx not in used_indices:
                            matched_em = em
                            break

                # 构建新记忆数据
                if matched_em:
                    # ✅ 使用原始 effective memory 数据，保留完整元信息
                    new_mem_data = matched_em.copy()
                    # 用 LLM 生成的 value 更新 summary（如果更有信息量）
                    if len(value) > len(matched_em.get('summary', '')):
                        new_mem_data['summary'] = value
                    if verbose:
                        time_info = matched_em.get('time_range', {})
                        print(f"  📅 Time Range: {time_info}")
                else:
                    # 回退：使用 LLM 返回的 value
                    new_mem_data = {
                        'topic': value[:100],
                        'summary': value,
                        'facts': [value],
                        'tags': [],
                        'memory_type': 'LongTermMemory',
                        'confidence': 0.9,
                    }
                    if verbose:
                        print(f"  ⚠️ 未匹配到原始 effective，使用 LLM value")

                result_id = db.add_memory_from_dict(user_id, new_mem_data)
                stats['ADD'] += 1
                result_record = {
                    'operation': 'ADD',
                    'memory_id': result_id,
                    'topic': new_mem_data.get('topic', value),
                    'eff_idx': new_effective_memories.index(matched_em) if matched_em else -1,
                }
                results.append(result_record)
                if verbose:
                    print(f"  ✅ ADD → {new_mem_data.get('topic', value)[:50]}... (id={result_id[:8] if result_id else '?'})")
            except Exception as e:
                stats['ERROR'] += 1
                results.append({'operation': 'ADD', 'memory_id': None, 'topic': str(value)[:80], 'error': str(e)})
                if verbose:
                    print(f"  ❌ ADD 失败: {e}")

        elif event == 'UPDATE':
            # 更新已有记忆
            target_mem = id_to_existing.get(mem_id)
            if target_mem:
                try:
                    # 合并新旧数据
                    update_data = {
                        'summary': value,
                        'facts': decision.get('facts', target_mem.facts),
                    }
                    success = db.merge_and_update(target_mem.memory_id, update_data)
                    stats['UPDATE'] += 1
                    results.append({
                        'operation': 'UPDATE',
                        'memory_id': target_mem.memory_id,
                        'topic': target_mem.topic,
                        'old_memory': decision.get('old_memory', '')
                    })
                    if verbose:
                        print(f"  ✅ UPDATE → {target_mem.topic[:50]}...")
                except Exception as e:
                    stats['ERROR'] += 1
                    results.append({'operation': 'UPDATE', 'memory_id': target_mem.memory_id, 'error': str(e)})
                    if verbose:
                        print(f"  ❌ UPDATE 失败: {e}")
            else:
                # 找不到目标，转为 ADD
                stats['ADD'] += 1
                results.append({'operation': 'ADD(fallback)', 'memory_id': None, 'topic': value})
                if verbose:
                    print(f"  ⚠️ UPDATE 目标不存在，转为 ADD: {value[:50]}")

        elif event == 'DELETE':
            # 删除已有记忆
            target_mem = id_to_existing.get(mem_id)
            if target_mem:
                try:
                    success = db.delete_memory(target_mem.memory_id)
                    stats['DELETE'] += 1
                    results.append({
                        'operation': 'DELETE',
                        'memory_id': target_mem.memory_id,
                        'topic': target_mem.topic,
                        'conflict_memory': decision.get('conflict_memory', '')
                    })
                    if verbose:
                        print(f"  🗑️ DELETE → {target_mem.topic[:50]}...")
                except Exception as e:
                    stats['ERROR'] += 1
                    results.append({'operation': 'DELETE', 'memory_id': target_mem.memory_id, 'error': str(e)})
            else:
                stats['NONE'] += 1
                results.append({'operation': 'NONE(no target)', 'memory_id': None, 'topic': value})

        else:  # NONE
            stats['NONE'] += 1
            results.append({'operation': 'NONE', 'memory_id': None, 'topic': value})

    return stats, results


# ── 核心流转函数（批量处理） ──────────────────────────────────────────────────

def sync_effective_to_longterm(
        user_id: str,
        effective_file: str = EFFECTIVE_MEMORIES_PATH,
        db_path: Optional[str] = None,
        prompt_md: str = PROMPT_MD_PATH,
        jsonl_output: Optional[str] = None,
        batch_size: int = 10,  # 每批处理多少条 effective
        verbose: bool = True) -> Dict[str, Any]:
    """
    将 effective_memories 文件全量同步到 Long-term Memory。

    使用批量处理模式：
    1. 加载所有 effective memories
    2. 加载所有已有的 long-term memories
    3. 分批调用 LLM 做决策（ADD/UPDATE/DELETE/NONE）
    4. 执行决策并写入数据库
    5. 导出 JSONL（包含 time_range）
    """
    print("=" * 70)
    print("Effective Memory → Long-term Memory 同步（批量模式）")
    print("=" * 70)

    # 初始化数据库
    db = LongTermMemoryDatabase(db_path) if db_path else LongTermMemoryDatabase()
    prompt_template = load_prompt_template(prompt_md)

    # 加载 effective memories
    effective_memories = load_effective_memories_from_file(effective_file)
    print(f"\n📂 加载 effective_memories: {len(effective_memories)} 条")

    # 加载已有 long-term memories
    existing_memories = db.get_memories_by_user(user_id)
    print(f"📂 已有 long-term memories: {len(existing_memories)} 条")

    if not effective_memories:
        print("⚠️ 没有 effective memories 需要处理")
        return {'stats': {}, 'results': [], 'db_stats': db.get_stats(user_id)}

    # 分批处理
    all_stats = {'ADD': 0, 'UPDATE': 0, 'DELETE': 0, 'NONE': 0, 'ERROR': 0}
    all_results = []

    for batch_start in range(0, len(effective_memories), batch_size):
        batch = effective_memories[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(effective_memories) + batch_size - 1) // batch_size

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"📦 处理批次 {batch_num}/{total_batches} ({len(batch)} 条)")
            print(f"{'─' * 50}")

        # 构建 Prompt
        prompt = build_batch_prompt(existing_memories, batch, prompt_template)

        # 调用 LLM
        if verbose:
            print(f"  🤖 调用 LLM 决策...")
        response = _call_llm(prompt)

        if not response:
            if verbose:
                print(f"  ❌ LLM 调用失败，本批跳过")
            continue

        # 解析决策
        decisions = parse_llm_decision(response)
        if verbose:
            print(f"  📋 收到 {len(decisions)} 条决策")

        # 执行决策
        batch_stats, batch_results = execute_batch_decisions(
            decisions, batch, existing_memories, db, user_id, verbose=verbose
        )

        # 更新已有记忆列表（用于下一批次的上下文）
        existing_memories = db.get_memories_by_user(user_id)

        # 累加统计
        for k, v in batch_stats.items():
            base_key = k.split('(')[0]
            all_stats[base_key] = all_stats.get(base_key, 0) + v
        all_results.extend(batch_results)

    # 打印统计
    print("\n" + "=" * 70)
    print("✅ 同步完成！")
    print(f"   ADD    : {all_stats.get('ADD', 0)}")
    print(f"   UPDATE : {all_stats.get('UPDATE', 0)}")
    print(f"   DELETE : {all_stats.get('DELETE', 0)}")
    print(f"   NONE   : {all_stats.get('NONE', 0)}")
    if all_stats.get('ERROR', 0) > 0:
        print(f"   ERROR  : {all_stats.get('ERROR', 0)}")

    # 数据库状态
    db_stats = db.get_stats(user_id)
    print(f"\n📊 数据库状态 ({user_id}):")
    print(f"   活跃记忆: {db_stats['total_active']} 条")
    print(f"   LongTermMemory: {db_stats['long_term_memory']} 条")
    print(f"   UserMemory: {db_stats['user_memory']} 条")

    # 导出 JSONL
    if jsonl_output is None:
        jsonl_path = os.path.join(
            os.path.dirname(db_path) if db_path else BASE_DIR,
            'data', 'longterm_memories.jsonl'
        )
    else:
        jsonl_path = jsonl_output

    _export_jsonl(db, jsonl_path, user_id=user_id, results=all_results)
    print(f"\n💾 同时导出 JSONL → {jsonl_path}")

    return {'stats': all_stats, 'results': all_results, 'db_stats': db_stats}


# ── 文件加载 ──────────────────────────────────────────────────────────────────────

def load_effective_memories_from_file(file_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载所有 effective_memories"""
    memories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                memories.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️ 解析失败: {e}")
    return memories


# ── JSONL 导出（包含 time_range） ──────────────────────────────────────────────

def _export_jsonl(db: LongTermMemoryDatabase, output_path: str, user_id: str = '', results: List[Dict] = None) -> None:
    """
    将 SQLite 中的长期记忆导出为 JSONL 文件
    包含操作类型（ADD/UPDATE/DELETE）、来源 Effective Memory ID 和 time_range
    """
    memories = db.get_memories_by_user(user_id=user_id)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 构建 operation 映射
    op_map = {}
    if results:
        for r in results:
            mid = r.get('memory_id', '')
            if mid:
                op_map[mid] = r.get('operation', 'UNKNOWN')

    export_records = []
    for mem in memories:
        record = {
            'memory_id': mem.memory_id,
            'user_id': mem.user_id,
            'topic': mem.topic,
            'summary': mem.summary,
            'facts': mem.facts,
            'tags': mem.tags,
            'confidence': mem.confidence,
            # ⏰ 关键：保留时间信息
            'time_range_start': getattr(mem, 'time_range_start', None),
            'time_range_end': getattr(mem, 'time_range_end', None),
            'created_at': mem.created_at,
            'updated_at': mem.updated_at,
            'is_deleted': getattr(mem, 'is_deleted', 0),
            # 操作信息
            'operation': op_map.get(mem.memory_id, 'UNKNOWN'),
            # 来源 Effective Memory IDs
            'source_effective_ids': getattr(mem, 'source_effective_ids', []),
        }
        export_records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in export_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    print(f"   已导出 {len(export_records)} 条长期记忆到 {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Effective Memory → Long-term Memory 同步（批量模式）')
    parser.add_argument('--user', type=str, default='caroline', help='用户 ID (如 caroline, gina)')
    parser.add_argument('--group', type=str, default=None, help='Group 名称 (如 group_1_gina)，自动推导 user 和 effective 文件路径')
    parser.add_argument('--batch-size', type=int, default=10, help='每批处理的 effective memory 数量（默认 10）')
    args = parser.parse_args()

    if args.group:
        # 从 group 名称推导路径
        effective_file = os.path.join(BASE_DIR, 'data', 'groups', args.group, 'effective', 'effective_memories.jsonl')
        if not os.path.exists(effective_file):
            print(f"❌ Effective Memory 文件不存在: {effective_file}")
            print(f"   请先运行: python src/memory/effective_memory/extract_from_session.py --group {args.group}")
            sys.exit(1)
        user_id = args.group.split('_')[-1]
        jsonl_output = os.path.join(BASE_DIR, 'data', 'groups', args.group, 'longterm', 'longterm_memories.jsonl')
        print(f"\n📂 使用 Group: {args.group} → User: {user_id}")
        print(f"   Effective File: {effective_file}")
        print(f"   Long-term Output: {jsonl_output}")
    else:
        user_id = args.user
        effective_file = EFFECTIVE_MEMORIES_PATH
        jsonl_output = None

    sync_effective_to_longterm(
        user_id=user_id,
        effective_file=effective_file,
        jsonl_output=jsonl_output,
        batch_size=args.batch_size,
        verbose=True
    )

