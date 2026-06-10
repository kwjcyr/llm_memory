"""
Effective Memory → Long-term Memory 流转器

职责：
1. 读取 effective_memories（或逐条接收）
2. 对每条 effective_memory，检索已有长期记忆中相似条目
3. 调用 LLM 判断应该执行 INSERT / UPDATE / DELETE / NOOP
4. 执行对应操作写入 LongTermMemoryDatabase

Prompt 模板从 config/prompt/longterm_ops.md 加载。
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


# ── LLM 决策解析 ──────────────────────────────────────────────────────────────────

def parse_llm_decision(response: str) -> Dict[str, Any]:
    """解析 LLM 返回的操作决策 JSON。"""
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    text = response.upper()
    if 'INSERT' in text:
        return {'operation': 'INSERT', 'target_memory_id': None, 'reason': 'inferred from text'}
    elif 'UPDATE' in text:
        return {'operation': 'UPDATE', 'target_memory_id': None, 'reason': 'inferred from text'}
    elif 'DELETE' in text:
        return {'operation': 'DELETE', 'target_memory_id': None, 'reason': 'inferred from text'}
    else:
        return {'operation': 'NOOP', 'target_memory_id': None, 'reason': 'fallback NOOP'}


# ── 核心决策函数 ──────────────────────────────────────────────────────────────────

def decide_operation(effective_memory: Dict[str, Any],
                     existing_memories: List[LongTermMemory],
                     prompt_template: str) -> Dict[str, Any]:
    """调用 LLM 决定对新的 effective_memory 执行什么操作。"""
    # 构建已有记忆摘要
    existing_str = ""
    if existing_memories:
        lines = []
        for i, mem in enumerate(existing_memories, 1):
            lines.append(
                f"[{i}] ID={mem.memory_id}\n"
                f"    Topic: {mem.topic}\n"
                f"    Summary: {mem.summary[:200]}...\n"
                f"    Facts: {'; '.join(mem.facts[:3])}\n"
            )
        existing_str = "\n".join(lines)
    else:
        existing_str = "(no existing memories)"

    new_memory_str = json.dumps({
        "topic": effective_memory.get("topic", ""),
        "summary": effective_memory.get("summary", ""),
        "facts": effective_memory.get("facts", [])[:5],
        "memory_type": effective_memory.get("memory_type", ""),
        "tags": effective_memory.get("tags", []),
    }, ensure_ascii=False, indent=2)

    prompt = prompt_template.replace('{{{existing_memories}}}', existing_str)
    prompt = prompt.replace('{{{new_memory}}}', new_memory_str)

    response = _call_llm(prompt)
    if not response:
        return {'operation': 'INSERT', 'target_memory_id': None, 'reason': 'LLM failed, fallback INSERT'}
    return parse_llm_decision(response)


# ── 主流转：处理单条 effective memory ──────────────────────────────────────

def process_effective_memory(user_id: str,
                              effective_memory: Dict[str, Any],
                              db: LongTermMemoryDatabase,
                              prompt_template: str,
                              verbose: bool = True) -> Tuple[str, str]:
    """处理单条 effective_memory，决策并写入 Long-term Memory。

    Returns (operation, memory_id_or_empty).
    """
    topic = effective_memory.get('topic', 'unknown')
    if verbose:
        print(f"\n  📥 处理: {topic}")

    related = db.search_memories(user_id, query=topic, top_k=5)
    decision = decide_operation(effective_memory, related, prompt_template)
    operation = decision.get('operation', 'INSERT').upper()
    target_id = decision.get('target_memory_id')
    reason = decision.get('reason', '')

    if verbose:
        print(f"  🤖 决策: {operation} | 原因: {reason}")

    result_id = None
    if operation == 'INSERT':
        result_id = db.add_memory_from_dict(user_id, effective_memory)
        if verbose:
            print(f"  ✅ INSERT → memory_id={result_id[:8]}...")
    elif operation == 'UPDATE':
        if target_id and db.get_memory_by_id(target_id):
            success = db.merge_and_update(target_id, effective_memory)
            result_id = target_id if success else None
        else:
            similar = db.find_similar_by_topic(user_id, topic, threshold=0.2)
            if similar:
                success = db.merge_and_update(similar.memory_id, effective_memory)
                result_id = similar.memory_id if success else None
            else:
                result_id = db.add_memory_from_dict(user_id, effective_memory)
                operation = 'INSERT(fallback)'
        if verbose:
            print(f"  ✅ UPDATE → {result_id[:8] if result_id else 'N/A'}")
    elif operation == 'DELETE':
        if target_id and db.get_memory_by_id(target_id):
            success = db.delete_memory(target_id)
            result_id = target_id if success else None
        else:
            similar = db.find_similar_by_topic(user_id, topic, threshold=0.5)
            if similar:
                success = db.delete_memory(similar.memory_id)
                result_id = similar.memory_id if success else None
            else:
                operation = 'NOOP(no target)'
        if verbose:
            print(f"  🗑️ DELETE → {result_id[:8] if result_id else 'N/A'}")
    else:
        if verbose:
            print(f"  ⏭ NOOP — 无需操作")
    return operation, result_id or ''


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


# ── 全量同步 ──────────────────────────────────────────────────────────────────────

def sync_effective_to_longterm(
        user_id: str,
        effective_file: str = EFFECTIVE_MEMORIES_PATH,
        db_path: Optional[str] = None,
        prompt_md: str = PROMPT_MD_PATH,
        jsonl_output: Optional[str] = None,
        verbose: bool = True) -> Dict[str, Any]:
    """将 effective_memories 文件全量同步到 Long-term Memory。"""
    print("=" * 70)
    print("Effective Memory → Long-term Memory 同步")
    print("=" * 70)

    db = LongTermMemoryDatabase(db_path) if db_path else LongTermMemoryDatabase()
    prompt_template = load_prompt_template(prompt_md)

    effective_memories = load_effective_memories_from_file(effective_file)
    print(f"\n📂 加载 effective_memories: {len(effective_memories)} 条")

    stats = {'INSERT': 0, 'UPDATE': 0, 'DELETE': 0, 'NOOP': 0, 'INSERT(fallback)': 0, 'NOOP(no target)': 0}
    results = []

    for i, em in enumerate(effective_memories, 1):
        op, mid = process_effective_memory(user_id, em, db, prompt_template, verbose=verbose)
        op_key = op.split('(')[0]
        stats[op_key] = stats.get(op_key, 0) + 1
        results.append({'operation': op, 'memory_id': mid, 'topic': em.get('topic', '')})

    print("\n" + "=" * 70)
    print("✅ 同步完成！")
    print(f"   INSERT: {stats.get('INSERT', 0)}")
    print(f"   UPDATE: {stats.get('UPDATE', 0)}")
    print(f"   DELETE: {stats.get('DELETE', 0)}")
    print(f"   NOOP  : {stats.get('NOOP', 0)}")

    db_stats = db.get_stats(user_id)
    print(f"\n📊 数据库状态 ({user_id}):")
    print(f"   活跃记忆: {db_stats['total_active']} 条")
    print(f"   LongTermMemory: {db_stats['long_term_memory']} 条")
    print(f"   UserMemory: {db_stats['user_memory']} 条")

    # ── 额外导出一份 JSONL ──
    if jsonl_output is None:
        jsonl_path = os.path.join(
            os.path.dirname(db_path) if db_path else BASE_DIR, 'data', 'longterm_memories.jsonl'
        )
    else:
        jsonl_path = jsonl_output
    _export_jsonl(db, jsonl_path, user_id=user_id, results=results)
    print(f"\n💾 同时导出 JSONL → {jsonl_path}")

    return {'stats': stats, 'results': results, 'db_stats': db_stats}


def _export_jsonl(db: LongTermMemoryDatabase, output_path: str, user_id: str = '', results: List[Dict] = None) -> None:
    """
    将 SQLite 中的长期记忆导出为 JSONL 文件
    包含操作类型（INSERT/UPDATE/DELETE）和来源 Effective Memory ID
    """
    memories = db.get_memories_by_user(user_id=user_id)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 构建 operation 映射（从 results 列表）
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
            'created_at': mem.created_at,
            'updated_at': mem.updated_at,
            'is_deleted': getattr(mem, 'is_deleted', 0),
            # 操作信息
            'operation': op_map.get(mem.memory_id, 'UNKNOWN'),
            # 来源 Effective Memory IDs
            'source_effective_ids': getattr(mem, 'source_effective_ids', []),
        }

        # 尝试提取 source_effective_ids（如果存储在某个字段中）
        # 这里我们直接使用 results 中的信息来补充
        export_records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in export_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    print(f"   已导出 {len(export_records)} 条长期记忆到 {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Effective Memory → Long-term Memory 同步')
    parser.add_argument('--user', type=str, default='caroline', help='用户 ID (如 caroline, gina)')
    parser.add_argument('--group', type=str, default=None, help='Group 名称 (如 group_1_gina)，自动推导 user 和 effective 文件路径')
    args = parser.parse_args()

    if args.group:
        # 从 group 名称推导 effective 文件路径
        effective_file = os.path.join(BASE_DIR, 'data', 'groups', args.group, 'effective', 'effective_memories.jsonl')
        if not os.path.exists(effective_file):
            print(f"❌ Effective Memory 文件不存在: {effective_file}")
            print(f"   请先运行: python src/memory/effective_memory/extract_from_session.py --group {args.group}")
            sys.exit(1)
        # 从 group 名称提取用户名（如 group_1_gina -> gina）
        user_id = args.group.split('_')[-1]
        # 输出到 groups/{group}/longterm/ 目录
        jsonl_output = os.path.join(BASE_DIR, 'data', 'groups', args.group, 'longterm', 'longterm_memories.jsonl')
        print(f"\n📂 使用 Group: {args.group} → User: {user_id}")
        print(f"   Effective File: {effective_file}")
        print(f"   Long-term Output: {jsonl_output}")
    else:
        user_id = args.user
        effective_file = EFFECTIVE_MEMORIES_PATH
        jsonl_output = None

    sync_effective_to_longterm(user_id=user_id, effective_file=effective_file, jsonl_output=jsonl_output, verbose=True)

