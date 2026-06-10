"""
Effective Memory → Long-term Memory 流转器

职责：
1. 读取 effective_memories.txt（或逐条接收）
2. 对每条 effective_memory，检索已有长期记忆中相似条目
3. 调用 LLM 判断应该执行 INSERT / UPDATE / DELETE / NOOP
4. 执行对应操作写入 LongTermMemoryDatabase

Prompt 模板从 config/prompt/longterm_ops.md 加载。
"""
import json
import os
import re
import sys
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from call_llm.call_llm_chat import call_llm as _call_llm_unified

from src.storage.sqlite.longterm_memory_db import LongTermMemoryDatabase, LongTermMemory

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROMPT_MD_PATH = os.path.join(BASE_DIR, 'config', 'prompt', 'longterm_ops.md')
EFFECTIVE_MEMORIES_PATH = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')


# ── Prompt 加载 ───────────────────────────────────────────────────────────────

def load_prompt_template(md_path: str = PROMPT_MD_PATH) -> str:
    """从 .md 文件加载 Prompt 模板"""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()


# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = 'LongCat-Flash-Chat-Eco',
             temperature: float = 0.1, max_tokens: int = 1024) -> Optional[str]:
    """调用 LLM，返回文本内容"""
    try:
        return _call_llm_unified(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return None


def parse_llm_decision(response: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的操作决策 JSON

    期望格式：
    {
      "operation": "INSERT" | "UPDATE" | "DELETE" | "NOOP",
      "target_memory_id": "<existing_id_or_null>",
      "reason": "<简短理由>"
    }
    """
    # 尝试提取 JSON 块
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 降级：从文本中推断操作类型
    text = response.upper()
    if 'INSERT' in text:
        return {'operation': 'INSERT', 'target_memory_id': None, 'reason': 'inferred from text'}
    elif 'UPDATE' in text:
        return {'operation': 'UPDATE', 'target_memory_id': None, 'reason': 'inferred from text'}
    elif 'DELETE' in text:
        return {'operation': 'DELETE', 'target_memory_id': None, 'reason': 'inferred from text'}
    else:
        return {'operation': 'NOOP', 'target_memory_id': None, 'reason': 'fallback NOOP'}


# ── 核心决策函数 ──────────────────────────────────────────────────────────────

def decide_operation(effective_memory: Dict[str, Any],
                     existing_memories: List[LongTermMemory],
                     prompt_template: str) -> Dict[str, Any]:
    """
    调用 LLM 决定对新的 effective_memory 执行什么操作

    Args:
        effective_memory: 新的 effective_memory 字典
        existing_memories: 当前用户已有的长期记忆列表（最多 10 条最相关）
        prompt_template: 从 longterm_ops.md 加载的 Prompt 模板

    Returns:
        {'operation': 'INSERT'|'UPDATE'|'DELETE'|'NOOP',
         'target_memory_id': <id or None>,
         'reason': <str>}
    """
    # 构建已有记忆摘要
    existing_str = ""
    if existing_memories:
        lines = []
        for i, mem in enumerate(existing_memories, 1):
            lines.append(
                f"[{i}] ID={mem.memory_id}\n"
                f"    Topic: {mem.topic}\n"
                f"    Summary: {mem.summary[:200]}...\n"
                f"    Facts: {'; '.join(mem.facts[:3])}"
            )
        existing_str = "\n".join(lines)
    else:
        existing_str = "(no existing memories)"

    # 构建新记忆描述
    new_memory_str = json.dumps({
        "topic": effective_memory.get("topic", ""),
        "summary": effective_memory.get("summary", ""),
        "facts": effective_memory.get("facts", [])[:5],
        "memory_type": effective_memory.get("memory_type", ""),
        "tags": effective_memory.get("tags", []),
    }, ensure_ascii=False, indent=2)

    # 填充 Prompt 模板
    prompt = prompt_template.replace('{{{existing_memories}}}', existing_str)
    prompt = prompt.replace('{{{new_memory}}}', new_memory_str)

    # 调用 LLM
    response = call_llm(prompt)
    if not response:
        return {'operation': 'INSERT', 'target_memory_id': None, 'reason': 'LLM failed, fallback INSERT'}

    return parse_llm_decision(response)


# ── 主流转函数 ────────────────────────────────────────────────────────────────

def process_effective_memory(user_id: str,
                              effective_memory: Dict[str, Any],
                              db: LongTermMemoryDatabase,
                              prompt_template: str,
                              verbose: bool = True) -> Tuple[str, str]:
    """
    处理单条 effective_memory，决策并写入 Long-term Memory

    Args:
        user_id: 用户 ID
        effective_memory: effective_memory 字典
        db: LongTermMemoryDatabase 实例
        prompt_template: 操作决策 Prompt 模板
        verbose: 是否打印详情

    Returns:
        (operation, memory_id) — 执行的操作和影响的 memory_id
    """
    topic = effective_memory.get('topic', 'unknown')

    if verbose:
        print(f"\n  📥 处理: {topic}")

    # 1. 检索相关的已有记忆（top-5）
    related = db.search_memories(user_id, query=topic, top_k=5)

    # 2. LLM 决策
    decision = decide_operation(effective_memory, related, prompt_template)
    operation = decision.get('operation', 'INSERT').upper()
    target_id = decision.get('target_memory_id')
    reason = decision.get('reason', '')

    if verbose:
        print(f"  🤖 决策: {operation} | 原因: {reason}")

    # 3. 执行操作
    result_id = None

    if operation == 'INSERT':
        result_id = db.add_memory_from_dict(user_id, effective_memory)
        if verbose:
            print(f"  ✅ INSERT → memory_id={result_id[:8]}...")

    elif operation == 'UPDATE':
        # 若 LLM 给出了目标 ID 且有效，直接 merge；否则找最相似的
        if target_id and db.get_memory_by_id(target_id):
            success = db.merge_and_update(target_id, effective_memory)
            result_id = target_id if success else None
        else:
            similar = db.find_similar_by_topic(user_id, topic, threshold=0.2)
            if similar:
                success = db.merge_and_update(similar.memory_id, effective_memory)
                result_id = similar.memory_id if success else None
            else:
                # 找不到可合并目标，回退 INSERT
                result_id = db.add_memory_from_dict(user_id, effective_memory)
                operation = 'INSERT(fallback)'

        if verbose:
            print(f"  ✅ UPDATE → memory_id={result_id[:8] if result_id else 'N/A'}...")

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
            print(f"  🗑️ DELETE → memory_id={result_id[:8] if result_id else 'N/A'}...")

    else:  # NOOP
        if verbose:
            print(f"  ⏭️ NOOP — 无需操作")

    return operation, result_id or ''


def load_effective_memories_from_file(file_path: str = EFFECTIVE_MEMORIES_PATH) -> List[Dict[str, Any]]:
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


def sync_effective_to_longterm(
        user_id: str,
        effective_file: str = EFFECTIVE_MEMORIES_PATH,
        db_path: Optional[str] = None,
        prompt_md: str = PROMPT_MD_PATH,
        verbose: bool = True) -> Dict[str, Any]:
    """
    将 effective_memories 文件全量同步到 Long-term Memory

    Args:
        user_id: 用户 ID
        effective_file: effective_memories.txt 路径
        db_path: 数据库路径（None=使用默认）
        prompt_md: longterm_ops.md 路径
        verbose: 是否打印详情

    Returns:
        统计结果字典
    """
    print("=" * 70)
    print("Effective Memory → Long-term Memory 同步")
    print("=" * 70)

    # 初始化
    db = LongTermMemoryDatabase(db_path) if db_path else LongTermMemoryDatabase()
    prompt_template = load_prompt_template(prompt_md)

    # 加载 effective_memories
    effective_memories = load_effective_memories_from_file(effective_file)
    print(f"\n📂 加载 effective_memories: {len(effective_memories)} 条")

    # 统计
    stats = {'INSERT': 0, 'UPDATE': 0, 'DELETE': 0, 'NOOP': 0, 'INSERT(fallback)': 0, 'NOOP(no target)': 0}
    results = []

    for i, em in enumerate(effective_memories, 1):
        if verbose:
            print(f"\n[{i}/{len(effective_memories)}]", end='')

        op, mid = process_effective_memory(user_id, em, db, prompt_template, verbose=verbose)
        op_key = op.split('(')[0]  # 归一化为 INSERT/UPDATE/DELETE/NOOP
        stats[op_key] = stats.get(op_key, 0) + 1
        results.append({'operation': op, 'memory_id': mid, 'topic': em.get('topic', '')})

    # 打印汇总
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

    return {'stats': stats, 'results': results, 'db_stats': db_stats}


if __name__ == '__main__':
    result = sync_effective_to_longterm(
        user_id='caroline',
        verbose=True
    )

