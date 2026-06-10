"""
三层记忆流水线 (Memory Pipeline)

    Session Memory  ──►  Effective Memory  ──►  Long-term Memory
    (SQLite 对话)         (LLM 抽取结构化)         (持久化语义记忆)

使用方式:

    # 1. 全量初始化（从 JSON 数据导入）
    pipeline = MemoryPipeline(user_id='caroline')
    pipeline.import_from_json('/path/to/group_0_caroline.json')

    # 2. 逐步执行各阶段
    pipeline.run_extract()     # Session → Effective
    pipeline.run_consolidate() # Effective → Long-term (with LLM ops)

    # 3. 基于长期记忆回答问题
    answer = pipeline.answer('What career is Caroline pursuing?')

    # 4. 直接操作长期记忆
    pipeline.add_memory({...})           # 手动 add
    pipeline.update_memory(mid, {...})   # 手动 update
    pipeline.delete_memory(mid)          # 手动 delete
    pipeline.search('counseling')        # 搜索

    # 5. 一键全跑
    pipeline.run_all()
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from call_llm.call_llm_chat import call_llm as _call_llm_unified

# ── 路径常量 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EFFECTIVE_FILE = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')
EXTRACT_PROMPT_MD = os.path.join(BASE_DIR, 'config', 'prompt', 'extract.md')
QA_PROMPT_MD = os.path.join(BASE_DIR, 'config', 'prompt', 'qa.md')
LONGTERM_OPS_PROMPT_MD = os.path.join(BASE_DIR, 'config', 'prompt', 'longterm_ops.md')

# ── 局部导入（延迟，避免循环依赖）─────────────────────────────────────────────

def _get_session_db():
    from src.storage.sqlite.session_memory_db import ConversationDatabase
    return ConversationDatabase()


def _get_longterm_db(db_path: Optional[str] = None):
    from src.storage.sqlite.longterm_memory_db import LongTermMemoryDatabase
    return LongTermMemoryDatabase(db_path) if db_path else LongTermMemoryDatabase()


# ── LLM 工具 ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str = 'LongCat-Flash-Chat-Eco',
              temperature: float = 0.1, max_tokens: int = 4096) -> Optional[str]:
    """调用 LLM，返回文本内容"""
    try:
        return _call_llm_unified(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return None


def _load_prompt(md_path: str) -> str:
    """加载 .md Prompt 文件"""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 文本中提取 JSON 对象"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ── MemoryPipeline 主类 ──────────────────────────────────────────────────────

class MemoryPipeline:
    """
    三层记忆流水线：Session → Effective → Long-term

    Attributes:
        user_id (str): 用户标识
        verbose (bool): 是否打印详细日志
    """

    def __init__(self, user_id: str, verbose: bool = True,
                 longterm_db_path: Optional[str] = None):
        self.user_id = user_id
        self.verbose = verbose
        self._longterm_db = _get_longterm_db(longterm_db_path)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ═══════════════════════════════════════════════════════════
    # Stage 0: 导入 JSON → Session Memory
    # ═══════════════════════════════════════════════════════════

    def import_from_json(self, json_file: str, output_txt: Optional[str] = None) -> Dict[str, int]:
        """
        从 LocoMo/自定义 JSON 文件导入对话到 Session Memory

        Args:
            json_file: 输入 JSON 文件路径
            output_txt: 同步写入的 .txt 文件路径（可选）

        Returns:
            {'success': N, 'error': N}
        """
        from src.storage.sqlite.session_memory_db import add_session_memory

        self._log(f"\n{'='*60}")
        self._log(f"Stage 0: 导入 JSON → Session Memory")
        self._log(f"  文件: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = data.get('data', [])
        self._log(f"  记录数: {len(records)}")

        success, error = 0, 0
        for i, record in enumerate(records, 1):
            try:
                user_id = record.get('userId', self.user_id)
                memory_id = record.get('memoryId', '')
                messages = record.get('messages', [])
                show_ts = messages[0].get('showTimestamp', 0) if messages else 0

                user_content, assistant_content = '', ''
                for msg in messages:
                    role = msg.get('role', '')
                    if role == 'user':
                        user_content = msg.get('content', '')
                    elif role == 'assistant':
                        assistant_content = msg.get('content', '')

                ts_sec = int(show_ts) / 1000
                timestamp = datetime.fromtimestamp(ts_sec).isoformat()

                if user_content or assistant_content:
                    add_session_memory(
                        user_id=user_id,
                        user_content=user_content,
                        assistant_content=assistant_content,
                        timestamp=timestamp,
                        file_path=output_txt,
                        memory_id=memory_id
                    )
                    success += 1
            except Exception as e:
                error += 1
                if self.verbose:
                    print(f"  ⚠️ 第{i}条失败: {e}")

        self._log(f"  ✅ 导入完成: success={success}, error={error}")
        return {'success': success, 'error': error}

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Session Memory → Effective Memory
    # ═══════════════════════════════════════════════════════════

    def run_extract(self,
                    session_file: Optional[str] = None,
                    output_file: str = EFFECTIVE_FILE,
                    time_window_hours: int = 4,
                    append: bool = False) -> int:
        """
        Session Memory → Effective Memory（LLM 抽取）

        Args:
            session_file: session memories .txt 文件路径
            output_file: 输出 effective_memories.txt 路径
            time_window_hours: 合并对话的时间窗口（小时）
            append: True=追加写入，False=覆盖

        Returns:
            成功抽取的条数
        """
        self._log(f"\n{'='*60}")
        self._log(f"Stage 1: Session Memory → Effective Memory")
        self._log(f"  时间窗口: {time_window_hours}h | 输出: {output_file}")

        # 加载 Prompt
        extract_template = _load_prompt(EXTRACT_PROMPT_MD)
        # 提取 suffix（conversation 占位部分）
        suffix_start = extract_template.rfind('## Conversation')
        if suffix_start > 0:
            prompt_template = extract_template[:suffix_start].strip()
            prompt_suffix = "\n\n## Conversation\n\n{{{conversation}}}"
        else:
            prompt_template = extract_template
            prompt_suffix = "\n\nConversation:\n{{{conversation}}}"

        # 加载 session memories
        if session_file is None:
            default_txt = os.path.join(BASE_DIR, 'data', 'caroline_memories.txt')
            session_file = default_txt
        grouped = self._group_sessions_by_time(session_file, time_window_hours)
        self._log(f"  共 {len(grouped)} 个时间窗口")

        # 清空或追加
        if not append and os.path.exists(output_file):
            os.remove(output_file)

        success = 0
        for wid, memories in grouped.items():
            topic_preview = f"[{len(memories)} msgs]"
            self._log(f"  处理 {wid} {topic_preview}...")

            conversation = self._format_conversation(memories)
            full_prompt = prompt_template + prompt_suffix.replace('{{{conversation}}}', conversation)

            response = _call_llm(full_prompt)
            if not response:
                self._log(f"    ❌ LLM 无响应")
                continue

            extracted = _extract_json(response)
            if not extracted:
                self._log(f"    ⚠️ JSON 解析失败")
                continue

            # 附加元数据
            extracted['original_text'] = conversation
            extracted['source_memory_ids'] = [m.get('memory_id', '') for m in memories]
            extracted['time_range'] = {
                'start': memories[0].get('timestamp', ''),
                'end': memories[-1].get('timestamp', '')
            }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(extracted, ensure_ascii=False) + '\n')

            self._log(f"    ✅ {extracted.get('topic', 'N/A')}")
            success += 1

        self._log(f"\n  ✅ 抽取完成: {success}/{len(grouped)} 条")
        return success

    def _group_sessions_by_time(self, file_path: str, window_hours: int) -> Dict[str, List[Dict]]:
        """按时间窗口分组 session memories"""
        all_memories = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    mem = json.loads(line.strip())
                    mem['_ts'] = datetime.fromisoformat(mem['timestamp'])
                    all_memories.append(mem)
                except Exception:
                    continue

        all_memories.sort(key=lambda x: x['timestamp'])
        if not all_memories:
            return {}

        grouped = {}
        window_start = all_memories[0]['_ts']
        current = []
        wid = 0

        for mem in all_memories:
            if mem['_ts'] - window_start <= timedelta(hours=window_hours):
                current.append(mem)
            else:
                if current:
                    grouped[f'window_{wid}'] = current
                    wid += 1
                window_start = mem['_ts']
                current = [mem]

        if current:
            grouped[f'window_{wid}'] = current

        return grouped

    def _format_conversation(self, memories: List[Dict]) -> str:
        """格式化对话"""
        lines = []
        for m in memories:
            if m.get('user_content'):
                lines.append(f"User: {m['user_content']}")
            if m.get('assistant_content'):
                lines.append(f"Assistant: {m['assistant_content']}")
        return '\n'.join(lines)

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Effective Memory → Long-term Memory
    # ═══════════════════════════════════════════════════════════

    def run_consolidate(self, effective_file: str = EFFECTIVE_FILE) -> Dict[str, Any]:
        """
        Effective Memory → Long-term Memory（LLM 判断 INSERT/UPDATE/DELETE/NOOP）

        Args:
            effective_file: effective_memories.txt 路径

        Returns:
            统计结果字典
        """
        self._log(f"\n{'='*60}")
        self._log(f"Stage 2: Effective Memory → Long-term Memory")

        from src.memory.effective_memory.effective_to_longterm import (
            sync_effective_to_longterm
        )
        result = sync_effective_to_longterm(
            user_id=self.user_id,
            effective_file=effective_file,
            prompt_md=LONGTERM_OPS_PROMPT_MD,
            verbose=self.verbose
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 长期记忆 CRUD 接口
    # ═══════════════════════════════════════════════════════════

    def add_memory(self, effective_memory: Dict[str, Any]) -> str:
        """
        手动添加长期记忆

        Args:
            effective_memory: 包含 topic/summary/facts/memory_type/tags 等字段的字典

        Returns:
            memory_id
        """
        mid = self._longterm_db.add_memory_from_dict(self.user_id, effective_memory)
        self._log(f"  ✅ 添加长期记忆: {effective_memory.get('topic', '')} → {mid[:8]}...")
        return mid

    def update_memory(self, memory_id: str,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None,
                      facts: Optional[List[str]] = None,
                      tags: Optional[List[str]] = None,
                      confidence: Optional[float] = None) -> bool:
        """
        更新指定长期记忆

        Args:
            memory_id: 记忆 ID
            topic/summary/facts/tags/confidence: 要更新的字段（None 表示不更新）

        Returns:
            是否成功
        """
        ok = self._longterm_db.update_memory(
            memory_id=memory_id,
            topic=topic, summary=summary,
            facts=facts, tags=tags,
            confidence=confidence
        )
        self._log(f"  {'✅' if ok else '❌'} 更新记忆 {memory_id[:8]}...")
        return ok

    def delete_memory(self, memory_id: str, hard_delete: bool = False) -> bool:
        """
        删除指定长期记忆（默认软删除）

        Args:
            memory_id: 记忆 ID
            hard_delete: True=物理删除

        Returns:
            是否成功
        """
        ok = self._longterm_db.delete_memory(memory_id, hard_delete=hard_delete)
        label = "物理删除" if hard_delete else "软删除"
        self._log(f"  {'✅' if ok else '❌'} {label}记忆 {memory_id[:8]}...")
        return ok

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索长期记忆

        Args:
            query: 搜索词
            top_k: 返回数量

        Returns:
            记忆字典列表
        """
        results = self._longterm_db.search_memories(self.user_id, query, top_k)
        return [r.to_dict() for r in results]

    def list_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """列出用户所有长期记忆"""
        mems = self._longterm_db.get_memories_by_user(self.user_id, limit=limit)
        return [m.to_dict() for m in mems]

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆库统计"""
        return self._longterm_db.get_stats(self.user_id)

    # ═══════════════════════════════════════════════════════════
    # Stage 3: QA — 基于长期记忆回答问题
    # ═══════════════════════════════════════════════════════════

    def answer(self, question: str, top_k: int = 5,
               fields: Optional[List[str]] = None) -> str:
        """
        基于长期记忆回答问题

        Args:
            question: 问题
            top_k: 检索的记忆数量
            fields: 搜索字段

        Returns:
            答案字符串
        """
        self._log(f"\n{'='*60}")
        self._log(f"QA: {question}")

        # 检索相关记忆
        related = self._longterm_db.search_memories(self.user_id, question, top_k, fields)
        self._log(f"  检索到 {len(related)} 条相关记忆")

        if not related:
            return "The memories do not contain information about this."

        # 构建记忆上下文
        memory_blocks = []
        for i, mem in enumerate(related, 1):
            block = (
                f"[Memory {i}]\n"
                f"Topic: {mem.topic}\n"
                f"Summary: {mem.summary}\n"
                f"Facts:\n" + '\n'.join(f"  - {f}" for f in mem.facts)
            )
            memory_blocks.append(block)
        memories_str = '\n\n'.join(memory_blocks)

        # 加载 QA Prompt 并填充
        qa_template = _load_prompt(QA_PROMPT_MD)
        prompt = qa_template.replace('{{{memories}}}', memories_str)
        prompt = prompt.replace('{{{question}}}', question)

        answer_text = _call_llm(prompt, max_tokens=512)
        if not answer_text:
            return "Failed to generate answer."

        self._log(f"  💬 {answer_text.strip()}")
        return answer_text.strip()

    # ═══════════════════════════════════════════════════════════
    # 全流程一键运行
    # ═══════════════════════════════════════════════════════════

    def run_all(self,
                json_file: Optional[str] = None,
                session_txt: Optional[str] = None,
                effective_file: str = EFFECTIVE_FILE,
                time_window_hours: int = 4) -> Dict[str, Any]:
        """
        全流程一键执行：
        (可选) JSON导入 → Session → Effective → Long-term

        Args:
            json_file: 原始 JSON 文件（若已导入可 None）
            session_txt: session memories .txt 文件
            effective_file: effective memories 输出路径
            time_window_hours: 时间窗口

        Returns:
            各阶段统计
        """
        report = {'user_id': self.user_id, 'stages': {}}

        # Stage 0: 导入
        if json_file:
            r0 = self.import_from_json(json_file, session_txt)
            report['stages']['import'] = r0

        # Stage 1: 抽取
        n = self.run_extract(
            session_file=session_txt,
            output_file=effective_file,
            time_window_hours=time_window_hours
        )
        report['stages']['extract'] = {'effective_count': n}

        # Stage 2: 固化
        r2 = self.run_consolidate(effective_file)
        report['stages']['consolidate'] = r2

        # 统计
        stats = self.get_stats()
        report['final_stats'] = stats

        self._log(f"\n{'='*60}")
        self._log(f"🏁 全流程完成！用户 {self.user_id} 长期记忆: {stats['total_active']} 条")

        return report


# ── 命令行入口 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # 示例用法
    pipeline = MemoryPipeline(user_id='caroline', verbose=True)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == 'extract':
            # python -m src.memory.pipeline extract
            pipeline.run_extract()

        elif cmd == 'consolidate':
            # python -m src.memory.pipeline consolidate
            pipeline.run_consolidate()

        elif cmd == 'qa':
            # python -m src.memory.pipeline qa "What career is Caroline pursuing?"
            q = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else "What career is Caroline pursuing?"
            ans = pipeline.answer(q)
            print(f"\nAnswer: {ans}")

        elif cmd == 'stats':
            stats = pipeline.get_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))

        elif cmd == 'list':
            mems = pipeline.list_memories()
            for m in mems:
                print(f"[{m['memory_id'][:8]}] {m['topic']}")

        elif cmd == 'all':
            json_file = sys.argv[2] if len(sys.argv) > 2 else None
            pipeline.run_all(json_file=json_file)

    else:
        # 默认：演示 QA
        print("使用方式: python -m src.memory.pipeline [extract|consolidate|qa|stats|list|all]")
        q = "What fields would Caroline be likely to pursue in her education?"
        ans = pipeline.answer(q)
        print(f"\nQ: {q}\nA: {ans}")

