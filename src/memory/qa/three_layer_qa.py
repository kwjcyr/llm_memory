"""
三层记忆问答系统（Long-term → Effective → Session 降级策略）

核心特性：
1. 优先使用 Long-term Memory（高度提炼的知识）
2. 降级到 Effective Memory（中等粒度的摘要）
3. 最终降级到 Session Memory 原文（最详细的时间信息）
4. 保留原始时间戳（对时间推理问题至关重要）
5. 支持按 Group/User 加载数据

使用方法：
    from src.memory.qa.three_layer_qa import ThreeLayerQA

    qa = ThreeLayerQA(group='group_1_gina')
    answer = qa.answer("When did Gina start her clothing store?")

    # 返回：
    # {
    #   'answer': 'Gina launched an ad campaign for her clothing store on January 29, 2023...',
    #   'evidence': {
    #     'longterm': [...],      # 来自 Long-term Memory
    #     'effective': [...],     # 来自 Effective Memory
    #     'session': [...]        # 来自 Session Memory 原文
    #   },
    #   'retrieval_stats': {
    #     'longterm': 3,
    #     'effective': 5,
    #     'session': 10,
    #     'total_facts_used': 18
    #   }
    # }
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ── 路径配置 ──────────────────────────────────────────────────────────────
BASE_DIR = str(Path(__file__).resolve().parents[3])  # qa → memory → src → 项目根
sys.path.insert(0, BASE_DIR)

from src.call_llm.call_llm_chat import call_llm as _call_llm_unified
from src.storage.sqlite.longterm_memory_db import LongTermMemoryDatabase


@dataclass
class EvidenceItem:
    """证据条目"""
    layer: str                    # 'longterm' | 'effective' | 'session'
    content: str                  # 主要内容（topic/summary/text）
    details: Dict[str, Any]       # 详细信息（facts/tags/time等）
    score: float                  # 检索相关度分数
    source_id: str                # 来源 ID（memory_id/session_id）
    timestamp: Optional[str] = None  # 时间戳


@dataclass
class QAResult:
    """QA 结果"""
    answer: str
    evidence: Dict[str, List[EvidenceItem]]  # 按层组织的证据
    retrieval_stats: Dict[str, int]          # 检索统计
    confidence: float                        # 置信度（基于检索到的信息量）


class ThreeLayerQA:
    """
    三层记忆问答系统

    检索优先级：
    1. Long-term Memory（SQLite）- 高度提炼的长期知识
    2. Effective Memory（JSONL）- 中等粒度的对话摘要
    3. Session Memory（JSONL）- 原始对话原文（含精确时间）

    降级策略：
    - 如果 Long-term 找到足够证据（score > threshold），直接使用
    - 否则补充 Effective Memory 的信息
    - 如果还不足，使用 Session Memory 原文补充细节和时间
    """

    def __init__(self, group: str = None, user_id: str = None, db_path: str = None):
        """
        初始化 QA 系统

        Args:
            group: Group 名称（如 group_1_gina），自动推导路径
            user_id: 用户 ID（如 gina）
            db_path: SQLite 数据库路径
        """
        self.group = group
        self.user_id = user_id or (group.split('_')[-1] if group else 'caroline')
        self.db_path = db_path or os.path.join(BASE_DIR, 'data', 'longterm_memories.db')

        # 推导文件路径
        if group:
            base_dir = os.path.join(BASE_DIR, 'data', 'groups', group)
            self.effective_file = os.path.join(base_dir, 'effective', 'effective_memories.jsonl')
            self.session_file = os.path.join(base_dir, 'session', 'memories.jsonl')
        else:
            self.effective_file = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')
            self.session_file = os.path.join(BASE_DIR, 'data', 'caroline_memories.txt')

        # 初始化数据库
        self.db = LongTermMemoryDatabase(self.db_path)

        # 缓存（避免重复加载）
        self._effective_cache = None
        self._session_cache = None
        self._session_map = None  # session_id -> session dict

    def _load_effective_memories(self) -> List[Dict[str, Any]]:
        """加载并缓存 Effective Memories"""
        if self._effective_cache is not None:
            return self._effective_cache

        memories = []
        if os.path.exists(self.effective_file):
            with open(self.effective_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        memories.append(json.loads(line))
                    except Exception as e:
                        print(f"⚠️ 解析 effective memory 失败: {e}")

        self._effective_cache = memories
        return memories

    def _load_session_memories(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
        """加载并缓存 Session Memories，返回 (列表, ID映射)"""
        if self._session_cache is not None:
            return self._session_cache, self._session_map

        memories = []
        mem_map = {}

        if os.path.exists(self.session_file):
            with open(self.session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        mem = json.loads(line)
                        memories.append(mem)
                        # 构建 ID 映射
                        mem_id = str(mem.get('memory_id', ''))
                        if mem_id:
                            mem_map[mem_id] = mem
                    except Exception as e:
                        print(f"⚠️ 解析 session memory 失败: {e}")

        self._session_cache = memories
        self._session_map = mem_map
        return memories, mem_map

    def _search_longterm(self, query: str, top_k: int = 5) -> List[EvidenceItem]:
        """
        从 Long-term Memory 检索相关记忆

        Returns:
            EvidenceItem 列表
        """
        memories = self.db.search_memories(
            user_id=self.user_id,
            query=query,
            top_k=top_k,
            fields=['topic', 'summary', 'facts', 'tags']
        )

        evidence = []
        for mem in memories:
            item = EvidenceItem(
                layer='longterm',
                content=mem.topic,
                details={
                    'summary': mem.summary,
                    'facts': mem.facts,
                    'tags': mem.tags,
                    'confidence': mem.confidence,
                    'time_range_start': mem.time_range_start,
                    'time_range_end': mem.time_range_end,
                },
                score=0.0,  # 稍后计算
                source_id=mem.memory_id,
                timestamp=mem.time_range_start
            )
            evidence.append(item)

        return evidence

    def _search_effective(self, query: str, top_k: int = 5) -> List[EvidenceItem]:
        """
        从 Effective Memory 检索相关记忆

        Returns:
            EvidenceItem 列表
        """
        memories = self._load_effective_memories()
        keywords = set(query.lower().split())

        scored = []
        for mem in memories:
            score = 0.0

            # 多字段评分
            topic = mem.get('topic', '').lower()
            summary = mem.get('summary', '').lower()
            facts_text = ' '.join(mem.get('facts', [])).lower()
            tags_text = ' '.join(mem.get('tags', [])).lower()

            if any(kw in topic for kw in keywords): score += 3
            if any(kw in summary for kw in keywords): score += 2
            if any(kw in facts_text for kw in keywords): score += 2
            if any(kw in tags_text for kw in keywords): score += 1

            if score > 0:
                time_range = mem.get('time_range', {})
                item = EvidenceItem(
                    layer='effective',
                    content=mem.get('topic', ''),
                    details={
                        'summary': mem.get('summary', ''),
                        'facts': mem.get('facts', []),
                        'tags': mem.get('tags', []),
                        'memory_type': mem.get('memory_type', ''),
                        'confidence': mem.get('confidence'),
                        'source_memory_ids': mem.get('source_memory_ids', []),
                    },
                    score=score,
                    source_id=mem.get('effective_id', ''),
                    timestamp=time_range.get('start') if time_range else None
                )
                scored.append((score, item))

        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _search_session(self, query: str, top_k: int = 10,
                       source_ids: List[str] = None) -> List[EvidenceItem]:
        """
        从 Session Memory 检索相关原文

        Args:
            query: 搜索查询
            top_k: 返回数量
            source_ids: 可选，只检索特定 ID 的 session（来自 effective 的 source_memory_ids）

        Returns:
            EvidenceItem 列表（包含原始对话文本和精确时间戳）
        """
        memories, mem_map = self._load_session_memories()
        keywords = set(query.lower().split())

        # 如果指定了 source_ids，优先搜索这些
        if source_ids:
            target_memories = [mem_map.get(sid) for sid in source_ids if sid in mem_map]
            target_memories = [m for m in target_memories if m]  # 过滤 None
        else:
            target_memories = memories

        scored = []
        for mem in target_memories:
            score = 0.0

            # 在多个字段中搜索
            text = mem.get('text', '').lower() or mem.get('content', '').lower()

            if any(kw in text for kw in keywords):
                score += 2

            # 时间信息也参与匹配（如果查询包含时间词）
            time_keywords = {'when', 'what time', 'date', 'which day', 'before', 'after'}
            if keywords & time_keywords:
                timestamp = mem.get('timestamp', '') or ''
                if timestamp:
                    score += 1  # 有时间戳的记录加分

            if score > 0:
                item = EvidenceItem(
                    layer='session',
                    content=text[:200] + ('...' if len(text) > 200 else ''),
                    details={
                        'full_text': text,
                        'timestamp': mem.get('timestamp', ''),
                        'user_id': mem.get('user_id', ''),
                        'speaker': mem.get('speaker', ''),
                        'message_type': mem.get('message_type', ''),
                    },
                    score=score,
                    source_id=str(mem.get('memory_id', '')),
                    timestamp=mem.get('timestamp', '')
                )
                scored.append((score, item))

        # 按分数排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def retrieve_evidence(self, question: str,
                         longterm_k: int = 5,
                         effective_k: int = 5,
                         session_k: int = 10) -> Dict[str, List[EvidenceItem]]:
        """
        三层降级检索策略

        策略：
        1. 先从 Long-term 检索（高质量知识）
        2. 从 Effective 补充（中等粒度）
        3. 从 Session 原文补充细节和时间

        Args:
            question: 用户问题
            longterm_k: Long-term 检索数量
            effective_k: Effective 检索数量
            session_k: Session 检索数量

        Returns:
            {'longterm': [...], 'effective': [...], 'session': [...]}
        """
        evidence = {
            'longterm': [],
            'effective': [],
            'session': []
        }

        # Layer 1: Long-term Memory
        print(f"🔍 [Layer 1/3] 搜索 Long-term Memory...")
        lt_evidence = self._search_longterm(question, top_k=longterm_k)
        evidence['longterm'] = lt_evidence
        print(f"   ✅ 找到 {len(lt_evidence)} 条 Long-term 记忆")

        # Layer 2: Effective Memory
        print(f"🔍 [Layer 2/3] 搜索 Effective Memory...")
        eff_evidence = self._search_effective(question, top_k=effective_k)
        evidence['effective'] = eff_evidence
        print(f"   ✅ 找到 {len(eff_evidence)} 条 Effective 记忆")

        # Layer 3: Session Memory（从 effective 的 source_ids 定位相关原文）
        print(f"🔍 [Layer 3/3] 搜索 Session Memory 原文...")

        # 收集所有 effective 的 source_memory_ids
        all_source_ids = []
        for eff_item in eff_evidence:
            src_ids = eff_item.details.get('source_memory_ids', [])
            all_source_ids.extend(src_ids)

        # 去重
        unique_source_ids = list(set(all_source_ids))[:50]  # 限制数量避免过多

        # 检索相关的 session 原文
        if unique_source_ids:
            session_evidence = self._search_session(
                question,
                top_k=session_k,
                source_ids=unique_source_ids
            )
        else:
            # 如果没有 source_ids，全局搜索
            session_evidence = self._search_session(question, top_k=session_k)

        evidence['session'] = session_evidence
        print(f"   ✅ 找到 {len(session_evidence)} 条 Session 记忆")

        return evidence

    def build_context(self, evidence: Dict[str, List[EvidenceItem]],
                     max_tokens: int = 8000) -> str:
        """
        构建给 LLM 的上下文，按层组织，突出时间信息

        Args:
            evidence: 三层证据
            max_tokens: 最大 token 数限制

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        # Layer 1: Long-term Memory（最精炼的知识）
        if evidence['longterm']:
            context_parts.append("\n=== 长期记忆（已验证的事实） ===")
            for i, item in enumerate(evidence['longterm'], 1):
                time_str = f" ⏰ {item.timestamp}" if item.timestamp else ""
                context_parts.append(
                    f"\n[{i}]{time_str}\n"
                    f"主题: {item.content}\n"
                    f"摘要: {item.details.get('summary', 'N/A')}\n"
                    f"事实: {'; '.join(item.details.get('facts', []))}"
                )

        # Layer 2: Effective Memory（对话摘要）
        if evidence['effective']:
            context_parts.append("\n\n=== 有效记忆（对话摘要） ===")
            for i, item in enumerate(evidence['effective'], 1):
                time_str = f" ⏰ {item.timestamp}" if item.timestamp else ""
                context_parts.append(
                    f"\n[{i}]{time_str}\n"
                    f"主题: {item.content}\n"
                    f"摘要: {item.details.get('summary', 'N/A')}\n"
                    f"关键事实: {'; '.join(item.details.get('facts', [])[:5])}"
                )

        # Layer 3: Session Memory（原始对话，含精确时间）
        if evidence['session']:
            context_parts.append("\n\n=== 会话记录（原始对话与精确时间） ===")
            for i, item in enumerate(evidence['session'][:10], 1):  # 限制数量
                time_str = f" ⏰ {item.timestamp}" if item.timestamp else ""
                full_text = item.details.get('full_text', item.content)
                # 截断过长的文本
                display_text = full_text[:300] + ('...' if len(full_text) > 300 else '')

                speaker = item.details.get('speaker', '')
                speaker_str = f"[{speaker}] " if speaker else ""

                context_parts.append(
                    f"\n[{i}]{time_str}{speaker_str}\n"
                    f"{display_text}"
                )

        return '\n'.join(context_parts)

    def call_llm_for_answer(self, question: str, context: str) -> str:
        """
        调用 LLM 基于三层上下文回答问题

        Prompt 设计要点：
        - 强调时间信息的准确性
        - 要求引用来源层
        - 支持时间推理
        """
        prompt = f"""你是一个智能问答助手，需要根据提供的三层记忆信息准确回答用户的问题。

## 可用的记忆层次：

{context}

## 用户问题：
{question}

## 回答要求：

1. **准确性优先**：严格基于提供的记忆信息回答，不要编造
2. **时间敏感**：如果问题涉及时间（when/before/after/how long），必须给出具体日期或时间段
3. **引用来源**：如果可能，说明答案来自哪个记忆层次（长期记忆/有效记忆/会话记录）
4. **简洁明了**：直接回答，不要冗余解释
5. **不确定性处理**：如果记忆中没有足够信息，明确说明"根据现有记忆无法确定"

## 特别注意：
- 会话记录中的时间戳是最精确的原始数据
- 长期记忆是经过验证的高度概括事实
- 有效记忆是中间层的对话摘要

请直接给出答案："""

        try:
            return _call_llm_unified(prompt, temperature=0.1, max_tokens=1024)
        except Exception as e:
            return f"❌ LLM 调用失败: {e}"

    def answer(self, question: str,
              longterm_k: int = 5,
              effective_k: int = 5,
              session_k: int = 10,
              verbose: bool = True) -> QAResult:
        """
        主函数：使用三层记忆回答问题

        Args:
            question: 用户问题
            longterm_k: Long-term 检索数量
            effective_k: Effective 检索数量
            session_k: Session 检索数量
            verbose: 是否打印详细信息

        Returns:
            QAResult 对象
        """
        if verbose:
            print("=" * 80)
            print(f"❓ 问题: {question}")
            print(f"👤 用户: {self.user_id} | 📁 Group: {self.group or 'default'}")
            print("=" * 80)

        # Step 1: 三层检索
        evidence = self.retrieve_evidence(
            question,
            longterm_k=longterm_k,
            effective_k=effective_k,
            session_k=session_k
        )

        # 统计检索结果
        total_evidence = (
            len(evidence['longterm']) +
            len(evidence['effective']) +
            len(evidence['session'])
        )

        stats = {
            'longterm': len(evidence['longterm']),
            'effective': len(evidence['effective']),
            'session': len(evidence['session']),
            'total': total_evidence
        }

        if verbose:
            print(f"\n📊 检索统计: {stats}")

        # Step 2: 构建上下文
        context = self.build_context(evidence)

        if verbose:
            print(f"\n📝 构建上下文完成 ({len(context)} 字符)")

        # Step 3: 调用 LLM 回答
        if verbose:
            print(f"🤖 调用 LLM 生成答案...")

        answer = self.call_llm_for_answer(question, context)

        # Step 4: 计算置信度（基于检索到的信息量）
        confidence = min(1.0, total_evidence / 15.0)  # 简单启发式

        result = QAResult(
            answer=answer,
            evidence=evidence,
            retrieval_stats=stats,
            confidence=confidence
        )

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"💡 答案: {answer}")
            print(f"📈 置信度: {confidence:.2f}")
            print(f"{'=' * 80}")

        return result


# ── 便捷函数 ────────────────────────────────────────────────────────────────

def answer_with_three_layers(question: str,
                            group: str = 'group_1_gina',
                            verbose: bool = True) -> QAResult:
    """
    便捷接口：使用三层记忆回答问题

    Args:
        question: 用户问题
        group: Group 名称
        verbose: 是否打印详情

    Returns:
        QAResult 对象
    """
    qa = ThreeLayerQA(group=group)
    return qa.answer(question, verbose=verbose)


if __name__ == '__main__':
    # 测试示例
    test_questions = [
        "When did Gina start her clothing store?",
        "What happened to Jon's banking job?",
        "Did Jon and Gina meet before or after Jon lost his job?",
    ]

    for q in test_questions:
        result = answer_with_three_layers(q, group='group_1_gina')
        print("\n" + "="*80 + "\n")

