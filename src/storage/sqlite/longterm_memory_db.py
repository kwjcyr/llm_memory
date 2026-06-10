"""
Long-term Memory Database
基于 SQLite 的长期记忆存储，支持 Add / Update / Delete / Search 操作。

每条长期记忆由 effective_memory 经 LLM 判断后写入，包含：
- memory_id: UUID
- user_id: 用户标识
- topic: 主题（5-10词）
- summary: 摘要叙述
- facts: JSON 数组，具体事实
- memory_type: LongTermMemory | UserMemory
- tags: JSON 数组，关键词
- confidence: 置信度 0-1
- ttl_days: 过期天数（null=永久）
- source_effective_ids: 来源 effective memory id 列表
- time_range_start / time_range_end: 记忆时间跨度
- created_at / updated_at: 系统时间
- is_deleted: 软删除标志
"""
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'data',
    'longterm_memories.db'
)


@dataclass
class LongTermMemory:
    """长期记忆数据模型"""
    memory_id: str
    user_id: str
    topic: str
    summary: str
    facts: List[str]
    memory_type: str          # LongTermMemory | UserMemory
    tags: List[str]
    confidence: float
    ttl_days: Optional[int]
    source_effective_ids: List[str]
    time_range_start: Optional[str]
    time_range_end: Optional[str]
    created_at: str
    updated_at: str
    is_deleted: int = 0       # 0=正常, 1=已删除

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'LongTermMemory':
        return cls(
            memory_id=row['memory_id'],
            user_id=row['user_id'],
            topic=row['topic'],
            summary=row['summary'],
            facts=json.loads(row['facts']),
            memory_type=row['memory_type'],
            tags=json.loads(row['tags']),
            confidence=row['confidence'],
            ttl_days=row['ttl_days'],
            source_effective_ids=json.loads(row['source_effective_ids']),
            time_range_start=row['time_range_start'],
            time_range_end=row['time_range_end'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            is_deleted=row['is_deleted'],
        )

    def to_text(self) -> str:
        """拼接成可检索的自然语言文本"""
        parts = [f"Topic: {self.topic}", f"Summary: {self.summary}"]
        if self.facts:
            parts.append("Facts: " + "; ".join(self.facts))
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        return "\n".join(parts)


class LongTermMemoryDatabase:
    """长期记忆数据库操作类"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS longterm_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    facts TEXT NOT NULL DEFAULT '[]',
                    memory_type TEXT NOT NULL DEFAULT 'LongTermMemory',
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    ttl_days INTEGER,
                    source_effective_ids TEXT NOT NULL DEFAULT '[]',
                    time_range_start TEXT,
                    time_range_end TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_deleted INTEGER NOT NULL DEFAULT 0
                )
            ''')
            # 索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lt_user_id ON longterm_memories(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lt_memory_type ON longterm_memories(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lt_is_deleted ON longterm_memories(is_deleted)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_lt_created_at ON longterm_memories(created_at)')

    # ============== Add (增) ==============

    def add_memory(self,
                   user_id: str,
                   topic: str,
                   summary: str,
                   facts: List[str],
                   memory_type: str = 'LongTermMemory',
                   tags: Optional[List[str]] = None,
                   confidence: float = 1.0,
                   ttl_days: Optional[int] = None,
                   source_effective_ids: Optional[List[str]] = None,
                   time_range_start: Optional[str] = None,
                   time_range_end: Optional[str] = None,
                   memory_id: Optional[str] = None) -> str:
        """
        添加一条长期记忆

        Returns:
            memory_id
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO longterm_memories
                (memory_id, user_id, topic, summary, facts, memory_type, tags,
                 confidence, ttl_days, source_effective_ids,
                 time_range_start, time_range_end, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ''', (
                memory_id, user_id, topic, summary,
                json.dumps(facts, ensure_ascii=False),
                memory_type,
                json.dumps(tags or [], ensure_ascii=False),
                confidence, ttl_days,
                json.dumps(source_effective_ids or [], ensure_ascii=False),
                time_range_start, time_range_end,
                now, now
            ))
        return memory_id

    def add_memory_from_dict(self, user_id: str, effective_memory: Dict[str, Any]) -> str:
        """
        从 effective_memory 字典直接创建长期记忆（方便从抽取结果直接写入）

        Args:
            user_id: 用户 ID
            effective_memory: effective_memory JSON 对象（含 topic/summary/facts 等字段）

        Returns:
            memory_id
        """
        time_range = effective_memory.get('time_range', {})
        return self.add_memory(
            user_id=user_id,
            topic=effective_memory.get('topic', ''),
            summary=effective_memory.get('summary', ''),
            facts=effective_memory.get('facts', []),
            memory_type=effective_memory.get('memory_type', 'LongTermMemory'),
            tags=effective_memory.get('tags', []),
            confidence=effective_memory.get('confidence', 1.0),
            ttl_days=effective_memory.get('ttl_days'),
            source_effective_ids=effective_memory.get('source_memory_ids', []),
            time_range_start=time_range.get('start'),
            time_range_end=time_range.get('end'),
        )

    # ============== Read (查) ==============

    def get_memory_by_id(self, memory_id: str) -> Optional[LongTermMemory]:
        """根据 memory_id 查询"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM longterm_memories WHERE memory_id = ? AND is_deleted = 0',
                (memory_id,)
            )
            row = cursor.fetchone()
            return LongTermMemory.from_row(row) if row else None

    def get_memories_by_user(self,
                              user_id: str,
                              memory_type: Optional[str] = None,
                              limit: Optional[int] = None,
                              include_deleted: bool = False) -> List[LongTermMemory]:
        """查询用户的所有长期记忆"""
        query = 'SELECT * FROM longterm_memories WHERE user_id = ?'
        params: List[Any] = [user_id]

        if not include_deleted:
            query += ' AND is_deleted = 0'
        if memory_type:
            query += ' AND memory_type = ?'
            params.append(memory_type)

        query += ' ORDER BY created_at DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [LongTermMemory.from_row(row) for row in cursor.fetchall()]

    def search_memories(self,
                        user_id: str,
                        query: str,
                        top_k: int = 5,
                        fields: Optional[List[str]] = None) -> List[LongTermMemory]:
        """
        关键词语义搜索（简单 BM25 风格评分）

        Args:
            user_id: 用户 ID
            query: 搜索查询
            top_k: 返回数量
            fields: 搜索字段列表，默认 ['topic', 'summary', 'facts', 'tags']

        Returns:
            按相关度排序的记忆列表
        """
        if fields is None:
            fields = ['topic', 'summary', 'facts', 'tags']

        memories = self.get_memories_by_user(user_id)
        keywords = set(query.lower().split())

        scored = []
        for mem in memories:
            score = 0.0
            if 'topic' in fields:
                text = mem.topic.lower()
                score += sum(3.0 for kw in keywords if kw in text)
            if 'summary' in fields:
                text = mem.summary.lower()
                score += sum(2.0 for kw in keywords if kw in text)
            if 'facts' in fields:
                text = ' '.join(mem.facts).lower()
                score += sum(2.0 for kw in keywords if kw in text)
            if 'tags' in fields:
                text = ' '.join(mem.tags).lower()
                score += sum(1.0 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def find_similar_by_topic(self, user_id: str, topic: str, threshold: float = 0.3) -> Optional[LongTermMemory]:
        """
        根据 topic 相似度查找最匹配的记忆（用于 update 判断）

        Args:
            user_id: 用户 ID
            topic: 待比较的主题
            threshold: 相似度阈值（词重叠率）

        Returns:
            最相似的记忆，若低于阈值返回 None
        """
        memories = self.get_memories_by_user(user_id)
        if not memories:
            return None

        topic_words = set(topic.lower().split())
        best_score = 0.0
        best_mem = None

        for mem in memories:
            mem_words = set(mem.topic.lower().split())
            if not topic_words or not mem_words:
                continue
            overlap = len(topic_words & mem_words)
            union = len(topic_words | mem_words)
            score = overlap / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_mem = mem

        return best_mem if best_score >= threshold else None

    # ============== Update (改) ==============

    def update_memory(self,
                      memory_id: str,
                      topic: Optional[str] = None,
                      summary: Optional[str] = None,
                      facts: Optional[List[str]] = None,
                      tags: Optional[List[str]] = None,
                      confidence: Optional[float] = None,
                      ttl_days: Optional[int] = None,
                      memory_type: Optional[str] = None) -> bool:
        """
        更新长期记忆的指定字段

        Returns:
            是否更新成功
        """
        updates = []
        params: List[Any] = []

        if topic is not None:
            updates.append('topic = ?')
            params.append(topic)
        if summary is not None:
            updates.append('summary = ?')
            params.append(summary)
        if facts is not None:
            updates.append('facts = ?')
            params.append(json.dumps(facts, ensure_ascii=False))
        if tags is not None:
            updates.append('tags = ?')
            params.append(json.dumps(tags, ensure_ascii=False))
        if confidence is not None:
            updates.append('confidence = ?')
            params.append(confidence)
        if ttl_days is not None:
            updates.append('ttl_days = ?')
            params.append(ttl_days)
        if memory_type is not None:
            updates.append('memory_type = ?')
            params.append(memory_type)

        if not updates:
            return False

        updates.append('updated_at = ?')
        params.append(datetime.now().isoformat())
        params.append(memory_id)

        query = f"UPDATE longterm_memories SET {', '.join(updates)} WHERE memory_id = ? AND is_deleted = 0"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount > 0

    def merge_and_update(self, memory_id: str, new_effective: Dict[str, Any]) -> bool:
        """
        将新的 effective_memory 信息合并到已有的长期记忆中
        （合并 facts 列表，更新 summary，追加 tags）

        Args:
            memory_id: 要更新的记忆 ID
            new_effective: 新的 effective_memory 字典

        Returns:
            是否成功
        """
        existing = self.get_memory_by_id(memory_id)
        if not existing:
            return False

        # 合并 facts（去重）
        old_facts_set = set(existing.facts)
        new_facts = new_effective.get('facts', [])
        merged_facts = existing.facts + [f for f in new_facts if f not in old_facts_set]

        # 合并 tags（去重）
        old_tags_set = set(existing.tags)
        new_tags = new_effective.get('tags', [])
        merged_tags = list(old_tags_set | set(new_tags))

        # 更新 summary（追加新内容）
        new_summary = new_effective.get('summary', '')
        merged_summary = existing.summary
        if new_summary and new_summary not in existing.summary:
            merged_summary = existing.summary + ' ' + new_summary

        # 取较高置信度
        new_conf = new_effective.get('confidence', existing.confidence)
        merged_confidence = max(existing.confidence, new_conf)

        return self.update_memory(
            memory_id=memory_id,
            summary=merged_summary,
            facts=merged_facts,
            tags=merged_tags,
            confidence=merged_confidence
        )

    # ============== Delete (删) ==============

    def delete_memory(self, memory_id: str, hard_delete: bool = False) -> bool:
        """
        删除长期记忆

        Args:
            memory_id: 记忆 ID
            hard_delete: True=物理删除, False=软删除（默认）

        Returns:
            是否成功
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if hard_delete:
                cursor.execute('DELETE FROM longterm_memories WHERE memory_id = ?', (memory_id,))
            else:
                cursor.execute(
                    'UPDATE longterm_memories SET is_deleted = 1, updated_at = ? WHERE memory_id = ?',
                    (datetime.now().isoformat(), memory_id)
                )
            return cursor.rowcount > 0

    def delete_memories_by_user(self, user_id: str, hard_delete: bool = False) -> int:
        """删除用户的所有长期记忆"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if hard_delete:
                cursor.execute('DELETE FROM longterm_memories WHERE user_id = ?', (user_id,))
            else:
                cursor.execute(
                    'UPDATE longterm_memories SET is_deleted = 1, updated_at = ? WHERE user_id = ?',
                    (datetime.now().isoformat(), user_id)
                )
            return cursor.rowcount

    def restore_memory(self, memory_id: str) -> bool:
        """恢复软删除的记忆"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE longterm_memories SET is_deleted = 0, updated_at = ? WHERE memory_id = ?',
                (datetime.now().isoformat(), memory_id)
            )
            return cursor.rowcount > 0

    # ============== Stats ==============

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) as total FROM longterm_memories WHERE user_id = ? AND is_deleted = 0',
                (user_id,)
            )
            total = cursor.fetchone()['total']

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM longterm_memories WHERE user_id = ? AND is_deleted = 0 AND memory_type = 'LongTermMemory'",
                (user_id,)
            )
            long_term_cnt = cursor.fetchone()['cnt']

            cursor.execute(
                "SELECT COUNT(*) as cnt FROM longterm_memories WHERE user_id = ? AND is_deleted = 1",
                (user_id,)
            )
            deleted_cnt = cursor.fetchone()['cnt']

        return {
            'user_id': user_id,
            'total_active': total,
            'long_term_memory': long_term_cnt,
            'user_memory': total - long_term_cnt,
            'deleted': deleted_cnt,
        }

    def export_to_jsonl(self, user_id: str, output_path: str, include_deleted: bool = False):
        """导出用户记忆为 JSONL 文件"""
        memories = self.get_memories_by_user(user_id, include_deleted=include_deleted)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for mem in memories:
                f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + '\n')
        return len(memories)


# ============== 便捷接口函数 ==============

def add_longterm_memory(user_id: str, effective_memory: Dict[str, Any],
                        db_path: str = DEFAULT_DB_PATH) -> str:
    """添加长期记忆（便捷接口）"""
    db = LongTermMemoryDatabase(db_path)
    return db.add_memory_from_dict(user_id, effective_memory)


def update_longterm_memory(memory_id: str, db_path: str = DEFAULT_DB_PATH, **kwargs) -> bool:
    """更新长期记忆（便捷接口）"""
    db = LongTermMemoryDatabase(db_path)
    return db.update_memory(memory_id, **kwargs)


def delete_longterm_memory(memory_id: str, db_path: str = DEFAULT_DB_PATH,
                           hard_delete: bool = False) -> bool:
    """删除长期记忆（便捷接口）"""
    db = LongTermMemoryDatabase(db_path)
    return db.delete_memory(memory_id, hard_delete=hard_delete)


def search_longterm_memories(user_id: str, query: str, top_k: int = 5,
                              db_path: str = DEFAULT_DB_PATH) -> List[LongTermMemory]:
    """搜索长期记忆（便捷接口）"""
    db = LongTermMemoryDatabase(db_path)
    return db.search_memories(user_id, query, top_k)

