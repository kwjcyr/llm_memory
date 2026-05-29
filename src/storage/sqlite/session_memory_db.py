import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

# 默认文件路径
DEFAULT_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'data',
    'session_memories.jsonl'
)


@dataclass
class Conversation:
    """对话记录数据模型"""
    user_id: str
    timestamp: str
    user_content: str
    assistant_content: str
    turn_id: int
    memory_id: str  # 随机生成的唯一标识

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'Conversation':
        """从数据库行创建对象"""
        return cls(
            user_id=row['user_id'],
            timestamp=row['timestamp'],
            user_content=row['user_content'],
            assistant_content=row['assistant_content'],
            turn_id=row['turn_id'],
            memory_id=row['memory_id']
        )


class ConversationDatabase:
    """对话记录数据库操作类"""

    def __init__(self, db_path: str = 'conversations.db'):
        """
        初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
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
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引以提高查询性能
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON conversations(user_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON conversations(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_turn_id
                ON conversations(user_id, turn_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_id
                ON conversations(memory_id)
            ''')

    # ============== Create (增) ==============

    def add_conversation(self,
                        user_id: str,
                        user_content: str,
                        assistant_content: str,
                        turn_id: Optional[int] = None,
                        timestamp: Optional[str] = None,
                        memory_id: Optional[str] = None) -> str:
        """
        添加一条对话记录

        Args:
            user_id: 用户 ID
            user_content: 用户输入内容
            assistant_content: 助手回答内容
            turn_id: 轮次 ID（可选，不提供则自动计算）
            timestamp: 时间戳（可选，默认为当前时间）
            memory_id: 唯一标识（可选，默认为随机生成 UUID）

        Returns:
            memory_id: 对话记录的唯一标识
        """
        # 生成默认值
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if turn_id is None:
            # 自动获取该用户的下一个 turn_id
            turn_id = self.get_next_turn_id(user_id)

        # 创建对话记录
        conversation = Conversation(
            user_id=user_id,
            timestamp=timestamp,
            user_content=user_content,
            assistant_content=assistant_content,
            turn_id=turn_id,
            memory_id=memory_id
        )

        # 插入数据库
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations
                (memory_id, user_id, timestamp, user_content, assistant_content, turn_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conversation.memory_id,
                conversation.user_id,
                conversation.timestamp,
                conversation.user_content,
                conversation.assistant_content,
                conversation.turn_id
            ))

        return memory_id

    def add_conversations_batch(self,
                               conversations: List[Dict[str, Any]]) -> int:
        """
        批量添加对话记录

        Args:
            conversations: 对话记录列表，每个包含：
                - user_id (必填)
                - user_content (必填)
                - assistant_content (必填)
                - turn_id (可选)
                - timestamp (可选)
                - memory_id (可选)

        Returns:
            成功添加的记录数量
        """
        if not conversations:
            return 0

        rows = []
        for conv in conversations:
            memory_id = conv.get('memory_id', str(uuid.uuid4()))
            timestamp = conv.get('timestamp', datetime.now().isoformat())

            # 如果没提供 turn_id，需要获取
            turn_id = conv.get('turn_id')
            if turn_id is None:
                turn_id = self.get_next_turn_id(conv['user_id'])

            rows.append((
                memory_id,
                conv['user_id'],
                timestamp,
                conv['user_content'],
                conv['assistant_content'],
                turn_id
            ))

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO conversations
                (memory_id, user_id, timestamp, user_content, assistant_content, turn_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', rows)

        return len(rows)

    # ============== Read (查) ==============

    def get_conversation_by_memory_id(self, memory_id: str) -> Optional[Conversation]:
        """
        根据 memory_id 查询对话

        Args:
            memory_id: 对话唯一标识

        Returns:
            Conversation 对象，不存在则返回 None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM conversations WHERE memory_id = ?
            ''', (memory_id,))
            row = cursor.fetchone()
            if row:
                return Conversation.from_row(row)
            return None

    def get_conversations_by_user_id(self,
                                     user_id: str,
                                     limit: Optional[int] = None,
                                     offset: int = 0,
                                     order_by: str = 'DESC') -> List[Conversation]:
        """
        根据用户 ID 查询对话历史

        Args:
            user_id: 用户 ID
            limit: 返回数量限制（None 表示不限制）
            offset: 偏移量
            order_by: 排序方式 ('ASC' 或 'DESC')

        Returns:
            Conversation 对象列表
        """
        query = '''
            SELECT * FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp {}
        '''.format(order_by)

        if limit:
            query += ' LIMIT ? OFFSET ?'
            params = (user_id, limit, offset)
        else:
            params = (user_id,)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [Conversation.from_row(row) for row in rows]

    def get_conversation_by_turn_id(self,
                                    user_id: str,
                                    turn_id: int) -> Optional[Conversation]:
        """
        根据轮次 ID 查询对话

        Args:
            user_id: 用户 ID
            turn_id: 轮次 ID

        Returns:
            Conversation 对象，不存在则返回 None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM conversations
                WHERE user_id = ? AND turn_id = ?
            ''', (user_id, turn_id))
            row = cursor.fetchone()
            if row:
                return Conversation.from_row(row)
            return None

    def get_all_conversations(self,
                             limit: Optional[int] = None,
                             offset: int = 0) -> List[Conversation]:
        """
        获取所有对话记录

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            Conversation 对象列表
        """
        query = 'SELECT * FROM conversations ORDER BY timestamp DESC'

        if limit:
            query += f' LIMIT {limit} OFFSET {offset}'

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [Conversation.from_row(row) for row in rows]

    def search_conversations(self,
                            user_id: Optional[str] = None,
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            keyword: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Conversation]:
        """
        搜索对话记录（支持多条件）

        Args:
            user_id: 用户 ID
            start_time: 开始时间（ISO 格式）
            end_time: 结束时间（ISO 格式）
            keyword: 搜索关键词（在 user_content 和 assistant_content 中搜索）
            limit: 返回数量限制

        Returns:
            Conversation 对象列表
        """
        query = 'SELECT * FROM conversations WHERE 1=1'
        params = []

        if user_id:
            query += ' AND user_id = ?'
            params.append(user_id)

        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)

        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)

        if keyword:
            query += ' AND (user_content LIKE ? OR assistant_content LIKE ?)'
            params.extend([f'%{keyword}%', f'%{keyword}%'])

        query += ' ORDER BY timestamp DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [Conversation.from_row(row) for row in rows]

    # ============== Helper Methods ==============

    def get_next_turn_id(self, user_id: str) -> int:
        """
        获取用户的下一个轮次 ID

        Args:
            user_id: 用户 ID

        Returns:
            下一个 turn_id（从 1 开始）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MAX(turn_id) as max_turn
                FROM conversations
                WHERE user_id = ?
            ''', (user_id,))
            row = cursor.fetchone()
            max_turn = row['max_turn'] if row else None

            return (max_turn or 0) + 1

    def get_user_conversation_count(self, user_id: str) -> int:
        """
        获取用户的对话记录数量

        Args:
            user_id: 用户 ID

        Returns:
            对话记录数量
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM conversations
                WHERE user_id = ?
            ''', (user_id,))
            row = cursor.fetchone()
            return row['count'] if row else 0

    def get_latest_conversation(self, user_id: str) -> Optional[Conversation]:
        """
        获取用户最新的对话记录

        Args:
            user_id: 用户 ID

        Returns:
            最新的 Conversation 对象
        """
        conversations = self.get_conversations_by_user_id(user_id, limit=1)
        return conversations[0] if conversations else None

    # ============== Update (改) ==============

    def update_conversation(self,
                           memory_id: str,
                           user_content: Optional[str] = None,
                           assistant_content: Optional[str] = None) -> bool:
        """
        更新对话记录

        Args:
            memory_id: 对话唯一标识
            user_content: 新的用户输入（可选）
            assistant_content: 新的助手回答（可选）

        Returns:
            是否更新成功
        """
        if not user_content and not assistant_content:
            return False

        updates = []
        params = []

        if user_content:
            updates.append('user_content = ?')
            params.append(user_content)

        if assistant_content:
            updates.append('assistant_content = ?')
            params.append(assistant_content)

        params.append(memory_id)
        query = f"UPDATE conversations SET {', '.join(updates)} WHERE memory_id = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount > 0

    # ============== Delete (删) ==============

    def delete_conversation(self, memory_id: str) -> bool:
        """
        删除对话记录

        Args:
            memory_id: 对话唯一标识

        Returns:
            是否删除成功
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM conversations WHERE memory_id = ?
            ''', (memory_id,))
            return cursor.rowcount > 0

    def delete_conversations_by_user(self, user_id: str) -> int:
        """
        删除用户的所有对话记录

        Args:
            user_id: 用户 ID

        Returns:
            删除的记录数量
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM conversations WHERE user_id = ?
            ''', (user_id,))
            return cursor.rowcount

    def clear_all_conversations(self) -> int:
        """
        清空所有对话记录

        Returns:
            删除的记录数量
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations')
            return cursor.rowcount

    def export_conversations(self,
                            user_id: Optional[str] = None,
                            format: str = 'json') -> str:
        """
        导出对话记录

        Args:
            user_id: 用户 ID（None 表示导出所有）
            format: 导出格式 ('json' 或其他)

        Returns:
            导出的 JSON 字符串
        """
        if user_id:
            conversations = self.get_conversations_by_user_id(user_id)
        else:
            conversations = self.get_all_conversations()

        if format == 'json':
            return json.dumps([c.to_dict() for c in conversations],
                            indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


# ============== 会话记忆管理接口 ==============

def add_session_memory(user_id: str,
                      user_content: str,
                      assistant_content: str,
                      timestamp: str,
                      file_path: Optional[str] = None,
                      memory_id: Optional[str] = None) -> str:
    """
    添加短期会话记忆（自动管理 turn_id 和 memory_id），并同步写入文件

    Args:
        user_id: 用户 ID
        user_content: 用户输入内容
        assistant_content: 助手回答内容
        timestamp: 时间戳（必填，ISO 格式）
        file_path: 文件路径（可选，默认为 data/session_memories.jsonl）
        memory_id: 自定义 memory_id（可选，不提供则自动生成 UUID）

    Returns:
        memory_id: 生成的唯一标识
    """
    db = ConversationDatabase()

    # 如果没有提供 memory_id，使用数据库自动生成的
    if memory_id is None:
        memory_id = db.add_conversation(
            user_id=user_id,
            user_content=user_content,
            assistant_content=assistant_content,
            timestamp=timestamp
        )
    else:
        # 使用提供的 memory_id
        memory_id = db.add_conversation(
            user_id=user_id,
            user_content=user_content,
            assistant_content=assistant_content,
            timestamp=timestamp,
            memory_id=memory_id
        )

    # 同步写入文件
    if file_path is None:
        file_path = DEFAULT_FILE_PATH

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 创建 JSON 记录（一行一条）
    record = {
        'memory_id': memory_id,
        'user_id': user_id,
        'timestamp': timestamp,
        'user_content': user_content,
        'assistant_content': assistant_content
    }

    # 追加写入文件（一行一条 JSON）
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return memory_id


def get_session_memory(user_id: str,
                       limit: Optional[int] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None) -> List[Conversation]:
    """
    获取会话记忆（支持最近 N 轮和时间范围筛选）

    Args:
        user_id: 用户 ID
        limit: 最近 N 轮对话（可选，None 表示不限制）
        start_time: 开始时间（可选，ISO 格式）
        end_time: 结束时间（可选，ISO 格式）

    Returns:
        Conversation 对象列表
    """
    db = ConversationDatabase()

    # 如果指定了时间范围，使用搜索功能
    if start_time or end_time:
        return db.search_conversations(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    # 否则直接按用户 ID 查询
    return db.get_conversations_by_user_id(user_id, limit=limit)

