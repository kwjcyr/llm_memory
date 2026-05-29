import os
import sys
from datetime import datetime, timedelta

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.storage.sqlite.session_memory_db import add_session_memory, get_session_memory, ConversationDatabase


def demo():
    print("=" * 70)
    print("会话记忆管理接口 Demo")
    print("=" * 70)

    import tempfile
    temp_db = tempfile.mktemp(suffix='.db')
    temp_file = tempfile.mktemp(suffix='.jsonl')

    original_db_path = ConversationDatabase.__init__
    def custom_init(self, db_path=temp_db):
        self.db_path = db_path
        self._init_db()
    ConversationDatabase.__init__ = custom_init

    try:
        user_id = "user_98765"

        # 1. 添加会话记忆
        print("\n【1. 使用 add_session_memory 添加记忆】")

        # 添加带时间戳的记忆（timestamp 是必填项）
        print(f"\n同步写入文件：{temp_file}")

        timestamp1 = datetime.now().isoformat()
        memory_id1 = add_session_memory(
            user_id=user_id,
            user_content="你好，我想学习 Python",
            assistant_content="太好了！Python 是一门非常棒的编程语言。你从哪里开始学起呢？",
            timestamp=111,
            file_path=temp_file
        )
        print(f"添加记忆 1，memory_id: {memory_id1}")

        timestamp2 = datetime.now().isoformat()
        memory_id2 = add_session_memory(
            user_id=user_id,
            user_content="从基础语法开始",
            assistant_content="好的！我们先从变量、数据类型和控制结构开始学习吧。",
            timestamp=timestamp2,
            file_path=temp_file
        )
        print(f"添加记忆 2，memory_id: {memory_id2}")

        timestamp3 = datetime.now().isoformat()
        memory_id3 = add_session_memory(
            user_id=user_id,
            user_content="什么是变量？",
            assistant_content="变量是用来存储数据的容器。在 Python 中，你可以这样定义变量：\n\n```python\nname = 'Alice'\nage = 25\n```",
            timestamp=timestamp3,
            file_path=temp_file
        )
        print(f"添加记忆 3，memory_id: {memory_id3}")

        # 添加过去时间的记忆
        timestamp4 = (datetime.now() - timedelta(hours=1)).isoformat()
        memory_id4 = add_session_memory(
            user_id=user_id,
            user_content="那我们继续学习函数",
            assistant_content="很好！函数是组织好的、可重复使用的代码块...",
            timestamp=timestamp4,
            file_path=temp_file
        )
        print(f"添加记忆 4（1 小时前），memory_id: {memory_id4}")

        # 2. 查看文件内容
        print("\n【2. 查看文件内容（TXT 格式）】")
        print(f"文件路径：{temp_file}")
        print("-" * 80)
        with open(temp_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        print("-" * 80)

        # 2. 获取会话记忆
        print("\n【2. 使用 get_session_memory 获取记忆】")

        # 获取所有记忆
        print("\n获取所有记忆:")
        all_memories = get_session_memory(user_id)
        print(f"共 {len(all_memories)} 条:")
        for m in all_memories:
            print(f"  [Turn {m.turn_id}] {m.user_content[:30]}...")

        # 获取最近 2 轮
        print(f"\n获取最近 2 轮:")
        recent_2 = get_session_memory(user_id, limit=2)
        for m in recent_2:
            print(f"  [Turn {m.turn_id}] {m.user_content}")

        # 按时间范围获取
        print(f"\n按时间范围获取（最近 1 小时内）:")
        end_time = datetime.now().isoformat()
        start_time = (datetime.now() - timedelta(hours=2)).isoformat()
        time_filtered = get_session_memory(
            user_id,
            start_time=start_time,
            end_time=end_time
        )
        print(f"找到 {len(time_filtered)} 条:")
        for m in time_filtered:
            print(f"  [{m.timestamp[:19]}] Turn {m.turn_id}")

        # 组合使用：limit + 时间范围
        print(f"\n组合筛选：最近 1 条 + 时间范围:")
        combined = get_session_memory(
            user_id,
            limit=1,
            start_time=start_time,
            end_time=end_time
        )
        for m in combined:
            print(f"  [Turn {m.turn_id}] {m.user_content}")

        # 3. 直接访问底层 Conversation 对象
        print("\n【3. 访问完整对话内容】")
        memories = get_session_memory(user_id, limit=1)
        if memories:
            m = memories[0]
            print(f"User ID: {m.user_id}")
            print(f"Turn ID: {m.turn_id}")
            print(f"Memory ID: {m.memory_id}")
            print(f"Timestamp: {m.timestamp}")
            print(f"User Content: {m.user_content}")
            print(f"Assistant Content: {m.assistant_content[:100]}...")

        print("\n" + "=" * 70)
        print("Demo 完成!")
        print("=" * 70)

    finally:
        # 恢复原始初始化方法
        ConversationDatabase.__init__ = original_db_path
        # 清理临时文件
        if os.path.exists(temp_db):
            os.remove(temp_db)


if __name__ == '__main__':
    demo()

