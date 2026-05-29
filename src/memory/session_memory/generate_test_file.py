"""
生成测试 TXT 文件
"""
from datetime import datetime

from src.storage.sqlite.session_memory_db import add_session_memory

# 生成永久文件
file_path = "/data/session_memories.txt"

print(f"正在生成文件到：{file_path}")

# 添加几条测试数据
add_session_memory(
    user_id="user_test_001",
    user_content="你好，我想学习 Python 编程",
    assistant_content="太好了！Python 是一门非常棒的编程语言。你从哪里开始学起呢？",
    timestamp=datetime.now().isoformat(),
    file_path=file_path
)

add_session_memory(
    user_id="user_test_001",
    user_content="从基础语法开始吧",
    assistant_content="好的！我们先从变量、数据类型和控制结构开始学习。Python 的语法非常简洁易懂。",
    timestamp=datetime.now().isoformat(),
    file_path=file_path
)

add_session_memory(
    user_id="user_test_001",
    user_content="什么是变量？",
    assistant_content="变量是用来存储数据的容器。在 Python 中，你可以这样定义变量：\n\n```python\nname = 'Alice'\nage = 25\n```\n\n就这么简单！",
    timestamp=datetime.now().isoformat(),
    file_path=file_path
)

print(f"文件已生成：{file_path}")
print("\n文件内容预览:")
print("=" * 80)
with open(file_path, 'r', encoding='utf-8') as f:
    print(f.read())

