"""
从 JSON 文件导入对话记录到 session memory
"""
import json
from datetime import datetime

from src.storage.sqlite.session_memory_db import add_session_memory


def import_from_json(json_file_path, output_file_path=None):
    """
    从 JSON 文件导入对话记录

    Args:
        json_file_path: JSON 文件路径
        output_file_path: 输出文件路径（可选）
    """
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data.get('data', [])
    print(f"共读取 {len(records)} 条记录")

    success_count = 0
    error_count = 0

    for i, record in enumerate(records, 1):
        try:
            # 提取数据
            user_id = record.get('userId', 'unknown')
            memory_id = record.get('memoryId', '')
            messages = record.get('messages', [])
            show_timestamp = messages[0].get('showTimestamp', 0) if messages else 0

            # 提取 user_content 和 assistant_content
            user_content = ""
            assistant_content = ""

            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    user_content = content
                elif role == 'assistant':
                    assistant_content = content

            # 转换时间戳（毫秒转 ISO 格式）
            timestamp_ms = int(show_timestamp)
            timestamp_sec = timestamp_ms / 1000
            timestamp = datetime.fromtimestamp(timestamp_sec).isoformat()

            # 添加记忆（使用 memoryId 作为生成的 memory_id）
            if user_content or assistant_content:
                result_memory_id = add_session_memory(
                    user_id=user_id,
                    user_content=user_content,
                    assistant_content=assistant_content,
                    timestamp=timestamp,
                    file_path=output_file_path,
                    memory_id=memory_id  # 使用原有的 memoryId
                )
                success_count += 1

                if i % 10 == 0 or i == len(records):
                    print(f"已处理 {i}/{len(records)} 条记录")
            else:
                print(f"第 {i} 条记录内容为空，跳过")

        except Exception as e:
            error_count += 1
            print(f"第 {i} 条记录处理失败：{e}")

    print(f"\n导入完成！")
    print(f"成功：{success_count} 条")
    print(f"失败：{error_count} 条")

    return success_count, error_count


if __name__ == '__main__':
    json_file = '/data/input/group_0_caroline.json'
    output_file = '/data/caroline_memories.txt'

    print(f"正在导入文件：{json_file}")
    print(f"输出到：{output_file}")
    print("=" * 80)

    success, errors = import_from_json(json_file, output_file)

    print("\n" + "=" * 80)
    print(f"导入完成！共成功 {success} 条，失败 {errors} 条")
    print(f"文件保存在：{output_file}")

