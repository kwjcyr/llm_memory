"""
从原始 JSON 导出为 Session Memory（JSONL 格式）

用法：
  python import_from_json.py --group group_0_caroline
  python import_from_json.py --group group_1_jon
  python import_from_json.py --input data/input/group_0_caroline.json --output data/groups/group_0_caroline/session/memories.jsonl
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 项目根目录（src/memory/session_memory → 上 3 级）
BASE_DIR = str(Path(__file__).resolve().parents[3])
GROUPS_DIR = os.path.join(BASE_DIR, 'data', 'groups')


def parse_api_records(json_file_path: str) -> List[Dict[str, Any]]:
    """
    解析 API 响应 JSON 格式为统一的 session memory 记录列表。

    输入: { code, message, data: [ {memoryId, userId, createTime, messages:[{role,content,showTimestamp}]} ] }
    输出: [ {memory_id, user_id, timestamp, user_content, assistant_content} ]
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    records = []
    for item in raw.get('data', []):
        memory_id  = str(item.get('memoryId', ''))
        user_id    = item.get('userId', '')
        messages   = item.get('messages', [])

        user_content      = ''
        assistant_content = ''
        show_ts           = None

        for msg in messages:
            role    = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                user_content = content
                show_ts = msg.get('showTimestamp')
            elif role == 'assistant':
                assistant_content = content

        # 时间戳转换：毫秒 Unix → ISO
        if show_ts:
            try:
                ts = datetime.fromtimestamp(int(show_ts) / 1000)
                timestamp = ts.strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                timestamp = str(item.get('createTime', ''))[:19]
        else:
            timestamp = str(item.get('createTime', ''))[:19]

        if user_content or assistant_content:
            records.append({
                'memory_id':         memory_id,
                'user_id':           user_id,
                'timestamp':         timestamp,
                'user_content':      user_content,
                'assistant_content': assistant_content,
            })
    return records


def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> int:
    """将记录列表写入 JSONL 文件，返回写入条数。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(records)


def resolve_group_paths(group_name: str) -> tuple[str, str]:
    """根据 group 名称推导输入输出路径。

    group_name = 'group_0_caroline'
      => input  = data/groups/group_0_caroline/raw/group_0_caroline.json
      => output = data/groups/group_0_caroline/session/memories.jsonl
    """
    group_dir = os.path.join(GROUPS_DIR, group_name)
    raw_dir = os.path.join(group_dir, 'raw')
    raw_file = os.path.join(raw_dir, f'{group_name}.json')
    if not os.path.exists(raw_file):
        raw_file = os.path.join(BASE_DIR, 'data', 'input', f'{group_name}.json')

    output_file = os.path.join(group_dir, 'session', 'memories.jsonl')
    return raw_file, output_file


def main():
    parser = argparse.ArgumentParser(
        description='从原始 API JSON 导出为 Session Memory (JSONL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 指定 group 名称（自动推导路径）
  python import_from_json.py --group group_0_caroline
  python import_from_json.py --group group_0_melanie

  # 手动指定输入输出
  python import_from_json.py --input data/input/group_0_caroline.json --output data/my_session.jsonl
        """
    )
    parser.add_argument('--group', '-g', type=str, default=None,
                        help='group 名称，如 group_0_caroline')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='原始 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 JSONL 文件路径')
    args = parser.parse_args()

    in_path, out_path = args.input, args.output

    if args.group:
        g_in, g_out = resolve_group_paths(args.group)
        if in_path is None:
            in_path = g_in
        if out_path is None:
            out_path = g_out

    if not in_path or not os.path.exists(in_path):
        print(f"输入文件不存在: {in_path}")
        sys.exit(1)

    print("=" * 70)
    print("Session Memory 导入")
    print(f"  输入 : {in_path}")
    print(f"  输出 : {out_path}")
    print("=" * 70)

    records = parse_api_records(in_path)
    count = save_jsonl(records, out_path)

    print(f"\n完成！共导出 {count} 条 Session Memory")
    print(f"   文件: {out_path}")


if __name__ == '__main__':
    main()

