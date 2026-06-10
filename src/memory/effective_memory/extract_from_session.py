"""
从 Session Memory 抽取 Effective Memory
按时间窗口合并对话，使用 LLM 进行抽取。

用法：
  python extract_from_session.py --group group_0_caroline
  python extract_from_session.py --input data/groups/group_0_caroline/session/memories.jsonl
  python extract_from_session.py --group group_0_melanie --window 8
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# ─── 路径常量 ──────────────────────────────────────────────────────────────
BASE_DIR = str(Path(__file__).resolve().parents[3])
GROUPS_DIR = os.path.join(BASE_DIR, 'data', 'groups')
EXTRACT_MD_PATH   = os.path.join(BASE_DIR, 'config', 'prompt', 'extract.md')
EFFECTIVE_JSON_PATH = os.path.join(BASE_DIR, 'config', 'prompt', 'effective.json')

# ─── LLM 调用（统一从 .env 读取配置）───────────────────────────────────────
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from call_llm.call_llm_chat import call_llm, get_config


# ─── Prompt 配置加载 ──────────────────────────────────────────────────────

def load_effective_config() -> Dict[str, Any]:
    """加载抽取配置。优先 extract.md，回退 effective.json。"""
    if os.path.exists(EXTRACT_MD_PATH):
        with open(EXTRACT_MD_PATH, 'r', encoding='utf-8') as f:
            md_content = f.read()
        suffix_marker = '\n## Conversation\n'
        if suffix_marker in md_content:
            template_part = md_content.split(suffix_marker)[0].strip()
            prompt_suffix = suffix_marker + '\n{{{conversation}}}\n'
        else:
            template_part = md_content
            prompt_suffix = '\n\nConversation:\n{{{conversation}}}'
        return {
            'extractParam': {
                'llmModelParam': {
                    'modelName': get_config()['model'],
                    'temperature': get_config()['temperature'],
                    'topP': get_config()['top_p'],
                    'maxToken': get_config()['max_tokens'],
                },
                'promptTemplate': template_part,
                'promptSuffix': prompt_suffix,
            },
            '_source': 'extract.md',
        }
    with open(EFFECTIVE_JSON_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['_source'] = 'effective.json'
    return config


# ─── API JSON 解析 ──────────────────────────────────────────────────────────

def _parse_api_json(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将 API 响应格式转为 session memory 记录格式。"""
    records = []
    for item in raw.get('data', []):
        memory_id  = str(item.get('memoryId', ''))
        user_id    = item.get('userId', '')
        messages   = item.get('messages', [])
        user_content = ''
        assistant_content = ''
        show_ts = None
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                user_content = content
                show_ts = msg.get('showTimestamp')
            elif role == 'assistant':
                assistant_content = content
        if show_ts:
            try:
                ts = datetime.fromtimestamp(int(show_ts) / 1000)
                timestamp = ts.strftime('%Y-%m-%dT%H:%M:%S')
            except Exception:
                timestamp = str(item.get('createTime', ''))[:19]
        else:
            timestamp = str(item.get('createTime', ''))[:19]
        records.append({
            'memory_id': memory_id,
            'user_id': user_id,
            'timestamp': timestamp,
            'user_content': user_content,
            'assistant_content': assistant_content,
        })
    return records


# ─── Session Memory 加载 ────────────────────────────────────────────────────

def load_session_memories(file_path: str, time_window_hours: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """加载 session memories 并按时间窗口分组。支持 JSONL 和 API JSON 两种格式。"""
    all_memories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read().strip()

    first_line = raw_text.split('\n')[0].strip()
    try:
        first_obj = json.loads(first_line)
        is_jsonl = True
    except json.JSONDecodeError:
        is_jsonl = False

    if is_jsonl and isinstance(first_obj, dict) and 'data' in first_obj and 'code' in first_obj:
        all_memories = _parse_api_json(json.loads(raw_text))
    elif is_jsonl:
        for line in raw_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                all_memories.append(json.loads(line))
            except Exception as e:
                print(f"解析记忆失败：{e}")
    else:
        try:
            raw_json = json.loads(raw_text)
            if isinstance(raw_json, dict) and 'data' in raw_json:
                all_memories = _parse_api_json(raw_json)
            elif isinstance(raw_json, list):
                all_memories = raw_json
        except Exception as e:
            print(f"文件格式无法识别：{e}")
            return {}

    valid_memories = []
    for m in all_memories:
        try:
            m['_parsed_timestamp'] = datetime.fromisoformat(m['timestamp'])
            valid_memories.append(m)
        except Exception as e:
            print(f"跳过无效记录：{e}")
    all_memories = valid_memories

    # 按时间排序 + 窗口分组
    all_memories.sort(key=lambda x: x['timestamp'])
    grouped: Dict[str, List] = {}
    if not all_memories:
        return grouped

    earliest = all_memories[0]['_parsed_timestamp']
    win_start = earliest
    cur_batch: List = []
    wid = 0
    for m in all_memories:
        if m['_parsed_timestamp'] - win_start <= timedelta(hours=time_window_hours):
            cur_batch.append(m)
        else:
            if cur_batch:
                grouped[f'window_{wid}'] = cur_batch
                wid += 1
            win_start = m['_parsed_timestamp']
            cur_batch = [m]
    if cur_batch:
        grouped[f'window_{wid}'] = cur_batch
    return grouped


# ─── 对话格式化 ──────────────────────────────────────────────────────────

def format_conversation(memories: List[Dict[str, Any]]) -> str:
    lines = []
    for m in memories:
        uc = m.get('user_content', '')
        ac = m.get('assistant_content', '')
        if uc:
            lines.append(f"User: {uc}")
        if ac:
            lines.append(f"Assistant: {ac}")
    return "\n".join(lines)


# ─── LLM 抽取（使用统一 call_llm）─────────────────────────────────────────────────

def call_llm_for_extraction(conversation: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """调用 LLM 进行记忆抽取，返回解析后的 JSON 或 None。"""
    template = config.get('extractParam', {}).get('promptTemplate', '')
    suffix   = config.get('extractParam', {}).get('promptSuffix', '')
    full_prompt = (template + suffix).replace('{{{conversation}}}', conversation)

    try:
        response = call_llm(full_prompt)
        # 尝试提取 JSON
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
        print(f"响应中未找到有效 JSON: {response[:200]}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{e}\n原始响应：{response[:500]}")
        return None
    except Exception as e:
        print(f"LLM 调用失败：{e}")
        return None


# ─── 保存结果 ──────────────────────────────────────────────────────────────

def save_effective_memory(memory: Dict, output_path: str,
                         original_conversation: str = '',
                         memory_ids: List[str] = None,
                         time_range: Dict[str, str] = None):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if original_conversation:
        memory['original_text'] = original_conversation
    if memory_ids:
        memory['source_memory_ids'] = memory_ids
    if time_range:
        memory['time_range'] = time_range
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(memory, ensure_ascii=False) + '\n')


# ─── Group 路径推导 ─────────────────────────────────────────────────────────

def resolve_extract_paths(group_name: str) -> tuple[str, str]:
    """
    group_name = 'group_0_caroline'
      => input  = data/groups/group_0_caroline/session/memories.jsonl
      => output = data/groups/group_0_caroline/effective/effective_memories.jsonl
    """
    group_dir = os.path.join(GROUPS_DIR, group_name)
    session_file = os.path.join(group_dir, 'session', 'memories.jsonl')
    effective_file = os.path.join(group_dir, 'effective', 'effective_memories.jsonl')
    return session_file, effective_file


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def extract_effective_memories(
    input_path: str = None,
    output_path: str = None,
    time_window_hours: int = 4,
):
    if input_path is None:
        input_path = os.path.join(BASE_DIR, 'data', 'caroline_memories.txt')
    if output_path is None:
        output_path = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')

    cfg = get_config()
    print("=" * 70)
    print("Effective Memory 抽取")
    print(f"  模型 : {cfg['model']}")
    print(f"  输入 : {input_path}")
    print(f"  输出 : {output_path}")
    print("=" * 70)

    print("\n1. 加载 Prompt 配置...")
    config = load_effective_config()
    print(f"   来源: {config['_source']}")

    print(f"\n2. 加载 Session Memories ({time_window_hours}h 窗口)...")
    grouped = load_session_memories(input_path, time_window_hours)
    total_wins = len(grouped)
    total_mem = sum(len(v) for v in grouped.values())
    print(f"   {total_wins} 个窗口, {total_mem} 条记忆")

    if not grouped:
        print("   没有记忆数据")
        return

    print("\n3. 开始抽取...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    ok = 0
    for wid, mems in grouped.items():
        print(f"\n  {wid} ({len(mems)} 条)")
        earliest = mems[0]['timestamp']
        latest  = mems[-1]['timestamp']
        print(f"    时间: {earliest} ~ {latest}")

        ids = [m['memory_id'] for m in mems]
        conv = format_conversation(mems)
        result = call_llm_for_extraction(conv, config)

        if result:
            print(f"    OK: {result.get('topic', '?')}")
            save_effective_memory(result, output_path,
                                original_conversation=conv,
                                memory_ids=ids,
                                time_range={'start': earliest, 'end': latest})
            ok += 1
        else:
            print(f"    FAIL")

    print(f"\n{'=' * 70}")
    print(f"完成！{ok}/{total_wins} 窗口成功 → {output_path}")


# ─── CLI 入口 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从 Session Memory 抽取 Effective Memory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 指定 group 名称（自动推导路径）
  python extract_from_session.py --group group_0_caroline
  python extract_from_session.py --group group_0_melanie

  # 手动指定输入输出
  python extract_from_session.py --input data/caroline_memories.txt --output data/my_effective.txt

  # 调整时间窗口
  python extract_from_session.py --group group_0_caroline --window 8
        """,
    )
    parser.add_argument('--group', '-g', type=str, default=None,
                        help='group 名称，自动推导 input/output 路径')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Session Memory 文件路径（JSONL）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Effective Memory 输出路径')
    parser.add_argument('--window', '-w', type=int, default=4,
                        help='时间窗口（小时），默认 4')
    args = parser.parse_args()

    in_p, out_p = args.input, args.output
    if args.group:
        g_in, g_out = resolve_extract_paths(args.group)
        if in_p is None:
            in_p = g_in
        if out_p is None:
            out_p = g_out

    extract_effective_memories(
        input_path=in_p,
        output_path=out_p,
        time_window_hours=args.window,
    )

