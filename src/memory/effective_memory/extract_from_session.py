"""
从 Session Memory 抽取 Effective Memory
按照时间窗口（4 小时）合并对话，使用 LLM 进行抽取
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests


def load_effective_config() -> Dict[str, Any]:
    """加载 effective.json 配置"""
    # 使用绝对路径
    config_path = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_session_memories(time_window_hours: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载 session memories 并按时间窗口分组

    Args:
        time_window_hours: 时间窗口（小时）

    Returns:
        字典，key 为时间窗口标识，value 为该窗口内的记忆列表
    """
    # 使用绝对路径
    file_path = '/Users/kwjcyr/data/llm_memory/data/caroline_memories.txt'

    all_memories = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                memory = json.loads(line.strip())
                # 解析时间戳
                timestamp = datetime.fromisoformat(memory['timestamp'])
                memory['_parsed_timestamp'] = timestamp
                all_memories.append(memory)
            except Exception as e:
                print(f"解析记忆失败：{e}")
                continue

    # 按时间排序
    all_memories.sort(key=lambda x: x['timestamp'])

    # 按时间窗口分组
    grouped_memories = {}
    if not all_memories:
        return grouped_memories

    # 找到最早的时间
    earliest_time = all_memories[0]['_parsed_timestamp']

    # 创建时间窗口
    current_window_start = earliest_time
    current_window_memories = []
    window_id = 0

    for memory in all_memories:
        # 检查是否在当前窗口内
        if memory['_parsed_timestamp'] - current_window_start <= timedelta(hours=time_window_hours):
            current_window_memories.append(memory)
        else:
            # 保存当前窗口并开始新窗口
            if current_window_memories:
                grouped_memories[f"window_{window_id}"] = current_window_memories
                window_id += 1
            current_window_start = memory['_parsed_timestamp']
            current_window_memories = [memory]

    # 保存最后一个窗口
    if current_window_memories:
        grouped_memories[f"window_{window_id}"] = current_window_memories

    return grouped_memories


def format_conversation(memories: List[Dict[str, Any]]) -> str:
    """
    格式化对话为 LLM 输入格式

    Args:
        memories: 记忆列表

    Returns:
        格式化的对话文本
    """
    conversation_lines = []

    for memory in memories:
        user_content = memory.get('user_content', '')
        assistant_content = memory.get('assistant_content', '')

        if user_content:
            conversation_lines.append(f"User: {user_content}")

        if assistant_content:
            conversation_lines.append(f"Assistant: {assistant_content}")

    return "\n".join(conversation_lines)


def call_llm_for_extraction(conversation: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用 LLM 进行记忆抽取

    Args:
        conversation: 格式化的对话
        config: effective.json 配置

    Returns:
        抽取的有效记忆
    """
    # 获取配置参数
    extract_param = config.get('extractParam', {})
    llm_model_param = extract_param.get('llmModelParam', {})
    prompt_template = extract_param.get('promptTemplate', '')
    prompt_suffix = extract_param.get('promptSuffix', '')
    friday_app_id = config.get('fridayAppId', '')

    # 构建完整 prompt
    full_prompt = prompt_template + prompt_suffix.replace('{{{conversation}}}', conversation)

    # 构建请求（使用 call_llm_chat.py 中的方式）
    url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"

    # 使用 effective.json 中配置的 fridayAppId
    friday_app_id = config.get('fridayAppId', '1980915965710716958')
    headers = {'Authorization': f'Bearer {friday_app_id}'}

    payload = {
        "model": llm_model_param.get('modelName', 'LongCat-Flash-Chat-Eco'),
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "temperature": llm_model_param.get('temperature', 0.1),
        "top_p": llm_model_param.get('topP', 1),
        "max_tokens": llm_model_param.get('maxToken', 4096),
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        # 解析响应
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']

            # 尝试解析为 JSON
            try:
                # 查找 JSON 部分
                json_start = content.find('{')
                json_end = content.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    extracted_memory = json.loads(json_str)
                    return extracted_memory
                else:
                    print(f"⚠️ 响应中未找到有效 JSON: {content[:200]}")
                    return None
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 解析失败：{e}")
                print(f"原始响应：{content[:500]}")
                return None
        else:
            print(f"⚠️ 响应格式异常：{result}")
            return None

    except Exception as e:
        print(f"❌ 调用 LLM 失败：{e}")
        return None


def save_effective_memory(memory: Dict[str, Any], output_path: str,
                         original_conversation: str = "",
                         memory_ids: List[str] = None,
                         time_range: Dict[str, str] = None):
    """
    保存有效记忆到文件

    Args:
        memory: 有效记忆对象
        output_path: 输出文件路径
        original_conversation: 原始对话内容
        memory_ids: 原文中所有记忆的 memory_id 列表
        time_range: 时间范围 {start: ..., end: ...}
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 添加元数据字段（不通过 LLM，由代码直接添加）
    if original_conversation:
        memory['original_text'] = original_conversation

    if memory_ids:
        memory['source_memory_ids'] = memory_ids

    if time_range:
        memory['time_range'] = time_range

    # 追加写入
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(memory, ensure_ascii=False) + '\n')


def extract_effective_memories(time_window_hours: int = 4):
    """
    主函数：从 session memory 抽取 effective memory

    Args:
        time_window_hours: 时间窗口（小时）
    """
    print("=" * 80)
    print("从 Session Memory 抽取 Effective Memory")
    print("=" * 80)

    # 1. 加载配置
    print("\n1. 加载配置...")
    config = load_effective_config()
    print(f"   Friday App ID: {config.get('fridayAppId', 'N/A')}")
    print(f"   模型：{config.get('extractParam', {}).get('llmModelParam', {}).get('modelName', 'N/A')}")

    # 2. 加载 session memories 并按时间窗口分组
    print(f"\n2. 加载 session memories 并按 {time_window_hours} 小时窗口分组...")
    grouped_memories = load_session_memories(time_window_hours)
    total_windows = len(grouped_memories)
    total_memories = sum(len(memories) for memories in grouped_memories.values())
    print(f"   共 {total_windows} 个时间窗口，{total_memories} 条记忆")

    if not grouped_memories:
        print("   ⚠️ 没有记忆数据")
        return

    # 3. 遍历每个时间窗口进行抽取
    print("\n3. 开始逐个窗口抽取有效记忆...")
    print("=" * 80)

    # 定义输出路径
    output_path = '/Users/kwjcyr/data/llm_memory/data/effective_memories.txt'
    # 清空旧文件
    if os.path.exists(output_path):
        os.remove(output_path)

    success_count = 0
    total_count = len(grouped_memories)

    for window_id, memories in grouped_memories.items():
        print(f"\n处理 {window_id} ({len(memories)} 条记忆)...")

        # 显示时间范围
        earliest = memories[0]['timestamp']
        latest = memories[-1]['timestamp']
        print(f"   时间范围：{earliest} 到 {latest}")

        # 收集所有 memory_ids
        memory_ids = [m['memory_id'] for m in memories]

        # 格式化对话
        conversation = format_conversation(memories)
        print(f"   对话长度：{len(conversation)} 字符")

        # 调用 LLM 抽取
        effective_memory = call_llm_for_extraction(conversation, config)

        if effective_memory:
            print(f"   ✅ 抽取成功：{effective_memory.get('topic', 'N/A')}")

            # 保存结果（包含原文、memory_ids、时间范围）
            time_range = {'start': earliest, 'end': latest}
            save_effective_memory(effective_memory, output_path,
                                original_conversation=conversation,
                                memory_ids=memory_ids,
                                time_range=time_range)
            success_count += 1
        else:
            print(f"   ❌ 抽取失败")



if __name__ == '__main__':
    # 设置时间窗口为 4 小时
    extract_effective_memories(time_window_hours=4)

