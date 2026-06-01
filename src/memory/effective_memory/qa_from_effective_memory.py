"""
基于 Effective Memory 的问答系统
从 effective_memories.txt 中检索相关信息，调用大模型回答问题
"""
import json
from typing import List, Dict, Any

import requests


def load_effective_memories(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 effective memories

    Args:
        file_path: 文件路径

    Returns:
        记忆列表
    """
    memories = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                memory = json.loads(line.strip())
                memories.append(memory)
            except Exception as e:
                print(f"解析记忆失败：{e}")
                continue

    return memories


def search_relevant_memories(question: str, memories: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    搜索与问题相关的记忆（使用更智能的关键词扩展）

    Args:
        question: 用户问题
        memories: 所有记忆
        top_k: 返回最相关的 K 条记忆

    Returns:
        相关记忆列表
    """
    # 扩展关键词（同义词和相关词）
    question_keywords = set(question.lower().split())

    # 添加同义词扩展
    keyword_expansions = {
        'education': ['education', 'study', 'learning', 'degree', 'certification', 'training'],
        'fields': ['fields', 'area', 'subject', 'discipline', 'major'],
        'pursue': ['pursue', 'study', 'learn', 'get', 'obtain', 'certification'],
        'psychology': ['psychology', 'mental health', 'counseling', 'therapist'],
        'counseling': ['counseling', 'counselor', 'therapy', 'therapist', 'psychology']
    }

    expanded_keywords = set(question_keywords)
    for keyword in question_keywords:
        if keyword in keyword_expansions:
            expanded_keywords.update(keyword_expansions[keyword])

    scored_memories = []

    for memory in memories:
        score = 0

        # 在 topic 中搜索
        topic = memory.get('topic', '').lower()
        if any(keyword in topic for keyword in expanded_keywords):
            score += 3

        # 在 summary 中搜索
        summary = memory.get('summary', '').lower()
        if any(keyword in summary for keyword in expanded_keywords):
            score += 2

        # 在 facts 中搜索
        facts = ' '.join(memory.get('facts', [])).lower()
        if any(keyword in facts for keyword in expanded_keywords):
            score += 2

        # 在 tags 中搜索
        tags = ' '.join(memory.get('tags', [])).lower()
        if any(keyword in tags for keyword in expanded_keywords):
            score += 1

        # 在 original_text 中搜索
        original_text = memory.get('original_text', '').lower()
        if any(keyword in original_text for keyword in expanded_keywords):
            score += 1

        if score > 0:
            scored_memories.append((score, memory))

    # 按分数排序
    scored_memories.sort(key=lambda x: x[0], reverse=True)

    # 返回 top_k
    return [memory for score, memory in scored_memories[:top_k]]


def call_llm_for_qa(question: str, relevant_memories: List[Dict[str, Any]], friday_app_id: str) -> str:
    """
    调用 LLM 基于检索到的记忆回答问题

    Args:
        question: 用户问题
        relevant_memories: 相关记忆列表
        friday_app_id: Friday App ID

    Returns:
        LLM 生成的答案
    """
    # 构建上下文
    context_parts = []

    for i, memory in enumerate(relevant_memories, 1):
        context = f"""
记忆 {i}:
- 主题：{memory.get('topic', 'N/A')}
- 摘要：{memory.get('summary', 'N/A')}
- 事实：{', '.join(memory.get('facts', []))}
- 时间范围：{memory.get('time_range', {}).get('start', 'N/A')} 到 {memory.get('time_range', {}).get('end', 'N/A')}
- 原文：{memory.get('original_text', 'N/A')[:500]}...
"""
        context_parts.append(context)

    context = "\n".join(context_parts)

    # 构建 prompt
    prompt = f"""你是一个智能助手，请基于以下检索到的记忆信息回答用户的问题。

检索到的记忆：
{context}

用户问题：{question}

请根据上述记忆信息，准确、简洁地回答用户的问题。如果记忆中没有相关信息，请诚实地说明。

要求：
1. 直接回答具体领域或专业名称
2. 如果提到多个领域，用逗号分隔
3. 不要添加多余的解释

答案："""

    # 调用 API
    url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
    headers = {'Authorization': f'Bearer {friday_app_id}'}

    payload = {
        "model": "LongCat-Flash-Chat-Eco",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content']
            return answer
        else:
            return f"API 响应异常：{result}"

    except Exception as e:
        return f"调用 LLM 失败：{e}"


def answer_question(question: str,
                   file_path: str = '/Users/kwjcyr/data/llm_memory/data/effective_memories.txt',
                   friday_app_id: str = '1980915965710716958',
                   top_k: int = 5) -> str:
    """
    主函数：基于 effective memory 回答问题

    Args:
        question: 用户问题
        file_path: effective memories 文件路径
        friday_app_id: Friday App ID
        top_k: 检索的相关记忆数量

    Returns:
        答案
    """
    print("=" * 80)
    print(f"问题：{question}")
    print("=" * 80)

    # 1. 加载记忆
    print("\n1. 加载 effective memories...")
    memories = load_effective_memories(file_path)
    print(f"   共加载 {len(memories)} 条记忆")

    # 2. 检索相关记忆
    print(f"\n2. 检索与问题相关的 top {top_k} 条记忆...")
    relevant_memories = search_relevant_memories(question, memories, top_k)
    print(f"   找到 {len(relevant_memories)} 条相关记忆")

    if not relevant_memories:
        return "抱歉，没有找到与问题相关的记忆信息。"

    # 显示检索到的记忆
    for i, memory in enumerate(relevant_memories, 1):
        print(f"   记忆{i}: {memory.get('topic', 'N/A')}")

    # 3. 调用 LLM 生成答案
    print(f"\n3. 调用 LLM 生成答案...")
    answer = call_llm_for_qa(question, relevant_memories, friday_app_id)

    print("\n" + "=" * 80)
    print(f"答案：{answer}")
    print("=" * 80)

    return answer


if __name__ == '__main__':
    # 示例问题
    question = "What fields would Caroline be likely to pursue in her education?"

    # 回答问题
    answer = answer_question(question)

