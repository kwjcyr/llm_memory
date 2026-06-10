"""
简化的 Prompt 优化器 - 专注于优化 effective.json 中的 prompt 模板
使用强化学习自动调整 prompt 以提高 LocoMo 测试分数
"""
import json
import os
from typing import List, Dict, Tuple, Any

import requests


class SimplePromptOptimizer:
    """简化的 Prompt 优化器 - 通过优化 effective.json 的 prompt 提高 LocoMo 召回效果"""

    def __init__(self):
        self.effective_config_path = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json'
        self.session_memories_path = '/Users/kwjcyr/data/llm_memory/data/caroline_memories.txt'
        self.locomo_data_path = '/Users/kwjcyr/data/llm_memory/data/input/locomo10.json'

        # 临时存储抽取的 effective memories
        self.temp_effective_memories_path = '/Users/kwjcyr/data/llm_memory/data/temp_effective_memories.txt'

        # 加载配置
        with open(self.effective_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载 LocoMo 数据
        with open(self.locomo_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # locomo10.json 格式：外层是 10 个对话组，每组包含 qa 列表
            # 需要展开所有 qa 对
            if isinstance(raw_data, list) and len(raw_data) > 0 and 'qa' in raw_data[0]:
                self.locomo_data = []
                for item in raw_data:
                    if 'qa' in item and isinstance(item['qa'], list):
                        self.locomo_data.extend(item['qa'])
            elif isinstance(raw_data, list):
                # 直接是 QA 列表
                self.locomo_data = raw_data
            else:
                self.locomo_data = [raw_data]

        # 调试：检查第一个 QA 对的结构
        if len(self.locomo_data) > 0:
            print(f"加载了 {len(self.locomo_data)} 个 QA 对")
            print(f"第一个 QA 对的 keys: {list(self.locomo_data[0].keys())}")

        # 加载 session memories（原始对话）
        self.session_memories = []
        with open(self.session_memories_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    memory = json.loads(line.strip())
                    self.session_memories.append(memory)
                except:
                    pass

        print(f"加载了 {len(self.session_memories)} 条 session memories")

        # 初始化 effective memories（用于检索）
        self.effective_memories = []

        # Prompt 变体模板（添加到 promptTemplate 的强化语句）
        self.prompt_additions = [
            "\n\nIMPORTANT: Provide exact dates, names, and specific details from the conversation.",
            "\n\nBe precise: extract exact facts with dates, times, and proper nouns.",
            "\n\nFocus on accuracy: answer with specific information mentioned in the text.",
            "\n\nExtract concrete facts: include exact dates, locations, and identities.",
            ""
        ]

        self.best_score = 0
        self.best_addition = ""
        self.best_effective_memories = []

    def extract_effective_memories(self, prompt_addition: str = "") -> List[Dict]:
        """使用特定 prompt 从 session memories 抽取 effective memories"""
        print(f"  正在抽取 effective memories...")

        # 使用完整的 prompt（原始模板 + 变体）
        base_prompt = self.config['extractParam']['promptTemplate']

        # 格式化所有对话
        conversation_lines = []
        print(f"  使用 {len(self.session_memories)} 条 session memories 进行抽取")
        # 显示前 3 条作为 sample
        for i, memory in enumerate(self.session_memories):
            user_content = memory.get('user_content', '')
            assistant_content = memory.get('assistant_content', '')
            if i < 3 and user_content:
                print(f"    Sample {i+1}: {user_content[:100]}...")
            if user_content:
                conversation_lines.append(f"User: {user_content}")
            if assistant_content:
                conversation_lines.append(f"Assistant: {assistant_content}")

        conversation_text = "\n".join(conversation_lines)

        # 添加明确的指令，告诉 LLM 要从实际对话中抽取，而不是模仿示例中的人名
        instruction_prefix = """
# IMPORTANT INSTRUCTION
The following conversation contains REAL data that you need to extract memories from.
DO NOT copy names or events from the examples above.
ONLY extract information from the actual conversation below.
Identify the REAL names mentioned in the conversation (e.g., Caroline, Melanie, etc.) and use THOSE names in your output.

"""

        # 构建最终 prompt
        full_prompt = base_prompt + prompt_addition + instruction_prefix
        final_prompt = full_prompt.replace('{{{conversation}}}', conversation_text)

        # 调用 LLM 进行抽取
        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId", "1980915965710716958")}'}

        llm_config = self.config['extractParam']['llmModelParam']
        payload = {
            "model": llm_config.get('modelName', 'LongCat-Flash-Chat-Eco'),
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": llm_config.get('temperature', 0.1),
            "top_p": llm_config.get('topP', 1),
            "max_tokens": llm_config.get('maxToken', 4096),
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            llm_output = result['choices'][0]['message']['content']

            # 解析 LLM 输出的 JSON
            # LLM 可能返回 markdown 包裹的 JSON，需要清理
            if '```json' in llm_output:
                llm_output = llm_output.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_output:
                llm_output = llm_output.split('```')[1].split('```')[0].strip()

            extracted_memory = json.loads(llm_output)

            # 将抽取的记忆包装成列表，并添加元数据
            effective_memories = []

            # 如果返回的是单个 memory
            if isinstance(extracted_memory, dict) and 'topic' in extracted_memory:
                effective_memories.append(extracted_memory)
            # 如果返回的是 memory 列表（从 facts 数组构造多个 memories）
            elif isinstance(extracted_memory, dict) and 'facts' in extracted_memory:
                # 将整个抽取结果作为一个 effective memory
                effective_memories.append(extracted_memory)

            print(f"  成功抽取 {len(effective_memories)} 条 effective memories")
            return effective_memories

        except Exception as e:
            print(f"  抽取失败：{e}")
            return []

    def save_effective_memories(self, memories: List[Dict], output_path: str):
        """保存 effective memories 到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for memory in memories:
                f.write(json.dumps(memory, ensure_ascii=False) + '\n')

    def search_memories(self, question: str, top_k: int = 3) -> List[Dict]:
        """检索相关记忆"""
        keywords = set(question.lower().split())
        scored = []

        for mem in self.effective_memories:
            score = 0
            text = (mem.get('topic', '') + ' ' + mem.get('summary', '')).lower()
            if any(kw in text for kw in keywords):
                score += 1
            if score > 0:
                scored.append((score, mem))

        # 按 score 排序
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored[:top_k]]

    def answer_with_prompt(self, question: str, prompt_addition: str) -> str:
        """使用特定 prompt 变体回答问题"""
        memories = self.search_memories(question)

        if not memories:
            return "No information"

        # 构建上下文
        context_parts = []
        for mem in memories:
            context_parts.append(f"- Topic: {mem.get('topic', '')}")
            context_parts.append(f"- Summary: {mem.get('summary', '')}")
            context_parts.append(f"- Facts: {', '.join(mem.get('facts', []))}")
            if mem.get('time_range'):
                context_parts.append(f"- Time: {mem['time_range'].get('start', '')} to {mem['time_range'].get('end', '')}")
            if mem.get('original_text'):
                context_parts.append(f"- Original: {mem.get('original_text', '')[:300]}")

        context = "\n".join(context_parts)

        # 使用 effective.json 中的 promptTemplate，添加检索到的记忆和问题
        base_prompt = self.config['extractParam']['promptTemplate']

        full_prompt = f"""{base_prompt}

# Retrieved Memories
Based on the following extracted memories from the conversation:
{context}

# Task
Please answer the following question based ONLY on the above retrieved memories.
If the information is not available in the memories, please say you don't have that information.

{prompt_addition}

# Question
{question}

# Answer
Please answer precisely and concisely based on the provided memories:"""

        # 调用 LLM
        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId", "1980915965710716958")}'}

        llm_config = self.config['extractParam']['llmModelParam']
        payload = {
            "model": llm_config.get('modelName', 'LongCat-Flash-Chat-Eco'),
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": llm_config.get('temperature', 0.1),
            "top_p": llm_config.get('topP', 1),
            "max_tokens": llm_config.get('maxToken', 1024),
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"    LLM 调用失败：{e}")
            return "Error"

    def calculate_score(self, predicted: str, ground_truth: str) -> float:
        """计算答案得分"""
        pred = predicted.lower().strip()
        truth = str(ground_truth).lower().strip()

        if pred == truth:
            return 1.0

        # 部分匹配
        truth_words = set(truth.split())
        pred_words = set(pred.split())

        if len(truth_words) == 0:
            return 0

        overlap = len(truth_words & pred_words)
        return overlap / len(truth_words)

    def evaluate_prompt(self, prompt_addition: str, sample_indices: List[int]) -> Tuple[float, List[Dict]]:
        """评估特定 prompt 变体的性能"""
        # 第一步：使用这个 prompt 从 session memories 抽取 effective memories
        print(f"\n  第一步：使用当前 prompt 变体抽取 effective memories...")
        effective_memories = self.extract_effective_memories(prompt_addition)

        if not effective_memories:
            print("  警告：未能抽取到 effective memories，跳过评估")
            return 0.0, []

        # 保存抽取的 memories 用于后续问答
        temp_memories = self.effective_memories  # 保存旧的
        self.effective_memories = effective_memories  # 使用新的

        print(f"\n  第二步：使用抽取的 memories 评估 {len(sample_indices)} 个 LocoMo 问题...")

        results = []
        total_score = 0

        for idx in sample_indices:
            qa = self.locomo_data[idx]
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')

            if not question or not ground_truth:
                print(f"  警告：索引 {idx} 的 QA 对缺少 question 或 answer 字段")
                continue

            # 使用抽取的 memories 来回答问题（不再添加额外的 prompt）
            predicted = self.answer_from_memories(question)
            score = self.calculate_score(predicted, ground_truth)

            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'score': score
            })

            total_score += score
            print(f"    问题 {len(results)}: 得分 {score:.4f}")

        avg_score = total_score / len(results) if results else 0

        # 恢复旧的 memories
        self.effective_memories = temp_memories

        return avg_score, results

    def answer_from_memories(self, question: str) -> str:
        """使用当前的 effective_memories 回答问题（不添加额外的 prompt）"""
        memories = self.search_memories(question)

        if not memories:
            return "No information found in memories"

        # 构建上下文
        context_parts = []
        for mem in memories:
            context_parts.append(f"- Topic: {mem.get('topic', '')}")
            context_parts.append(f"- Summary: {mem.get('summary', '')}")
            context_parts.append(f"- Facts: {', '.join(mem.get('facts', []))}")
            if mem.get('time_range'):
                context_parts.append(f"- Time: {mem['time_range'].get('start', '')} to {mem['time_range'].get('end', '')}")
            if mem.get('original_text'):
                context_parts.append(f"- Original: {mem.get('original_text', '')[:300]}")

        context = "\n".join(context_parts)

        # 构建简单的回答 prompt
        full_prompt = f"""Based on the following extracted memories:
{context}

Question: {question}

Please answer precisely and concisely based ONLY on the above memories. If the information is not available, say you don't have that information.

Answer:"""

        # 调用 LLM
        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId", "1980915965710716958")}'}

        llm_config = self.config['extractParam']['llmModelParam']
        payload = {
            "model": llm_config.get('modelName', 'LongCat-Flash-Chat-Eco'),
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.1,
            "top_p": 1,
            "max_tokens": 512,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"    LLM 调用失败：{e}")
            return "Error"

    def run_optimization(self, generations: int = 3, sample_size: int = 5):
        """运行优化"""
        print("=" * 80)
        print("Prompt 强化学习优化")
        print("=" * 80)

        # 随机选择测试样本（优先选择 Caroline 相关的问题）
        import random

        # 首先尝试找到 Caroline 相关的问题索引
        caroline_indices = []
        for i, qa in enumerate(self.locomo_data):
            question = str(qa.get('question', '')).lower()
            answer = str(qa.get('answer', '')).lower()
            if 'caroline' in question or 'caroline' in answer:
                caroline_indices.append(i)

        print(f"\n找到 {len(caroline_indices)} 个 Caroline 相关的问题")

        # 如果 Caroline 相关问题足够，只测试这些；否则混合其他问题
        if len(caroline_indices) >= sample_size:
            sample_indices = random.sample(caroline_indices, sample_size)
            print(f"使用 {sample_size} 个 Caroline 相关问题作为测试样本")
        else:
            # 混合所有问题
            all_indices = list(range(len(self.locomo_data)))
            sample_indices = random.sample(all_indices, min(sample_size, len(all_indices)))
            print(f"使用 {len(sample_indices)} 个混合问题作为测试样本（Caroline: {len(caroline_indices)}）")

        print(f"优化代数：{generations}")

        # 评估每个 prompt 变体
        for i, addition in enumerate(self.prompt_additions):
            print(f"\n测试 Prompt 变体 {i + 1}/{len(self.prompt_additions)}:")
            print(f"  变体：{addition[:80]}..." if addition else "  变体：[无额外提示]")

            score, results = self.evaluate_prompt(addition, sample_indices)
            print(f"  平均分：{score:.4f}")

            if score > self.best_score:
                self.best_score = score
                self.best_addition = addition
                print(f"  🏆 新的最佳!")

        print("\n" + "=" * 80)
        print(f"优化完成!")
        print(f"最佳分数：{self.best_score:.4f}")
        print(f"最佳 prompt 添加：{self.best_addition[:100] if self.best_addition else '[无]'}")
        print("=" * 80)

        # 显示详细结果
        print("\n最佳配置的详细结果:")
        for i, addition in enumerate(self.prompt_additions):
            if addition == self.best_addition:
                score, results = self.evaluate_prompt(addition, sample_indices)
                for r in results:
                    print(f"\n  Q: {r['question']}")
                    print(f"  A (predicted): {r['predicted']}")
                    print(f"  A (ground truth): {r['ground_truth']}")
                    print(f"  Score: {r['score']:.4f}")

        return self.best_score, self.best_addition

    def evaluate_baseline(self, sample_indices: List[int]) -> Tuple[float, List[Dict]]:
        """评估原始 prompt（优化前）的性能"""
        print("\n" + "=" * 80)
        print("评估原始 Prompt (Before Optimization)")
        print("=" * 80)
        return self.evaluate_prompt("", sample_indices)

    def evaluate_optimized(self, sample_indices: List[int]) -> Tuple[float, List[Dict]]:
        """评估优化后 prompt 的性能"""
        print("\n" + "=" * 80)
        print(f"评估优化后 Prompt (After Optimization)")
        print("=" * 80)
        return self.evaluate_prompt(self.best_addition, sample_indices)

    def compare_before_after(self, sample_size: int = 10) -> Dict[str, Any]:
        """对比优化前后的效果"""
        import random

        # 找到 Caroline 相关的问题
        caroline_indices = []
        for i, qa in enumerate(self.locomo_data):
            question = str(qa.get('question', '')).lower()
            answer = str(qa.get('answer', '')).lower()
            if 'caroline' in question or 'caroline' in answer:
                caroline_indices.append(i)

        # 优先选择 Caroline 相关问题
        if len(caroline_indices) >= sample_size:
            sample_indices = random.sample(caroline_indices, sample_size)
            print(f"\n使用 {sample_size} 个 Caroline 相关问题进行对比测试")
        else:
            all_indices = list(range(len(self.locomo_data)))
            sample_indices = random.sample(all_indices, min(sample_size, len(all_indices)))
            print(f"\n使用 {len(sample_indices)} 个混合问题进行对比测试（Caroline: {len(caroline_indices)}）")

        print("\n" + "=" * 80)
        print("LocoMo 基准测试 - 优化前后对比")
        print("=" * 80)
        print(f"测试样本数：{sample_size}")
        print("=" * 80)

        # 评估优化前
        before_score, before_results = self.evaluate_baseline(sample_indices)

        # 评估优化后
        after_score, after_results = self.evaluate_optimized(sample_indices)

        # 生成对比报告
        improvement = after_score - before_score
        improvement_rate = (improvement / before_score * 100) if before_score > 0 else float('inf')

        report = {
            'before_score': before_score,
            'after_score': after_score,
            'improvement': improvement,
            'improvement_rate': improvement_rate,
            'before_results': before_results,
            'after_results': after_results,
            'best_addition': self.best_addition
        }

        # 打印详细对比
        print("\n" + "=" * 80)
        print("对比结果")
        print("=" * 80)
        print(f"优化前平均分：{before_score:.4f}")
        print(f"优化后平均分：{after_score:.4f}")
        print(f"绝对提升：{improvement:.4f}")
        print(f"相对提升：{improvement_rate:.2f}%")
        print(f"最佳 Prompt 添加：{self.best_addition[:100] if self.best_addition else '[无]'}")
        print("=" * 80)

        # 打印每个问题的对比
        print("\n详细对比（每个问题）:")
        print("-" * 80)
        for i in range(len(before_results)):
            before_r = before_results[i]
            after_r = after_results[i]

            print(f"\n问题 {i+1}: {before_r['question'][:60]}...")
            print(f"  标准答案：{before_r['ground_truth']}")
            print(f"  优化前：{before_r['predicted']} (得分：{before_r['score']:.4f})")
            print(f"  优化后：{after_r['predicted']} (得分：{after_r['score']:.4f})")

            if after_r['score'] > before_r['score']:
                print(f"  ✅ 提升")
            elif after_r['score'] < before_r['score']:
                print(f"  ❌ 下降")
            else:
                print(f"  ➡️ 持平")

        print("\n" + "=" * 80)
        print("对比完成!")
        print("=" * 80)

        return report


if __name__ == '__main__':
    optimizer = SimplePromptOptimizer()

    # 先运行优化找到最佳 prompt
    optimizer.run_optimization(generations=3, sample_size=5)

    # 然后进行优化前后对比
    report = optimizer.compare_before_after(sample_size=10)

