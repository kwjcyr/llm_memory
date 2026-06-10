"""
基于强化学习的 Prompt 优化系统
通过 LocoMo 基准测试自动优化 effective.json 中的 prompt
"""
import copy
import json
import random
from typing import List, Dict, Any, Tuple

import requests


class PromptOptimizer:
    """Prompt 优化器"""

    def __init__(self,
                 effective_config_path: str = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json',
                 locomo_data_path: str = '/Users/kwjcyr/data/llm_memory/data/input/locomo10.json',
                 effective_memories_path: str = '/Users/kwjcyr/data/llm_memory/data/effective_memories.txt'):
        """
        初始化优化器

        Args:
            effective_config_path: effective.json 配置文件路径
            locomo_data_path: LocoMo 测试数据路径
            effective_memories_path: effective memories 文件路径
        """
        self.effective_config_path = effective_config_path
        self.locomo_data_path = locomo_data_path
        self.effective_memories_path = effective_memories_path

        # 加载配置
        self.config = self.load_effective_config()
        self.locomo_data = self.load_locomo_data()
        self.effective_memories = self.load_effective_memories()

        # 优化历史
        self.optimization_history = []

        # 可优化的参数
        self.optimizable_params = {
            'temperature': [0.1, 0.2, 0.3, 0.5, 0.7],
            'top_p': [0.9, 0.95, 1.0],
            'maxToken': [2048, 4096, 8192]
        }

        # prompt 模板的可变部分
        self.prompt_variations = {
            'instruction_prefix': [
                "You are a personal information organizer",
                "You are an expert memory extractor",
                "You specialize in extracting facts from conversations"
            ],
            'output_format_emphasis': [
                "Return exact facts with specific details",
                "Focus on precise dates and names",
                "Extract concrete information with timestamps"
            ]
        }

    def load_effective_config(self) -> Dict[str, Any]:
        """加载 effective.json 配置"""
        with open(self.effective_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_effective_config(self):
        """保存 effective.json 配置"""
        with open(self.effective_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def load_locomo_data(self) -> List[Dict[str, Any]]:
        """加载 LocoMo 测试数据"""
        with open(self.locomo_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_effective_memories(self) -> List[Dict[str, Any]]:
        """加载 effective memories"""
        memories = []
        with open(self.effective_memories_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    memory = json.loads(line.strip())
                    memories.append(memory)
                except:
                    continue
        return memories

    def calculate_similarity(self, predicted: str, ground_truth: str) -> float:
        """
        计算预测答案与标准答案的相似度（作为 reward）

        Args:
            predicted: 预测答案
            ground_truth: 标准答案

        Returns:
            相似度分数 0-1
        """
        # 简单的字符串匹配
        predicted_lower = predicted.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()

        # 完全匹配
        if predicted_lower == ground_truth_lower:
            return 1.0

        # 包含关键词匹配
        ground_truth_words = set(ground_truth_lower.split())
        predicted_words = set(predicted_lower.split())

        # 计算重叠度
        intersection = ground_truth_words & predicted_words
        if len(ground_truth_words) > 0:
            overlap_score = len(intersection) / len(ground_truth_words)
        else:
            overlap_score = 0

        # 日期特殊处理
        import re
        date_pattern = r'\d{1,2}\s+\w+\s+\d{4}'
        if re.search(date_pattern, ground_truth_lower) and re.search(date_pattern, predicted_lower):
            # 如果都是日期格式，给予更高的权重
            overlap_score = max(overlap_score, 0.5)

        return overlap_score

    def answer_question(self, question: str, config: Dict[str, Any]) -> str:
        """
        使用给定的配置回答问题

        Args:
            question: 问题
            config: 配置

        Returns:
            答案
        """
        # 检索相关记忆（简化版）
        relevant_memories = self.search_relevant_memories(question, top_k=3)

        if not relevant_memories:
            return "No relevant information found"

        # 构建上下文
        context = self.build_context(relevant_memories)

        # 构建 prompt
        prompt_template = config['extractParam']['promptTemplate']
        prompt_suffix = config['extractParam']['promptSuffix']
        full_prompt = prompt_template + prompt_suffix.replace('{{{conversation}}}', context)

        # 添加答题指令
        full_prompt += f"\n\nQuestion: {question}\n\nPlease answer based on the conversation above. Be precise and concise."

        # 调用 LLM
        llm_config = config['extractParam']['llmModelParam']
        friday_app_id = config.get('fridayAppId', '1980915965710716958')

        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {friday_app_id}'}

        payload = {
            "model": llm_config.get('modelName', 'LongCat-Flash-Chat-Eco'),
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": llm_config.get('temperature', 0.1),
            "top_p": llm_config.get('topP', 1),
            "max_tokens": llm_config.get('maxToken', 4096),
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return "API error"
        except Exception as e:
            return f"Error: {str(e)}"

    def search_relevant_memories(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索相关记忆（简化版）"""
        question_keywords = set(question.lower().split())

        scored_memories = []
        for memory in self.effective_memories:
            score = 0
            topic = memory.get('topic', '').lower()
            summary = memory.get('summary', '').lower()

            if any(kw in topic for kw in question_keywords):
                score += 3
            if any(kw in summary for kw in question_keywords):
                score += 2

            if score > 0:
                scored_memories.append((score, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored_memories[:top_k]]

    def build_context(self, memories: List[Dict[str, Any]]) -> str:
        """构建对话上下文"""
        lines = []
        for memory in memories:
            original_text = memory.get('original_text', '')
            if original_text:
                lines.append(original_text)
        return "\n".join(lines)

    def evaluate_current_config(self, sample_size: int = 10) -> Tuple[float, List[Dict]]:
        """
        评估当前配置的性能

        Args:
            sample_size: 测试问题数量

        Returns:
            (平均分数, 详细结果列表)
        """
        results = []
        total_score = 0

        # 随机选择 sample_size 个问题
        test_questions = random.sample(self.locomo_data, min(sample_size, len(self.locomo_data)))

        for qa_pair in test_questions:
            question = qa_pair['question']
            ground_truth = str(qa_pair['answer'])

            predicted = self.answer_question(question, self.config)
            score = self.calculate_similarity(predicted, ground_truth)

            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'score': score
            })

            total_score += score

        avg_score = total_score / len(results) if results else 0
        return avg_score, results

    def mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        变异配置（生成新的配置变体）

        Args:
            config: 原始配置

        Returns:
            变异后的配置
        """
        mutated = copy.deepcopy(config)

        # 变异 LLM 参数
        llm_config = mutated['extractParam']['llmModelParam']
        llm_config['temperature'] = random.choice(self.optimizable_params['temperature'])
        llm_config['top_p'] = random.choice(self.optimizable_params['top_p'])
        llm_config['maxToken'] = random.choice(self.optimizable_params['maxToken'])

        # 变异 prompt（简单添加强调）
        emphasis = random.choice(self.prompt_variations['output_format_emphasis'])
        original_prompt = mutated['extractParam']['promptTemplate']

        # 在 prompt 末尾添加强调
        if 'Output Format' in original_prompt:
            mutated['extractParam']['promptTemplate'] = original_prompt.replace(
                'Output Format',
                f'{emphasis}. Output Format'
            )

        return mutated

    def optimize(self, generations: int = 10, sample_size: int = 10):
        """
        执行优化

        Args:
            generations: 优化代数
            sample_size: 每代测试的问题数量
        """
        print("=" * 80)
        print("开始 Prompt 强化学习优化")
        print("=" * 80)

        best_score = 0
        best_config = None

        for gen in range(generations):
            print(f"\n第 {gen + 1}/{generations} 代优化")
            print("-" * 80)

            # 评估当前配置
            print("评估当前配置...")
            current_score, current_results = self.evaluate_current_config(sample_size)
            print(f"当前配置平均分：{current_score:.4f}")

            # 生成变异配置
            print("生成变异配置...")
            mutated_config = self.mutate_config(self.config)

            # 评估变异配置
            print("评估变异配置...")
            mutated_score, mutated_results = self.evaluate_current_config(sample_size)
            print(f"变异配置平均分：{mutated_score:.4f}")

            # 选择更好的配置
            if mutated_score > current_score:
                print(f"✅ 接受变异（提升：{mutated_score - current_score:.4f}）")
                self.config = mutated_config
                current_score = mutated_score
            else:
                print(f"❌ 拒绝变异")

            # 记录历史
            self.optimization_history.append({
                'generation': gen + 1,
                'score': current_score,
                'config': copy.deepcopy(self.config)
            })

            # 更新最佳配置
            if current_score > best_score:
                best_score = current_score
                best_config = copy.deepcopy(self.config)
                print(f"🏆 新的最佳分数：{best_score:.4f}")

        # 保存最佳配置
        if best_config:
            print(f"\n保存最佳配置到 {self.effective_config_path}")
            print(f"最佳分数：{best_score:.4f}")
            self.config = best_config
            self.save_effective_config()

        print("\n" + "=" * 80)
        print("优化完成!")
        print("=" * 80)

        # 打印优化历史
        print("\n优化历史:")
        for record in self.optimization_history:
            print(f"  代{record['generation']}: {record['score']:.4f}")

        return best_score, best_config


if __name__ == '__main__':
    # 创建优化器
    optimizer = PromptOptimizer()

    # 执行优化
    best_score, best_config = optimizer.optimize(generations=5, sample_size=5)

    print(f"\n最终最佳分数：{best_score:.4f}")

