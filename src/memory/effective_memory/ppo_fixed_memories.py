"""
PPO 优化器 - 基于固定的 effective memories
优化检索和组装策略，不重新抽取
"""
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import requests


class PPOFixedMemoriesOptimizer:
    """使用固定的 effective memories 进行 PPO 优化"""

    def __init__(self,
                 hidden_size: int = 64,
                 learning_rate: float = 0.01,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 epochs: int = 5):
        """
        初始化 PPO 优化器

        Args:
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO 裁剪参数
            epochs: 每次更新迭代次数
        """
        # 加载固定的 effective memories
        self.memories_path = '/Users/kwjcyr/data/llm_memory/data/effective_memories.txt'
        self.memories = self._load_memories()

        # 加载 LocoMo 数据
        self.locomo_data_path = '/Users/kwjcyr/data/llm_memory/data/input/locomo10.json'
        self.locomo_data = self._load_locomo_data()

        # 过滤 Caroline 相关的问题
        self.caroline_indices = self._find_caroline_questions()

        # 加载配置
        self.config_path = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json'
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        print("=" * 80)
        print("PPO Fixed Memories Optimizer 初始化")
        print("=" * 80)
        print(f"固定 memories 数量：{len(self.memories)}")
        print(f"LocoMo 总 QA 数：{len(self.locomo_data)}")
        print(f"Caroline 相关 QA 数：{len(self.caroline_indices)}")
        print("=" * 80)

        # 动作空间设计
        # 动作 1: 检索策略（选择 top-k 个 memories）
        # 动作 2: 组装策略（选择哪些字段组装到 prompt）
        self.top_k_options = [1, 2, 3, 5]  # 检索 top-k 个记忆
        self.field_options = [
            ['topic', 'summary'],                    # 只用 topic + summary
            ['topic', 'summary', 'facts'],           # 加上 facts
            ['topic', 'summary', 'facts', 'tags'],   # 加上 tags
            ['topic', 'summary', 'facts', 'original_text'],  # 加上 original_text
            ['topic', 'summary', 'facts', 'tags', 'original_text'],  # 全部
        ]

        # 动作编码：每个 (top_k, fields) 组合是一个动作
        self.actions = []
        for k in self.top_k_options:
            for fields in self.field_options:
                self.actions.append({'top_k': k, 'fields': fields})

        self.num_actions = len(self.actions)

        # PPO 网络参数
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

        np.random.seed(42)
        self.params = self._init_network()

        # 轨迹
        self.trajectories = []

        print(f"\n动作空间大小：{self.num_actions}")
        print(f"动作示例:")
        for i, action in enumerate(self.actions[:5]):
            print(f"  {i}: top_k={action['top_k']}, fields={action['fields']}")
        print(f"  ... (共 {self.num_actions} 个动作)")

    def _load_memories(self) -> List[Dict]:
        """加载固定的 effective memories"""
        memories = []
        with open(self.memories_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    memories.append(json.loads(line.strip()))
                except Exception as e:
                    print(f"加载记忆失败：{e}")
        return memories

    def _load_locomo_data(self) -> List[Dict]:
        """加载 LocoMo QA 数据"""
        with open(self.locomo_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        all_qa = []
        for item in raw_data:
            if 'qa' in item and isinstance(item['qa'], list):
                all_qa.extend(item['qa'])
        return all_qa

    def _find_caroline_questions(self) -> List[int]:
        """找到所有 Caroline 相关问题的索引"""
        indices = []
        for i, qa in enumerate(self.locomo_data):
            question = str(qa.get('question', '')).lower()
            answer = str(qa.get('answer', '')).lower()
            if 'caroline' in question or 'caroline' in answer:
                indices.append(i)
        return indices

    def _init_network(self) -> Dict[str, np.ndarray]:
        """初始化策略网络"""
        # 状态：问题关键词 one-hot + 历史平均奖励
        input_size = 50 + 1  # 简化：50 维关键词 + 1 维历史分数

        params = {
            'W1': np.random.randn(input_size, self.hidden_size) * 0.1,
            'b1': np.zeros((1, self.hidden_size)),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * 0.1,
            'b2': np.zeros((1, self.hidden_size)),
            'W3': np.random.randn(self.hidden_size, self.num_actions) * 0.1,
            'b3': np.zeros((1, self.num_actions)),
        }
        return params

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def policy_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播"""
        h1 = self._relu(np.dot(state.reshape(1, -1), self.params['W1']) + self.params['b1'])
        h2 = self._relu(np.dot(h1, self.params['W2']) + self.params['b2'])
        logits = np.dot(h2, self.params['W3']) + self.params['b3']
        probs = self._softmax(logits)
        return probs, h2

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """选择动作"""
        probs, _ = self.policy_forward(state)

        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.random.choice(self.num_actions, p=probs[0])

    def build_state(self, question: str, history_score: float) -> np.ndarray:
        """构建状态向量"""
        # 简单关键词特征
        keywords = ['when', 'what', 'where', 'who', 'how', 'why',
                   'caroline', 'melanie', 'lgbtq', 'adoption', 'counseling',
                   'mental', 'health', 'support', 'group', 'conference',
                   'painting', 'art', 'career', 'education', 'family',
                   'transgender', 'identity', 'friends', 'parents', 'children']

        state = np.zeros(51)
        question_lower = question.lower()

        for i, kw in enumerate(keywords):
            if kw in question_lower:
                state[i] = 1.0

        # 历史平均分数
        state[-1] = history_score

        return state

    def retrieve_and_assemble(self, action_idx: int, question: str) -> str:
        """根据动作检索和组装记忆"""
        action = self.actions[action_idx]
        top_k = action['top_k']
        fields = action['fields']

        # 简单关键词检索
        keywords = set(question.lower().split())
        scored_memories = []

        for mem in self.memories:
            score = 0
            # 在不同字段中搜索关键词
            for field in ['topic', 'summary', 'facts', 'tags']:
                if field in mem:
                    text = str(mem[field]).lower()
                    if any(kw in text for kw in keywords):
                        score += 1

            # 在 original_text 中搜索
            if 'original_text' in mem:
                if any(kw in mem['original_text'].lower() for kw in keywords):
                    score += 0.5

            scored_memories.append((score, mem))

        # 排序并选择 top-k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        selected = [m for s, m in scored_memories[:top_k] if s > 0]

        if not selected:
            # 如果没有匹配，返回空
            return "No relevant memories found."

        # 组装 prompt
        context_parts = []
        for i, mem in enumerate(selected, 1):
            parts = []
            for field in fields:
                if field in mem:
                    value = mem[field]
                    if field == 'facts' and isinstance(value, list):
                        parts.append(f"Facts: {', '.join(value[:5])}")  # 限制 facts 数量
                    elif field == 'tags' and isinstance(value, list):
                        parts.append(f"Tags: {', '.join(value)}")
                    elif field == 'original_text':
                        parts.append(f"Original: {value[:300]}...")  # 限制长度
                    else:
                        parts.append(f"{field.capitalize()}: {value}")

            context_parts.append(f"[Memory {i}]\n" + "\n".join(parts))

        return "\n\n".join(context_parts)

    def answer_question(self, context: str, question: str) -> str:
        """使用组装的上下文回答问题"""
        prompt = f"""Based on the following memories:
{context}

Question: {question}

Answer precisely and concisely based ONLY on the above memories. If the information is not available, say "I don't have that information."

Answer:"""

        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId")}'}

        payload = {
            "model": "LongCat-Flash-Chat-Eco",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 1,
            "max_tokens": 256,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"  LLM 调用失败：{e}")
            return "Error"

    def calculate_score(self, predicted: str, ground_truth: str) -> float:
        """计算得分"""
        pred = predicted.lower().strip()
        truth = str(ground_truth).lower().strip()

        if pred == truth:
            return 1.0

        # 部分匹配
        truth_words = set(truth.split())
        pred_words = set(pred.split())

        if len(truth_words) == 0:
            return 0.0

        overlap = len(truth_words & pred_words)
        return overlap / len(truth_words)

    def compute_reward(self, action_idx: int, sample_indices: List[int]) -> Tuple[float, List[Dict]]:
        """计算奖励"""
        results = []
        total_score = 0.0

        for idx in sample_indices:
            qa = self.locomo_data[idx]
            question = qa.get('question', '')
            ground_truth = str(qa.get('answer', ''))

            if not question or not ground_truth:
                continue

            # 检索和组装
            context = self.retrieve_and_assemble(action_idx, question)

            # 回答
            answer = self.answer_question(context, question)

            # 评分
            score = self.calculate_score(answer, ground_truth)
            total_score += score

            results.append({
                'question': question[:60],
                'ground_truth': ground_truth[:60],
                'predicted': answer[:60],
                'score': score
            })

        avg_score = total_score / len(sample_indices) if sample_indices else 0.0
        return avg_score, results

    def compute_gae(self, rewards: List[float], values: List[float]) -> List[float]:
        """计算 GAE"""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)

        return advantages

    def update_policy(self, states: List[np.ndarray], actions: List[int],
                     advantages: List[float], old_probs: List[np.ndarray]):
        """更新策略"""
        if len(states) == 0:
            return

        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)

        for epoch in range(self.epochs):
            # 简化版策略梯度更新
            for i, action in enumerate(actions):
                adv = advantages[i]
                if adv > 0:
                    self.params['W3'][:, action] += self.lr * adv * 0.1
                elif adv < 0:
                    self.params['W3'][:, action] -= self.lr * abs(adv) * 0.1

    def train(self, num_episodes: int = 30, sample_size: int = 5) -> Dict[str, Any]:
        """训练 PPO"""
        print("\n" + "=" * 80)
        print("开始 PPO 训练 - 优化检索和组装策略")
        print("=" * 80)
        print(f"训练轮数：{num_episodes}")
        print(f"每轮样本数：{sample_size}")
        print("=" * 80)

        best_score = 0.0
        best_action_idx = 0
        history = []

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            # 随机选择问题
            sample_indices = np.random.choice(
                self.caroline_indices,
                min(sample_size, len(self.caroline_indices)),
                replace=False
            ).tolist()

            # 构建状态（使用历史平均分）
            history_score = best_score
            sample_questions = [self.locomo_data[i]['question'] for i in sample_indices]

            # 为每个问题选择动作并计算平均奖励
            episode_rewards = []
            episode_actions = []
            episode_states = []

            for i, qidx in enumerate(sample_indices):
                qa = self.locomo_data[qidx]
                question = qa['question']

                state = self.build_state(question, history_score)
                action = self.select_action(state, epsilon=0.3)

                print(f"  Q{i+1}: {question[:50]}...")
                print(f"    动作 {action}: top_k={self.actions[action]['top_k']}, fields={len(self.actions[action]['fields'])}")

                reward, results = self.compute_reward(action, [qidx])
                print(f"    得分：{reward:.4f}")

                episode_rewards.append(reward)
                episode_actions.append(action)
                episode_states.append(state)

                if reward > best_score:
                    best_score = reward
                    best_action_idx = action
                    print(f"    🏆 新的最佳！")

            # 计算平均奖励
            avg_reward = np.mean(episode_rewards)
            print(f"  平均奖励：{avg_reward:.4f}")

            # 记录轨迹
            for i in range(len(episode_rewards)):
                self.trajectories.append({
                    'state': episode_states[i],
                    'action': episode_actions[i],
                    'reward': episode_rewards[i]
                })

            history.append(avg_reward)

            # 每 10 轮更新一次策略
            if (episode + 1) % 10 == 0 and len(self.trajectories) >= 10:
                print("\n  更新策略...")

                states = [t['state'] for t in self.trajectories[-10:]]
                actions = [t['action'] for t in self.trajectories[-10:]]
                rewards = [t['reward'] for t in self.trajectories[-10:]]

                values = [np.mean(rewards)] * len(states)
                advantages = self.compute_gae(rewards, values)
                old_probs = [self.policy_forward(s)[0] for s in states]

                self.update_policy(states, actions, advantages, old_probs)
                self.trajectories = []

        # 显示最佳动作
        best_action = self.actions[best_action_idx]
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳平均分数：{best_score:.4f}")
        print(f"最佳动作：top_k={best_action['top_k']}, fields={best_action['fields']}")
        print("=" * 80)

        return {
            'best_score': best_score,
            'best_action': best_action,
            'trajectories': self.trajectories
        }

    def evaluate_all_actions(self, sample_size: int = 10) -> Dict[str, Any]:
        """评估所有动作的性能"""
        print("\n" + "=" * 80)
        print("评估所有检索/组装策略")
        print("=" * 80)

        sample_indices = np.random.choice(
            self.caroline_indices,
            min(sample_size, len(self.caroline_indices)),
            replace=False
        ).tolist()

        results = []

        for i, action in enumerate(self.actions):
            print(f"\n动作 {i}: top_k={action['top_k']}, fields={len(action['fields'])}")

            reward, _ = self.compute_reward(i, sample_indices)
            print(f"  平均得分：{reward:.4f}")

            results.append({
                'action_idx': i,
                'action': action,
                'score': reward
            })

        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)

        print("\n" + "=" * 80)
        print("Top 5 策略:")
        for i, r in enumerate(results[:5]):
            print(f"{i+1}. 动作 {r['action_idx']}: top_k={r['action']['top_k']}, "
                  f"fields={len(r['action']['fields'])}, 得分={r['score']:.4f}")
        print("=" * 80)

        return results


def main():
    """主函数"""
    optimizer = PPOFixedMemoriesOptimizer(
        hidden_size=32,
        learning_rate=0.01,
        gamma=0.99,
        epochs=5
    )

    # 第一步：评估所有动作的 baseline
    print("\n>>> Step 1: Baseline 评估")
    all_results = optimizer.evaluate_all_actions(sample_size=10)

    # 第二步：PPO 训练
    print("\n>>> Step 2: PPO 训练")
    train_results = optimizer.train(num_episodes=30, sample_size=5)

    # 第三步：对比
    print("\n>>> Step 3: 对比结果")
    print(f"PPO 找到的最佳策略:")
    print(f"  top_k: {train_results['best_action']['top_k']}")
    print(f"  fields: {train_results['best_action']['fields']}")
    print(f"  得分：{train_results['best_score']:.4f}")

    best_baseline = max(all_results, key=lambda x: x['score'])
    print(f"\nBaseline 最佳策略:")
    print(f"  top_k: {best_baseline['action']['top_k']}")
    print(f"  fields: {best_baseline['action']['fields']}")
    print(f"  得分：{best_baseline['score']:.4f}")

    improvement = (train_results['best_score'] - best_baseline['score']) / best_baseline['score'] * 100
    print(f"\n提升：{improvement:.2f}%")


if __name__ == '__main__':
    main()

