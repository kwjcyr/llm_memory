"""
基于 PPO 的 Prompt 优化器
使用强化学习自动搜索最优 prompt
"""
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import requests


class PPOPromptOptimizer:
    """使用 PPO 算法优化 prompt"""

    def __init__(self,
                 hidden_size: int = 64,
                 learning_rate: float = 0.01,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 2):
        """
        初始化 PPO 优化器

        Args:
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO 裁剪参数
            epochs: 每次更新迭代次数
            batch_size: 批次大小
        """
        self.effective_config_path = '/Users/kwjcyr/data/llm_memory/config/prompt/effective.json'
        self.session_memories_path = '/Users/kwjcyr/data/llm_memory/data/caroline_memories.txt'
        self.locomo_data_path = '/Users/kwjcyr/data/llm_memory/data/input/locomo10.json'

        # 加载配置
        with open(self.effective_config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 加载 session memories
        self.session_memories = []
        with open(self.session_memories_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.session_memories.append(json.loads(line.strip()))
                except:
                    pass

        # 加载 LocoMo 数据
        with open(self.locomo_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            if isinstance(raw_data, list) and len(raw_data) > 0 and 'qa' in raw_data[0]:
                self.locomo_data = []
                for item in raw_data:
                    if 'qa' in item:
                        self.locomo_data.extend(item['qa'])
            else:
                self.locomo_data = raw_data

        # Prompt 动作空间（离散的 prompt 组件）
        self.prompt_components = [
            "\n\nIMPORTANT: Provide exact dates, names, and specific details.",
            "\n\nBe precise: extract exact facts with dates, times, and proper nouns.",
            "\n\nFocus on accuracy: answer with specific information mentioned in the text.",
            "\n\nExtract concrete facts: include exact dates, locations, and identities.",
            "\n\nPay attention to all details including objects, events, and preferences.",
            "\n\nRemember specific dates, numbers, and proper nouns.",
            "",  # 空 prompt
        ]

        self.num_actions = len(self.prompt_components)

        # PPO 超参数
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        # 初始化策略网络参数（简单的线性网络）
        np.random.seed(42)
        self.params = self._init_network()

        # 轨迹存储
        self.trajectories = []

        print(f"初始化 PPO 优化器")
        print(f"  动作空间大小：{self.num_actions}")
        print(f"  隐藏层大小：{hidden_size}")
        print(f"  学习率：{learning_rate}")
        print(f"  Session memories: {len(self.session_memories)}")
        print(f"  LocoMo QA: {len(self.locomo_data)}")

    def _init_network(self) -> Dict[str, np.ndarray]:
        """初始化策略网络参数"""
        # 输入：prompt 的 one-hot 编码 + 当前得分 (num_actions + 1)
        # 输出：选择每个动作的概率 (num_actions)
        input_size = self.num_actions + 1  # one-hot + 当前分数

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
        """ReLU 激活函数"""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def policy_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播，返回动作概率分布和隐藏层状态"""
        # state: (num_actions + 1,) - one-hot + score

        h1 = self._relu(np.dot(state.reshape(1, -1), self.params['W1']) + self.params['b1'])
        h2 = self._relu(np.dot(h1, self.params['W2']) + self.params['b2'])
        logits = np.dot(h2, self.params['W3']) + self.params['b3']

        probs = self._softmax(logits)
        return probs, h2

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """根据策略选择动作（带 epsilon-greedy 探索）"""
        probs, _ = self.policy_forward(state)

        # epsilon-greedy 探索
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.random.choice(self.num_actions, p=probs[0])

    def build_state(self, current_score: float, history: List[int]) -> np.ndarray:
        """构建状态向量"""
        # 使用最近的动作历史 + 当前分数
        state = np.zeros(self.num_actions + 1)

        # 编码动作历史（最近 3 个动作）
        recent_actions = history[-3:] if len(history) >= 3 else history
        for action in recent_actions:
            state[action] += 1.0

        # 归一化
        if len(recent_actions) > 0:
            state[:self.num_actions] /= len(recent_actions)

        # 添加当前分数
        state[-1] = current_score

        return state

    def compute_reward(self, prompt_addition: str, sample_indices: List[int]) -> float:
        """计算奖励（使用 QA 得分）"""
        # 抽取 effective memories
        effective_memories = self._extract_memories(prompt_addition)

        if not effective_memories:
            return 0.0

        # 评估 QA 得分
        total_score = 0.0
        for idx in sample_indices:
            qa = self.locomo_data[idx]
            question = qa.get('question', '')
            ground_truth = str(qa.get('answer', ''))

            answer = self._answer_question(question, effective_memories)
            score = self._calculate_score(answer, ground_truth)
            total_score += score

        avg_score = total_score / len(sample_indices) if sample_indices else 0.0
        return avg_score

    def _extract_memories(self, prompt_addition: str) -> List[Dict]:
        """抽取 effective memories"""
        base_prompt = self.config['extractParam']['promptTemplate']

        # 格式化对话
        conversation_lines = []
        for memory in self.session_memories[:50]:  # 限制长度
            user_content = memory.get('user_content', '')
            assistant_content = memory.get('assistant_content', '')
            if user_content:
                conversation_lines.append(f"User: {user_content}")
            if assistant_content:
                conversation_lines.append(f"Assistant: {assistant_content}")

        conversation_text = "\n".join(conversation_lines)

        instruction_prefix = """
# IMPORTANT INSTRUCTION
The following conversation contains REAL data that you need to extract memories from.
DO NOT copy names or events from the examples above.
ONLY extract information from the actual conversation below.
Identify the REAL names mentioned in the conversation (e.g., Caroline, Melanie, etc.) and use THOSE names in your output.

"""

        full_prompt = base_prompt + prompt_addition + instruction_prefix
        final_prompt = full_prompt.replace('{{{conversation}}}', conversation_text)

        # 调用 LLM
        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId", "1980915965710716958")}'}

        llm_config = self.config['extractParam']['llmModelParam']
        payload = {
            "model": llm_config.get('modelName', 'LongCat-Flash-Chat-Eco'),
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0.1,
            "top_p": 1,
            "max_tokens": llm_config.get('maxToken', 4096),
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            llm_output = result['choices'][0]['message']['content']

            # 解析 JSON
            if '```json' in llm_output:
                llm_output = llm_output.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_output:
                llm_output = llm_output.split('```')[1].split('```')[0].strip()

            extracted = json.loads(llm_output)

            if isinstance(extracted, dict) and 'topic' in extracted:
                return [extracted]
            return []

        except Exception as e:
            print(f"    抽取失败：{e}")
            return []

    def _answer_question(self, question: str, memories: List[Dict]) -> str:
        """使用记忆回答问题"""
        # 简单检索
        keywords = set(question.lower().split())
        context_parts = []

        for mem in memories:
            score = 0
            text = (mem.get('topic', '') + ' ' + mem.get('summary', '')).lower()
            if any(kw in text for kw in keywords):
                score += 1

            if score > 0:
                context_parts.append(f"Topic: {mem.get('topic', '')}")
                context_parts.append(f"Summary: {mem.get('summary', '')[:200]}")
                context_parts.append(f"Facts: {', '.join(mem.get('facts', []))[:200]}")

        if not context_parts:
            return "No information"

        context = "\n".join(context_parts)

        prompt = f"""Based on the following memories:
{context}

Question: {question}

Answer precisely and concisely:"""

        url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
        headers = {'Authorization': f'Bearer {self.config.get("fridayAppId")}'}

        payload = {
            "model": "LongCat-Flash-Chat-Eco",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "top_p": 1,
            "max_tokens": 512,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            result = response.json()
            return result['choices'][0]['message']['content']
        except:
            return "Error"

    def _calculate_score(self, predicted: str, ground_truth: str) -> float:
        """计算得分"""
        pred = predicted.lower().strip()
        truth = str(ground_truth).lower().strip()

        if pred == truth:
            return 1.0

        truth_words = set(truth.split())
        pred_words = set(pred.split())

        if len(truth_words) == 0:
            return 0.0

        overlap = len(truth_words & pred_words)
        return overlap / len(truth_words)

    def compute_gae(self, rewards: List[float], values: List[float]) -> List[float]:
        """计算 Generalized Advantage Estimation"""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)

        return advantages

    def update_policy(self, states: List[np.ndarray], actions: List[int],
                     advantages: List[float], old_probs: List[np.ndarray]):
        """使用 PPO 更新策略"""
        if len(states) == 0:
            return

        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)

        # PPO 更新
        for epoch in range(self.epochs):
            # 计算当前策略概率
            current_probs, _ = self.policy_forward(states[0])  # 简化：只用第一个状态

            # 计算概率比
            action_probs = current_probs[0, actions]
            old_action_probs = np.array([old_probs[i][0, actions[i]] for i in range(len(actions))])
            ratio = action_probs / (old_action_probs + 1e-8)

            # 计算 clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            loss = -np.minimum(surr1, surr2).mean()

            # 简化梯度更新（使用简化版策略梯度）
            for i, action in enumerate(actions):
                adv = advantages[i]
                if adv > 0:
                    # 增加选择该动作的概率
                    self.params['W3'][:, action] += self.lr * adv * 0.1
                elif adv < 0:
                    # 减少选择该动作的概率
                    self.params['W3'][:, action] -= self.lr * abs(adv) * 0.1

        print(f"  策略已更新 (epoch={self.epochs}, loss={loss:.4f})")

    def train(self, num_episodes: int = 20, sample_size: int = 3) -> Dict[str, Any]:
        """训练 PPO 策略"""
        print("\n" + "=" * 80)
        print("开始 PPO 训练")
        print("=" * 80)
        print(f"训练轮数：{num_episodes}")
        print(f"每轮样本数：{sample_size}")
        print("=" * 80)

        best_score = 0.0
        best_prompt = ""
        history = []

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            # 选择动作（prompt）
            current_score = best_score
            state = self.build_state(current_score, history)
            action = self.select_action(state, epsilon=0.3)  # 30% 探索

            prompt = self.prompt_components[action]
            print(f"  选择 Prompt {action}: {prompt[:60]}..." if prompt else "  选择 Prompt {action}: [空]")

            # 执行动作并计算奖励
            reward = self.compute_reward(prompt,
                                        np.random.choice(len(self.locomo_data),
                                                       min(sample_size, len(self.locomo_data)),
                                                       replace=False).tolist())

            print(f"  奖励：{reward:.4f}")

            # 记录轨迹
            self.trajectories.append({
                'state': state,
                'action': action,
                'reward': reward,
                'prompt': prompt
            })

            history.append(action)

            if reward > best_score:
                best_score = reward
                best_prompt = prompt
                print(f"  🏆 新的最佳！分数：{best_score:.4f}")

            # 每 5 轮更新一次策略
            if (episode + 1) % 5 == 0 and len(self.trajectories) >= 5:
                print("\n  更新策略...")

                # 准备数据
                states = [t['state'] for t in self.trajectories[-5:]]
                actions = [t['action'] for t in self.trajectories[-5:]]
                rewards = [t['reward'] for t in self.trajectories[-5:]]

                # 计算价值（简单用平均奖励）
                values = [np.mean(rewards)] * len(states)

                # 计算优势
                advantages = self.compute_gae(rewards, values)

                # 计算旧概率
                old_probs = [self.policy_forward(s)[0] for s in states]

                # 更新策略
                self.update_policy(states, actions, advantages, old_probs)

                # 清空轨迹
                self.trajectories = []

        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳分数：{best_score:.4f}")
        print(f"最佳 Prompt: {best_prompt[:100] if best_prompt else '[空]'}")
        print("=" * 80)

        return {
            'best_score': best_score,
            'best_prompt': best_prompt,
            'trajectories': self.trajectories
        }


if __name__ == '__main__':
    optimizer = PPOPromptOptimizer(
        hidden_size=32,
        learning_rate=0.01,
        gamma=0.99,
        epochs=5,
        batch_size=2
    )

    results = optimizer.train(num_episodes=15, sample_size=3)

    print("\n训练结果:")
    print(f"最佳分数：{results['best_score']:.4f}")
    print(f"最佳 Prompt: {results['best_prompt'][:100] if results['best_prompt'] else '[空]'}")

