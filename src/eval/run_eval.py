"""
Eval 评测脚本（三层记忆版本）
基于 Long-term / Effective / Session 三层记忆对 LocoMo QA 数据集进行评测

输出格式完全对齐 memory_evaluation 项目：
  - locomo_responses.json: 每道题的 Q/A/检索记忆（含三层证据链）
  - locomo_judged.json:   每道题 + LLM Judge 评分 + F1/BLEU 指标
  - locomo_grades.json:   总体统计 + 分类统计

用法：
    python src/eval/run_eval.py --group group_1_gina
    python src/eval/run_eval.py --group group_1_gina --top_k 10

输出：
    data/groups/{group}/eval/
      ├── locomo_responses.json   # 原始回答（含三层证据链）
      ├── locomo_judged.json     # 评分结果（LLM Judge + NLP指标）
      └── locomo_grades.json     # 汇总统计
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

# 导入三层 QA 系统
from memory.qa.three_layer_qa import ThreeLayerQA
from call_llm.call_llm_chat import call_llm as _call_llm_unified


# ─── 路径配置 ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GROUPS_DIR = os.path.join(DATA_DIR, 'groups')
LOCOMO_JSON = os.path.join(DATA_DIR, 'locomo10.json')

# 类别名称映射（与 memory_evaluation 保持一致）
CATEGORY_NAMES = {
    1: "Multi hop",
    2: "Temporal reasoning",
    3: "Open domain",
    4: "Single hop",
    5: "Adversarial",
}


# ─── NLP 指标计算（与 memory_evaluation/locomo_eval.py 一致） ───────────────

def tokenize(text: str) -> List[str]:
    """使用 NLTK 风格的简单分词"""
    text = text.lower().strip()
    # 去除标点符号，保留字母数字和空格
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def calculate_f1_score(gold_tokens: List[str], response_tokens: List[str]) -> float:
    """计算 F1 分数（与 memory_evaluation 一致）"""
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)

        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0

        precision = len(gold_set.intersection(response_set)) / len(response_set)
        recall = len(gold_set.intersection(response_set)) / len(gold_set)

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


def calculate_bleu1_score(gold_tokens: List[str], response_tokens: List[str]) -> float:
    """计算 BLEU-1 分数（简化版，与 memory_evaluation 一致）"""
    try:
        if not gold_tokens or not response_tokens:
            return 0.0

        # BLEU-1: unigram precision
        from collections import Counter
        response_counts = Counter(response_tokens)
        gold_counts = Counter(gold_tokens)

        # Clipped counts
        clipped = 0
        for word in response_counts:
            clipped += min(response_counts[word], gold_counts.get(word, 0))

        precision = clipped / len(response_tokens) if response_tokens else 0.0
        return precision
    except Exception as e:
        print(f"Failed to calculate BLEU-1 score: {e}")
        return 0.0


def calculate_nlp_metrics(gold_answer: str, generated_answer: str) -> Dict[str, float]:
    """计算 NLP 指标（F1 + BLEU-1）"""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    generated_answer = str(generated_answer) if generated_answer is not None else ""

    gold_tokens = tokenize(gold_answer)
    response_tokens = tokenize(generated_answer)

    return {
        "f1": calculate_f1_score(gold_tokens, response_tokens),
        "bleu1": calculate_bleu1_score(gold_tokens, response_tokens),
    }


# ─── LLM-as-a-Judge 评分（与 memory_evaluation 一致） ────────────────────────

def extract_label_json(text: str) -> Optional[str]:
    """
    从 LLM 响应中提取 {"label": "VALUE"} 格式的 JSON
    （与 memory_evaluation 完全一致）
    """
    if not text or not isinstance(text, str):
        return None

    patterns = [
        r'\{\s*"label"\s*:\s*"([^"]*)"\s*\}',
        r"\{\s*['\"]label['\"]\s*:\s*['\"]([^'\']*)['\"]?\s*\}",
        r'label["\']?\s*:\s*["\']?(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            label_value = match.group(1)
            return f'{{"label": "{label_value}"}}'

    return None


def llm_grader(question: str, gold_answer: str, generated_answer: str) -> bool:
    """
    使用 LLM 对答案进行评分（CORRECT/WRONG）
    与 memory_evaluation/locomo_eval.py 的 llm_grader 保持一致
    """
    question = str(question) if question is not None else ""
    gold_answer = str(gold_answer) if gold_answer is not None else ""
    generated_answer = str(generated_answer) if generated_answer is not None else ""

    system_prompt = "You are an expert grader that determines if answers to questions match a gold standard answer"

    accuracy_prompt = f"""
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

    try:
        response = _call_llm_unified(accuracy_prompt, temperature=0, max_tokens=256)

        # 提取 JSON 格式的 label
        json_str = extract_label_json(text=response)
        if json_str is None:
            print(f"  ⚠️ Failed to extract JSON from LLM response. Raw: {response[:200]}...")
            # 降级处理
            message_lower = response.lower()
            if "correct" in message_lower and "wrong" not in message_lower:
                print("  → Fallback: Detected 'CORRECT'")
                return True
            elif "wrong" in message_lower:
                print("  → Fallback: Detected 'WRONG'")
                return False
            else:
                print("  → Fallback failed: Cannot determine label")
                return False

        label = json.loads(json_str)["label"]
        result = label.strip().lower() == "correct"
        print(f"  ✅ LLM Judge: {'CORRECT' if result else 'WRONG'}")
        return result

    except Exception as e:
        print(f"  ❌ Error in llm_grader: {e}")
        return False


# ─── LocoMo 数据加载 ────────────────────────────────────────────────────────

def load_locomo_data(group_name: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    加载 LocoMo 数据并解析对话索引

    Args:
        group_name: 组名，如 group_0_caroline

    Returns:
        (qa_list, dia_index): QA 列表和对话索引字典
    """
    if not os.path.exists(LOCOMO_JSON):
        raise FileNotFoundError(f"LocoMo 文件不存在: {LOCOMO_JSON}")

    with open(LOCOMO_JSON, encoding='utf-8') as f:
        data = json.load(f)

    # 根据 group 名称确定使用哪个数据组
    group_idx = 0 if group_name.startswith('group_0') else 1
    g = data[group_idx]
    conv = g.get('conversation', {})

    # 构建 dia_id -> text 快速索引
    dia_index: Dict[str, Dict] = {}
    for key, val in conv.items():
        if not isinstance(val, list):
            continue
        session_num = key.replace('session_', '')
        date_key = f'session_{session_num}_date_time'
        date_time = conv.get(date_key, '')

        for turn in val:
            did = turn.get('dia_id', '')
            if did:
                dia_index[did] = {
                    'dia_id': did,
                    'speaker': turn.get('speaker', ''),
                    'text': turn.get('text', ''),
                    'session': f'Session {session_num}',
                    'date_time': date_time,
                }

    qa_list = g.get('qa', [])
    return qa_list, dia_index


# ─── Group 路径解析 ─────────────────────────────────────────────────────────

def resolve_group_paths(group_name: str) -> Dict[str, str]:
    """
    根据 group 名称解析相关路径

    Returns:
        包含各路径的字典
    """
    group_dir = os.path.join(GROUPS_DIR, group_name)
    eval_dir = os.path.join(group_dir, 'eval')

    effective_file = os.path.join(group_dir, 'effective', 'effective_memories.jsonl')
    session_file = os.path.join(group_dir, 'session', 'session_memories.txt')

    responses_file = os.path.join(eval_dir, 'locomo_responses.json')
    judged_file = os.path.join(eval_dir, 'locomo_judged.json')
    grades_file = os.path.join(eval_dir, 'locomo_grades.json')

    return {
        'group_dir': group_dir,
        'eval_dir': eval_dir,
        'effective_file': effective_file,
        'session_file': session_file,
        'responses_file': responses_file,
        'judged_file': judged_file,
        'grades_file': grades_file,
    }


# ─── 单题处理：生成回答 ─────────────────────────────────────────────────────

def process_single_question(
    qa_item: Dict,
    three_layer_qa: ThreeLayerQA,
    top_k: int,
    session_map: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    处理单道题：三层检索 + LLM 回答
    输出格式与 memory_evaluation/locomo_response_effective.py 一致（但使用三层记忆）
    """
    question = qa_item.get('question', '')
    category = qa_item.get('category', 0)

    # 对抗性问题特殊处理（与 memory_evaluation 一致）
    if category == 5:
        adversarial_answer = qa_item.get('adversarial_answer', '')
        import random
        if random.random() < 0.5:
            question_enhanced = f"{question} Select the correct answer: (a) Not mentioned in the conversation (b) {adversarial_answer}. "
        else:
            question_enhanced = f"{question} Select the correct answer: (a) {adversarial_answer} (b) Not mentioned in the conversation. "
    else:
        question_enhanced = question

    # ⭐ 使用三层 QA 系统回答问题
    result_qa = three_layer_qa.answer(
        question_enhanced,
        longterm_k=top_k,
        effective_k=top_k,
        session_k=top_k * 2,
        verbose=False
    )

    generated_answer = result_qa.answer

    # 构建三层证据链（用于保存到 responses）
    speaker_a_memories = []

    # Layer 1: Long-term Memory
    for item in result_qa.evidence.get('longterm', []):
        parts = [f"[Long-term] Topic: {item.content}"]
        if item.details.get('summary'):
            parts.append(f"Summary: {item.details['summary'][:100]}")
        if item.timestamp:
            parts.append(f"Time: {item.timestamp}")
        speaker_a_memories.append(" | ".join(parts))

    # Layer 2: Effective Memory
    for item in result_qa.evidence.get('effective', []):
        parts = [f"[Effective] Topic: {item.content}"]
        if item.details.get('summary'):
            parts.append(f"Summary: {item.details['summary'][:100]}")
        if item.timestamp:
            parts.append(f"Time: {item.timestamp}")
        speaker_a_memories.append(" | ".join(parts))

    # Layer 3: Session Memory
    for item in result_qa.evidence.get('session', [])[:5]:  # 限制数量
        parts = [f"[Session]"]
        if item.details.get('speaker'):
            parts.append(f"Speaker: {item.details['speaker']}")
        if item.timestamp:
            parts.append(f"Time: {item.timestamp}")
        text = item.details.get('full_text', item.content)[:150]
        parts.append(f"Text: {text}...")
        speaker_a_memories.append(" | ".join(parts))

    # 构建结果（与 memory_evaluation 格式一致）
    golden_answer = '' if category == 5 else str(qa_item.get('answer', ''))

    result = {
        "question": question,
        "category": category,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "tokens": 0,  # 暂不统计 token 用量
        "speaker_a_memories": speaker_a_memories,  # 三层证据链
        "speaker_b_memories": [],  # 当前只处理单用户记忆
        "confidence": result_qa.confidence,  # 新增：置信度
        "retrieval_stats": result_qa.retrieval_stats,  # 新增：检索统计
    }

    return result


# ─── 单题评分 ────────────────────────────────────────────────────────────────

def judge_single_question(
    response: Dict,
    llm_attempts: int = 1,
) -> Dict[str, Any]:
    """
    对单道题进行评分
    输出格式与 memory_evaluation/locomo_eval.py 的 process_single_question 一致
    """
    question = response.get("question")
    category = response.get("category")
    generated_answer = response.get("generated_answer")
    golden_answer = response.get("golden_answer")

    judgments_dict = {}
    nlp_metrics = {}

    if category == 5:
        # 对抗性问题：检查是否包含 "not mentioned"
        nlp_metrics = {"f1": 0.0, "bleu1": 0.0}
        key = "judgment_1"
        judgments_dict[key] = 'not mentioned' in generated_answer.lower()
    else:
        # LLM-as-a-judge 评分
        for i in range(llm_attempts):
            try:
                judgment = llm_grader(question, golden_answer, generated_answer)
                key = f"judgment_{i + 1}"
                judgments_dict[key] = judgment
            except Exception as e:
                print(f"Error in llm_grader for question: {question[:50]}, error: {e}")
                judgments_dict[f"judgment_{i + 1}"] = False

        # 计算 F1, BLEU-1 分数
        try:
            nlp_metrics = calculate_nlp_metrics(golden_answer, generated_answer)
        except Exception as e:
            print(f"Error calculating NLP metrics for question: {question[:50]}, error: {e}")
            nlp_metrics = {"f1": 0.0, "bleu1": 0.0}

    graded_response = {
        "question": question,
        "category": category,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "llm_judgments": judgments_dict,
        "nlp_metrics": nlp_metrics,
    }
    return graded_response


# ─── 统计计算（与 memory_evaluation/locomo_metric.py 一致） ─────────────────

def calculate_category_scores(data: Dict[str, List[Dict]]) -> Dict[int, Dict]:
    """按类别统计各项评估指标"""
    category_data = {}

    for group_key, questions in data.items():
        for question in questions:
            category_id = question["category"]

            if category_id not in category_data:
                category_data[category_id] = {
                    "llm_judgments": [],
                    "lexical": {"f1": [], "bleu1": []},
                    "total_questions": 0,
                }

            category_data[category_id]["total_questions"] += 1

            if "llm_judgments" in question:
                for judgment_key, judgment_value in question["llm_judgments"].items():
                    score = 1 if judgment_value else 0
                    category_data[category_id]["llm_judgments"].append(score)

            nlp_metrics = question.get("nlp_metrics", {})
            for metric in ["f1", "bleu1"]:
                value = nlp_metrics.get(metric)
                if value is not None:
                    category_data[category_id]["lexical"][metric].append(value)

    category_scores = {}
    for category_id, data in category_data.items():
        category_name = CATEGORY_NAMES.get(category_id, "Unknown")

        category_scores[category_id] = {
            "category_name": category_name,
            "total_questions": data["total_questions"],
            "llm_judge_score": 0.0,
            "lexical": {},
        }

        if data["llm_judgments"]:
            category_scores[category_id]["llm_judge_score"] = sum(data["llm_judgments"]) / len(data["llm_judgments"])

        for metric in ["f1", "bleu1"]:
            values = data["lexical"][metric]
            category_scores[category_id]["lexical"][metric] = sum(values) / len(values) if values else 0.0

    return category_scores


def calculate_overall_scores(data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    计算所有问题的总体统计指标
    需要剔除对抗性问题（category=5）的影响（与 memory_evaluation 一致）
    """
    overall_data = {
        "llm_judgments": [],
        "lexical": {"f1": [], "bleu1": []},
        "total_questions": 0,
    }

    for group_key, questions in data.items():
        for question in questions:
            if question["category"] == 5:
                continue

            overall_data["total_questions"] += 1

            if "llm_judgments" in question:
                for judgment_key, judgment_value in question["llm_judgments"].items():
                    score = 1 if judgment_value else 0
                    overall_data["llm_judgments"].append(score)

            nlp_metrics = question.get("nlp_metrics", {})
            for metric in ["f1", "bleu1"]:
                value = nlp_metrics.get(metric)
                if value is not None:
                    overall_data["lexical"][metric].append(value)

    overall_scores = {
        "total_questions": overall_data["total_questions"],
        "llm_judge_score": 0.0,
        "lexical": {},
    }

    if overall_data["llm_judgments"]:
        overall_scores["llm_judge_score"] = sum(overall_data["llm_judgments"]) / len(overall_data["llm_judgments"])

    for metric in ["f1", "bleu1"]:
        values = overall_data["lexical"][metric]
        overall_scores["lexical"][metric] = sum(values) / len(values) if values else 0.0

    return overall_scores


# ─── 主评测流程 ──────────────────────────────────────────────────────────────

def run_eval(
    group_name: str,
    top_k: int = 5,
    llm_grader_attempts: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    执行完整的 Eval 评测流程

    流程（与 memory_evaluation 一致）：
      1. Response 阶段：检索 + LLM 回答 → locomo_responses.json
      2. Judged 阶段：LLM Judge + NLP 指标 → locomo_judged.json
      3. Grades 阶段：汇总统计 → locomo_grades.json
    """
    print("\n" + "=" * 80)
    print("🚀 LocoMo QA Eval 评测")
    print("=" * 80)
    print(f"   Group: {group_name}")
    print(f"   Top-K: {top_k}")
    print(f"   LLM Grader Attempts: {llm_grader_attempts}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 解析路径
    paths = resolve_group_paths(group_name)
    os.makedirs(paths['eval_dir'], exist_ok=True)

    print(f"\n📂 路径配置:")
    print(f"   Effective: {paths['effective_file']}")
    print(f"   Responses: {paths['responses_file']}")
    print(f"   Judged:   {paths['judged_file']}")
    print(f"   Grades:   {paths['grades_file']}")

    # 2. 加载数据
    print(f"\n📦 加载数据...")

    # ⭐ 初始化三层 QA 系统
    three_layer_qa = ThreeLayerQA(group=group_name)
    print(f"   ✅ 三层 QA 系统已初始化 (User: {three_layer_qa.user_id})")

    # 加载 Session Memories（用于证据链）
    session_map = {}
    session_file_jsonl = os.path.join(
        os.path.dirname(paths['effective_file']),
        'session',
        'memories.jsonl'
    )
    if os.path.exists(session_file_jsonl):
        with open(session_file_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sm = json.loads(line.strip())
                    session_map[str(sm.get('memory_id', ''))] = sm
                except Exception:
                    pass
        print(f"   ✅ Session Memories: {len(session_map)} 条")

    # 加载 LocoMo QA 数据
    qa_list, dia_index = load_locomo_data(group_name)
    print(f"   ✅ LocoMo Questions: {len(qa_list)} 道")

    # 生成 group key（与 memory_evaluation 一致）
    group_idx = 0 if group_name.startswith('group_0') else 1
    group_key = f"locomo_group_{group_idx + 1}"

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Response — 检索 + LLM 回答
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"📝 PHASE 1: 生成回答 ({len(qa_list)} 道题)")
    print(f"{'='*80}")

    # 尝试加载已有的 responses（支持断点续跑）
    responses_data = {}
    if os.path.exists(paths['responses_file']):
        try:
            with open(paths['responses_file'], 'r', encoding='utf-8') as f:
                responses_data = json.load(f)
            if group_key in responses_data:
                print(f"   📋 已找到已有 responses: {len(responses_data[group_key])} 题")
        except Exception:
            responses_data = {}

    if group_key not in responses_data:
        responses_data[group_key] = []

    has_processed_cnt = len(responses_data[group_key])

    for idx, qa_item in enumerate(qa_list):
        # 跳过已处理的题目
        if idx < has_processed_cnt:
            continue

        print(f"\n   [{idx+1}/{len(qa_list)}] {qa_item.get('question', '')[:60]}...")

        result = process_single_question(
            qa_item=qa_item,
            three_layer_qa=three_layer_qa,
            top_k=top_k,
            session_map=session_map,
        )
        responses_data[group_key].append(result)

        # 每完成一题就保存一次（支持断点续跑）
        with open(paths['responses_file'], 'w', encoding='utf-8') as f:
            json.dump(responses_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 已回答并保存")

    print(f"\n💾 Responses 已保存: {paths['responses_file']}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: Judged — LLM Judge + NLP 指标
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"🔬 PHASE 2: 评分 ({len(responses_data[group_key])} 道题)")
    print(f"{'='*80}")

    # 尝试加载已有的 judged（支持断点续跑）
    judged_data = {}
    if os.path.exists(paths['judged_file']):
        try:
            with open(paths['judged_file'], 'r', encoding='utf-8') as f:
                judged_data = json.load(f)
            if group_key in judged_data:
                print(f"   📋 已找到已有 judged: {len(judged_data[group_key])} 题")
        except Exception:
            judged_data = {}

    if group_key not in judged_data:
        judged_data[group_key] = []

    has_judged_cnt = len(judged_data[group_key])
    responses_list = responses_data[group_key]

    for idx, response in enumerate(responses_list):
        # 跳过已评分的题目
        if idx < has_judged_cnt:
            continue

        print(f"\n   [{idx+1}/{len(responses_list)}] {response.get('question', '')[:60]}...")

        graded = judge_single_question(
            response=response,
            llm_attempts=llm_grader_attempts,
        )
        judged_data[group_key].append(graded)

        # 每完成一题就保存一次
        with open(paths['judged_file'], 'w', encoding='utf-8') as f:
            json.dump(judged_data, f, ensure_ascii=False, indent=2)

        scores = graded.get('nlp_metrics', {})
        judgment = graded.get('llm_judgments', {})
        j1 = judgment.get('judgment_1', 'N/A')
        print(f"   📊 F1={scores.get('f1', 0):.4f}, BLEU1={scores.get('bleu1', 0):.4f}, Judge={'✅' if j1 else '❌'}")

    print(f"\n💾 Judged 已保存: {paths['judged_file']}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: Grades — 汇总统计
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"📊 PHASE 3: 汇总统计")
    print(f"{'='*80}")

    # 计算整体得分
    overall_scores = calculate_overall_scores(judged_data)

    # 计算类别得分
    category_scores = calculate_category_scores(judged_data)

    grades_result = {
        "overall_scores": overall_scores,
        "category_scores": category_scores,
    }

    # 保存 grades
    with open(paths['grades_file'], 'w', encoding='utf-8') as f:
        json.dump(grades_result, f, ensure_ascii=False, indent=4)

    # 打印最终报告
    print(f"\n{'='*80}")
    print(f"📊 评测完成！总体统计")
    print(f"{'='*80}")
    print(f"   总题数（不含 Adversarial）: {overall_scores['total_questions']}")
    print(f"   LLM Judge Score:           {overall_scores['llm_judge_score']:.4f}")
    print(f"   F1 Score:                  {overall_scores['lexical']['f1']:.4f}")
    print(f"   BLEU-1 Score:              {overall_scores['lexical']['bleu1']:.4f}")
    print(f"\n   📁 分类统计:")
    for cat_id, stats in sorted(category_scores.items()):
        name = stats['category_name']
        count = stats['total_questions']
        lj = stats['llm_judge_score']
        f1 = stats['lexical']['f1']
        b1 = stats['lexical']['bleu1']
        print(f"      [{cat_id}] {name:20s}: {count:2d} 题 | Judge={lj:.4f}, F1={f1:.4f}, B1={b1:.4f}")

    print(f"\n💾 Grades 已保存: {paths['grades_file']}")
    print(f"{'='*80}")

    return {
        'metadata': {
            'group_name': group_name,
            'group_key': group_key,
            'eval_time': datetime.now().isoformat(),
            'top_k': top_k,
            'num_questions': len(qa_list),
            'num_effective_memories': len(memories),
            'llm_grader_attempts': llm_grader_attempts,
        },
        **grades_result,
        'responses_file': paths['responses_file'],
        'judged_file': paths['judged_file'],
        'grades_file': paths['grades_file'],
    }


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LocoMo QA Eval 评测工具（对齐 memory_evaluation 输出格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python src/eval/run_eval.py --group group_0_caroline

  # 自定义检索数量
  python src/eval/run_eval.py --group group_1_gina --top_k 10

  # 多次 LLM 评估取平均（更稳定但更慢）
  python src/eval/run_eval.py --group group_0_melanie --judge-attempts 3

  # 静默模式
  python src/eval/run_eval.py --group group_0_caroline --quiet

输出文件（data/groups/{group}/eval/）:
  ├── locomo_responses.json   # 原始回答（含检索到的有效记忆）
  ├── locomo_judged.json     # 评分结果（LLM Judge + F1/BLEU-1）
  └── locomo_grades.json     # 汇总统计（总分 + 分类得分）

注意:
  - 此脚本会自动断点续跑，可随时中断后重新执行
  - Adversarial 问题（category=5）不参与 F1/BLEU 计算
  - LLM Judge 使用 call_llm 接口，需配置 .env 中的 API Key
        """,
    )
    parser.add_argument(
        '--group',
        type=str,
        required=True,
        help='Group 名称 (如 group_0_caroline, group_1_gina)',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='每题检索的相关记忆数量 (默认: 5)',
    )
    parser.add_argument(
        '--judge-attempts',
        type=int,
        default=1,
        help='LLM Judge 评估次数，多次取平均更稳定 (默认: 1)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，减少输出',
    )

    args = parser.parse_args()

    try:
        result = run_eval(
            group_name=args.group,
            top_k=args.top_k,
            llm_grader_attempts=args.judge_attempts,
            verbose=not args.quiet,
        )
        print("\n✅ 评测成功完成！")
        print(f"\n📁 输出文件:")
        print(f"   Responses: {result['responses_file']}")
        print(f"   Judged:   {result['judged_file']}")
        print(f"   Grades:   {result['grades_file']}")
        return 0
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

