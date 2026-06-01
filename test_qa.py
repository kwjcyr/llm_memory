"""
测试问答系统回答 locomo10.json 中的问题
"""
from src.memory.effective_memory.qa_from_effective_memory import answer_question

# 测试 locomo10.json 中的问题
questions = [
    ('When did Caroline go to the LGBTQ support group?', '7 May 2023'),
    ('What fields would Caroline be likely to pursue in her education?', 'Psychology, counseling certification'),
    ('What is Caroline identity?', 'Transgender woman'),
]

print("=" * 80)
print("测试问答系统")
print("=" * 80)

for question, expected_answer in questions:
    print(f"\n问题：{question}")
    print(f"标准答案：{expected_answer}")
    print("-" * 80)

    answer = answer_question(question)
    print(f"系统答案：{answer}")
    print()

