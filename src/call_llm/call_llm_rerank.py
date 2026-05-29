import base64

import numpy as np
import requests

friday_app_id = "2004456985261174860"

def get_rerank(query: str, documents: list) -> list:
    url = "https://aigc.sankuai.com/v1/rerank"

    payload = {
        "model": "Qwen3-Reranker-0dot6B",
        "query": query,
        "documents": documents
    }

    headers = {
        'Authorization': f'Bearer {friday_app_id}',
        'Content-Type': 'application/json',
        'M-TransferContext-INF-CELL': 'gray-release-friday-ptest'
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    # 支持两种返回格式
    if 'results' in result:
        ranked = [(r['document'], r['relevance_score']) for r in result['results']]
        return ranked
    elif 'data' in result:
        # 按分数排序
        data = sorted(result['data'], key=lambda x: x['relevance_score'], reverse=True)
        ranked = [(documents[d['index']], d['relevance_score']) for d in data]
        return ranked
    else:
        print(f"❌ Rerank 请求失败: {result}")
        return []

if __name__ == '__main__':
    query = "年假怎么算？"
    recall_docs = ["1. 计算年假时数。", "2. 年假计算公式。", "3. 假期哪里玩"]
    rerank_results = get_rerank(query, recall_docs)
    print(rerank_results)