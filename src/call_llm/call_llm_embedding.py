import base64

import numpy as np
import requests

friday_app_id = "2004456985261174860"

def get_text_embedding(model: str, text: str) -> np.ndarray:
    """获取文本的 embedding 向量"""
    url = "https://aigc.sankuai.com/v1/openai/native/embeddings"

    payload = {
        "model": model,
        "input": text,
        "encoding_format": "float"
    }

    headers = {
        'Authorization': f'Bearer {friday_app_id}',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=payload)

    try:
        result = response.json()
        return result['data'][0]['embedding']
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}, response: {response.text[:200]}")
        return None

if __name__ == '__main__':
    print("qwen3_0dot6b_fine:", len(get_text_embedding("qwen3_0dot6b_fine", "测试")))
    print("qwen3_fine_custom:", len(get_text_embedding("qwen3_fine_custom", "测试")))