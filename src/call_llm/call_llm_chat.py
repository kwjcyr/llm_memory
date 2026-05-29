import base64

import numpy as np
import requests

friday_app_id = "2004456985261174860"
model = "LongCat-Flash-Chat-Eco"

def get_chat(query: str) -> str:
    """获取文本的 embedding 向量"""
    url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"

    headers = {'Authorization': f'Bearer {friday_app_id}'}

    try:
        request_content = {'messages': [{'role': 'user', 'content': query}], 'model': model, 'max_tokens': 2048, 'stream': False}
        res = requests.post(url, json=request_content, stream=True, headers=headers)
        for chunk in res.iter_content(chunk_size=10240):
            r = chunk.decode('utf-8')
            print(r)
    except Exception as e:
        print(f"❌ JSON 解析失败: {e}, response: {res.text[:200]}")
        return r

if __name__ == '__main__':
    print(get_chat("今天天气怎样"))