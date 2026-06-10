"""
统一 LLM 调用模块

所有 LLM 调用（extract / qa / longterm_ops / pipeline）都应通过此模块，
配置从 .env 文件读取，不再硬编码 app_id / model。
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# ─── 加载 .env ──────────────────────────────────────────────────────────────
# 从项目根目录加载 .env（支持多层调用时自动找到）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/call_llm → 项目根
_ENV_FILE = _PROJECT_ROOT / '.env'
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)
else:
    load_dotenv()  # fallback: 当前工作目录


def get_config() -> Dict[str, Any]:
    """返回当前 LLM 配置字典，所有字段均来自环境变量。"""
    return {
        'friday_app_id': os.getenv('FRIDAY_APP_ID', ''),
        'model':          os.getenv('LLM_MODEL', 'LongCat-Flash-Chat-Eco'),
        'temperature':   float(os.getenv('LLM_TEMPERATURE', '0.1')),
        'top_p':         float(os.getenv('LLM_TOP_P', '1')),
        'max_tokens':    int(os.getenv('LLM_MAX_TOKENS', '4096')),
        'base_url':      os.getenv('LLM_BASE_URL', 'https://aigc.sankuai.com/v1/openai/native/chat/completions'),
    }


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    统一的 LLM 调用接口。

    Args:
        prompt: 用户消息内容（完整 prompt）
        model: 模型名称，默认从 .env 读取
        temperature: 温度，默认从 .env 读取
        top_p: top_p，默认从 .env 读取
        max_tokens: 最大 token 数，默认从 .env 读取
        system_prompt: 可选的系统提示词

    Returns:
        LLM 返回的文本内容
    """
    cfg = get_config()

    headers = {'Authorization': f'Bearer {cfg["friday_app_id"]}'}

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})

    payload = {
        'model':       model or cfg['model'],
        'messages':    messages,
        'temperature': temperature if temperature is not None else cfg['temperature'],
        'top_p':       top_p if top_p is not None else cfg['top_p'],
        'max_tokens':  max_tokens if max_tokens is not None else cfg['max_tokens'],
        'stream':      False,
    }

    try:
        response = requests.post(cfg['base_url'], headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unexpected response format: {result}")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"LLM 调用超时 (>{120}s)")
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}")


# ─── 向后兼容的便捷函数 ───────────────────────────────────────────────────────
def get_chat(query: str) -> str:
    """向后兼容：简单的一轮对话。"""
    return call_llm(query)


if __name__ == '__main__':
    cfg = get_config()
    print(f"模型: {cfg['model']}")
    print(f"App ID: {cfg['friday_app_id'][:8]}...")
    r = get_chat("你好，请简短回复")
    print(f"回复: {r}")
