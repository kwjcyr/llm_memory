"""
Memory Explorer — FastAPI 后端

关联模型（统一 memoryId 列表）：
  Session  memory_id  ←── effective.source_memory_ids
  Effective time_range key ←── longterm.source_effective_ids

GET /api/sessions          → Session Memory 列表（来自 caroline_memories.txt）
GET /api/effectives        → Effective Memory 列表（来自 effective_memories.txt）
GET /api/longtermemories   → Long-term Memory 列表（来自 SQLite）
GET /api/links             → 三层关联关系图

PATCH /api/longtermemories/{id}   → 更新长期记忆
DELETE /api/longtermemories/{id}  → 删除长期记忆
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.sqlite.longterm_memory_db import LongTermMemoryDatabase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAROLINE_TXT    = os.path.join(BASE_DIR, 'data', 'caroline_memories.txt')
EFFECTIVE_TXT   = os.path.join(BASE_DIR, 'data', 'effective_memories.txt')
LONGTERM_DB     = os.path.join(BASE_DIR, 'data', 'longterm_memories.db')
LOCOMO_JSON     = os.path.join(BASE_DIR, 'data', 'input', 'locomo10.json')
HARNESS_DIR     = os.path.join(BASE_DIR, 'data', 'harness_output')
GROUPS_DIR       = os.path.join(BASE_DIR, 'data', 'groups')

# ─── User ID → 文件路径映射 ────────────────────────────────────────────────

USER_FILE_MAP = {
    # user_id -> (session_file, effective_file)
    'caroline': (
        os.path.join(GROUPS_DIR, 'group_0_caroline', 'session', 'memories.jsonl'),
        os.path.join(GROUPS_DIR, 'group_0_caroline', 'effective', 'effective_memories.jsonl'),
    ),
    'gina': (
        os.path.join(GROUPS_DIR, 'group_1_gina', 'session', 'memories.jsonl'),
        os.path.join(GROUPS_DIR, 'group_1_gina', 'effective', 'effective_memories.jsonl'),
    ),
    'melanie': (
        os.path.join(GROUPS_DIR, 'group_0_melanie', 'session', 'memories.jsonl'),
        os.path.join(GROUPS_DIR, 'group_0_melanie', 'effective', 'effective_memories.jsonl'),
    ),
    'jon': (
        os.path.join(GROUPS_DIR, 'group_1_jon', 'session', 'memories.jsonl'),
        os.path.join(GROUPS_DIR, 'group_1_jon', 'effective', 'effective_memories.jsonl'),
    ),
}

def _resolve_user_files(user_id: str) -> tuple:
    """根据 user_id 返回 (session_file, effective_file)，找不到则回退到默认路径"""
    if user_id in USER_FILE_MAP:
        return USER_FILE_MAP[user_id]
    # 回退到旧逻辑
    return CAROLINE_TXT, EFFECTIVE_TXT

app = FastAPI(title="Memory Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 给 effective memory 补一个稳定的 id（用行号）
                if 'effective_id' not in obj:
                    obj['effective_id'] = f"eff_{i}"
                rows.append(obj)
            except Exception:
                pass
    return rows


# ─── models ───────────────────────────────────────────────────────────────────

class UpdateMemoryBody(BaseModel):
    topic: Optional[str] = None
    summary: Optional[str] = None
    facts: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    confidence: Optional[float] = None


# ─── routes ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def get_sessions(user_id: str = "", limit: int = 500, order: str = "asc"):
    """Session Memory 列表（按 user_id 加载对应 group 的数据）"""
    session_file, _ = _resolve_user_files(user_id)
    rows = load_jsonl(session_file)
    if user_id:
        rows = [r for r in rows if user_id.lower() in r.get('user_id', '').lower()]
    rows.sort(key=lambda r: r.get('timestamp', ''), reverse=(order == 'desc'))
    return {"data": rows[:limit], "total": len(rows)}


@app.get("/api/effectives")
def get_effectives(user_id: str = "", order: str = "asc"):
    """Effective Memory 列表（按 user_id 加载对应 group 的数据）"""
    _, effective_file = _resolve_user_files(user_id)
    print(f"[DEBUG] get_effectives: user_id={user_id}, effective_file={effective_file}")
    rows = load_jsonl(effective_file)
    for r in rows:
        if 'original_text' in r:
            r['original_text_preview'] = r['original_text'][:300] + '...'
            del r['original_text']
    rows.sort(
        key=lambda r: r.get('time_range', {}).get('start', ''),
        reverse=(order == 'desc')
    )
    return {"data": rows, "total": len(rows)}


@app.get("/api/longtermemories")
def get_longtermemories(user_id: str = "caroline", order: str = "asc"):
    """Long-term Memory 列表（来自 SQLite），按 time_range_start 排序"""
    db = LongTermMemoryDatabase(LONGTERM_DB)
    # DB 默认 DESC，先取全部再在 Python 侧统一排序
    mems = db.get_memories_by_user(user_id)
    mems.sort(
        key=lambda m: m.time_range_start or m.created_at or '',
        reverse=(order == 'desc')
    )
    return {"data": [m.to_dict() for m in mems], "total": len(mems)}


@app.get("/api/links")
def get_links(user_id: str = "caroline"):
    """
    返回三层关联关系：
    {
      "session_to_effective": { "session_memory_id": ["eff_0", "eff_2", ...] },
      "effective_to_longterm": { "eff_0": ["lt_memory_id_1", ...] }
    }
    核心逻辑：
      - effective[i].source_memory_ids   → 反向映射：session_id → effective_id
      - longterm.source_effective_ids    → 正向映射：effective_id → longterm_id
        (source_effective_ids 里存的是 session memory_id 列表，
         通过比对 effective.source_memory_ids 做匹配)
    """
    _, effective_file = _resolve_user_files(user_id)
    session_file, _ = _resolve_user_files(user_id)
    effectives = load_jsonl(effective_file)
    sessions_map = {str(r['memory_id']): r for r in load_jsonl(session_file)}
    db = LongTermMemoryDatabase(LONGTERM_DB)
    longtermemories = db.get_memories_by_user(user_id)

    # Session → Effective（一条 session 可能对应多条 effective）
    session_to_effective: Dict[str, List[str]] = {}
    for eff in effectives:
        eff_id = eff['effective_id']
        for sid in eff.get('source_memory_ids', []):
            session_to_effective.setdefault(str(sid), []).append(eff_id)

    # Effective → Long-term
    # longterm.source_effective_ids 存的是 session memory_id 列表
    # 通过集合交集匹配 effective
    eff_session_sets = {
        eff['effective_id']: set(str(s) for s in eff.get('source_memory_ids', []))
        for eff in effectives
    }
    effective_to_longterm: Dict[str, List[str]] = {}
    for lt in longtermemories:
        lt_session_ids = set(str(s) for s in lt.source_effective_ids)
        for eff_id, eff_sessions in eff_session_sets.items():
            # 若有交集，认为 longterm 来自该 effective
            if lt_session_ids & eff_sessions:
                effective_to_longterm.setdefault(eff_id, []).append(lt.memory_id)

    return {
        "session_to_effective": session_to_effective,
        "effective_to_longterm": effective_to_longterm,
    }


@app.patch("/api/longtermemories/{memory_id}")
def update_longterm(memory_id: str, body: UpdateMemoryBody):
    """更新长期记忆"""
    db = LongTermMemoryDatabase(LONGTERM_DB)
    ok = db.update_memory(
        memory_id=memory_id,
        topic=body.topic,
        summary=body.summary,
        facts=body.facts,
        tags=body.tags,
        confidence=body.confidence,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}


@app.delete("/api/longtermemories/{memory_id}")
def delete_longterm(memory_id: str):
    """软删除长期记忆"""
    db = LongTermMemoryDatabase(LONGTERM_DB)
    ok = db.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}


# ─── eval routes ──────────────────────────────────────────────────────────────

@app.get("/api/eval/run")
def run_eval(question: str, user_id: str = "", top_k: int = 5):
    """
    对单条问题执行完整 QA 流程（三层记忆降级策略）：
    1. Long-term Memory（高度提炼的知识）
    2. Effective Memory（对话摘要）
    3. Session Memory 原文（精确时间信息）
    4. 调用 LLM 基于三层证据回答问题
    5. 返回 predicted answer + 三层证据链
    """
    from src.memory.qa.three_layer_qa import ThreeLayerQA

    # 推导 group 名称
    group = None
    if user_id:
        for gid, (sess_f, eff_f) in USER_FILE_MAP.items():
            if gid == user_id:
                # 反向查找 group
                for gname in ['group_0_caroline', 'group_1_gina', 'group_0_melanie', 'group_1_jon']:
                    if user_id in gname:
                        group = gname
                        break
                break

    # 初始化三层 QA 系统
    qa = ThreeLayerQA(group=group, user_id=user_id)

    # 执行三层检索和回答
    result = qa.answer(
        question,
        longterm_k=top_k,
        effective_k=top_k,
        session_k=top_k * 2,
        verbose=False
    )

    # 构建三层证据链返回给前端
    chain = {
        'longterm': [
            {
                'layer': 'longterm',
                'content': item.content,
                'topic': item.content,
                'summary': item.details.get('summary', ''),
                'facts': item.details.get('facts', []),
                'timestamp': item.timestamp,
                'source_id': item.source_id,
                'score': item.score,
            }
            for item in result.evidence.get('longterm', [])
        ],
        'effective': [
            {
                'layer': 'effective',
                'content': item.content,
                'topic': item.content,
                'summary': item.details.get('summary', ''),
                'facts': item.details.get('facts', []),
                'timestamp': item.timestamp,
                'source_id': item.source_id,
                'score': item.score,
            }
            for item in result.evidence.get('effective', [])
        ],
        'session': [
            {
                'layer': 'session',
                'content': item.content,
                'full_text': item.details.get('full_text', '')[:500],
                'timestamp': item.timestamp,
                'speaker': item.details.get('speaker', ''),
                'source_id': item.source_id,
                'score': item.score,
            }
            for item in result.evidence.get('session', [])
        ]
    }

    return {
        'question': question,
        'predicted': result.answer,
        'confidence': result.confidence,
        'three_layer_chain': chain,
        'retrieval_stats': result.retrieval_stats,
    }


def _load_locomo_group0() -> Dict[str, Any]:
    """加载 locomo10.json 第 0 组（Caroline/Melanie）并解析对话索引"""
    if not os.path.exists(LOCOMO_JSON):
        return {}
    with open(LOCOMO_JSON, encoding='utf-8') as f:
        data = json.load(f)
    g = data[0]
    conv = g.get('conversation', {})

    # 构建 dia_id -> text 快速索引，例如 "D1:3" -> {speaker, text, session, date_time}
    dia_index: Dict[str, Dict] = {}
    for key, val in conv.items():
        if not isinstance(val, list):
            continue
        # key: session_1, session_2, ...
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
    return {'qa': g.get('qa', []), 'dia_index': dia_index,
            'speaker_a': conv.get('speaker_a', ''), 'speaker_b': conv.get('speaker_b', '')}


CATEGORY_LABELS = {1: 'Single-hop', 2: 'Temporal', 3: 'Multi-hop', 4: 'Adversarial', 5: 'Open-ended'}


@app.get("/api/eval/questions")
def get_eval_questions(speaker: str = "caroline"):
    """返回 LocoMo QA 列表，含 evidence 的实际对话文本"""
    locomo = _load_locomo_group0()
    if not locomo:
        raise HTTPException(status_code=404, detail="locomo10.json not found")
    qa_list = locomo['qa']
    dia_index = locomo['dia_index']

    # 可按 speaker 过滤：evidence 里的对话含该 speaker 姓名
    speaker_filter = speaker.strip().lower()

    results = []
    for i, qa in enumerate(qa_list):
        question: str = qa.get('question', '')
        # 仅保留提到目标 speaker 的问题（或不过滤）
        if speaker_filter and speaker_filter not in question.lower():
            continue
        evidences = []
        for dia_id in qa.get('evidence', []):
            turn = dia_index.get(dia_id)
            if turn:
                evidences.append(turn)
            else:
                evidences.append({'dia_id': dia_id, 'speaker': '?', 'text': '(not found)', 'session': '', 'date_time': ''})
        results.append({
            'idx': i,
            'question': question,
            'ground_truth': str(qa.get('answer', '')),
            'category': qa.get('category', 0),
            'category_label': CATEGORY_LABELS.get(qa.get('category', 0), 'Unknown'),
            'evidence_ids': qa.get('evidence', []),
            'evidence_texts': evidences,
        })
    return {'data': results, 'total': len(results),
            'speaker_a': locomo['speaker_a'], 'speaker_b': locomo['speaker_b']}


@app.get("/api/eval/report")
def get_eval_report():
    """返回最新的 comparison_report（harness_output 目录里最新的）"""
    if not os.path.exists(HARNESS_DIR):
        return {'reports': []}
    reports = []
    for fname in sorted(os.listdir(HARNESS_DIR)):
        if not fname.startswith('comparison_report') or not fname.endswith('.json'):
            continue
        fpath = os.path.join(HARNESS_DIR, fname)
        try:
            with open(fpath, encoding='utf-8') as f:
                report = json.load(f)
            report['_filename'] = fname
            reports.append(report)
        except Exception:
            pass
    return {'reports': reports}


@app.get("/api/eval/answer")
def get_eval_answer(question: str, user_id: str = "", top_k: int = 5):
    """
    对单条问题用 effective_memories 做检索+评分，返回证据链。
    证据链包括：检索到的 effective memory 列表 + 每条 memory 的 source_memory_ids
    对应的 session 原文。
    """
    _, effective_file = _resolve_user_files(user_id)
    session_file, _ = _resolve_user_files(user_id)
    if not os.path.exists(effective_file):
        raise HTTPException(status_code=404, detail=f"effective file not found: {effective_file}")

    effectives = load_jsonl(effective_file)
    sessions_map = {str(r['memory_id']): r for r in load_jsonl(session_file)}

    q_lower = question.lower()
    q_words = set(q_lower.split())

    scored = []
    for eff in effectives:
        score = 0
        if any(w in eff.get('topic', '').lower() for w in q_words):
            score += 3
        if any(w in eff.get('summary', '').lower() for w in q_words):
            score += 2
        facts_text = ' '.join(eff.get('facts', []) or []).lower()
        if any(w in facts_text for w in q_words):
            score += 2
        tags_text = ' '.join(eff.get('tags', []) or []).lower()
        if any(w in tags_text for w in q_words):
            score += 1
        if score > 0:
            scored.append((score, eff))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_effs = [e for _, e in scored[:top_k]]

    # 构建证据链：每条 effective → 关联的 session 原文
    chain = []
    for eff in top_effs:
        src_ids = [str(s) for s in (eff.get('source_memory_ids') or [])]
        src_sessions = [sessions_map[sid] for sid in src_ids if sid in sessions_map]
        chain.append({
            'effective_id': eff.get('effective_id', ''),
            'topic': eff.get('topic', ''),
            'summary': eff.get('summary', ''),
            'facts': eff.get('facts', []),
            'time_range': eff.get('time_range', {}),
            'retrieval_score': scored[top_effs.index(eff)][0] if eff in top_effs else 0,
            'source_sessions': src_sessions,
        })

    return {'question': question, 'retrieved_chain': chain, 'total_effective': len(effectives)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

