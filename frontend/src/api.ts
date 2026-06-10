import type {
  SessionMemory, EffectiveMemory, LongTermMemory, Links,
  EvalQuestion, EvalAnswerResult, ComparisonReport,
} from './types';

const BASE = 'http://localhost:8000/api';

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json() as Promise<T>;
}

export async function fetchSessions(userId = '', order: 'asc' | 'desc' = 'asc'): Promise<SessionMemory[]> {
  const p = new URLSearchParams({ limit: '500', order });
  if (userId) p.set('user_id', userId);
  const res = await get<{ data: SessionMemory[] }>(`/sessions?${p}`);
  return res.data;
}

export async function fetchEffectives(order: 'asc' | 'desc' = 'asc'): Promise<EffectiveMemory[]> {
  const res = await get<{ data: EffectiveMemory[] }>(`/effectives?order=${order}`);
  return res.data;
}

export async function fetchLongtermemories(userId = 'caroline', order: 'asc' | 'desc' = 'asc'): Promise<LongTermMemory[]> {
  const res = await get<{ data: LongTermMemory[] }>(`/longtermemories?user_id=${userId}&order=${order}`);
  return res.data;
}

export async function fetchLinks(userId = 'caroline'): Promise<Links> {
  return get<Links>(`/links?user_id=${userId}`);
}

export async function patchLongterm(
  memoryId: string,
  body: Partial<Pick<LongTermMemory, 'topic' | 'summary' | 'facts' | 'tags' | 'confidence'>>,
): Promise<void> {
  const r = await fetch(`${BASE}/longtermemories/${memoryId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
}

export async function deleteLongterm(memoryId: string): Promise<void> {
  const r = await fetch(`${BASE}/longtermemories/${memoryId}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
}

// ─── Eval API ─────────────────────────────────────────────────────────────────

export async function fetchEvalQuestions(speaker = 'caroline'): Promise<{
  data: EvalQuestion[];
  total: number;
  speaker_a: string;
  speaker_b: string;
}> {
  return get(`/eval/questions?speaker=${speaker}`);
}

export async function fetchEvalAnswer(question: string, topK = 5): Promise<EvalAnswerResult> {
  const p = new URLSearchParams({ question, top_k: String(topK) });
  return get(`/eval/answer?${p}`);
}

export async function fetchEvalReport(): Promise<{ reports: ComparisonReport[] }> {
  return get('/eval/report');
}

