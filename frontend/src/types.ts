// ─── Session Memory ───────────────────────────────────────────────────────────
export interface SessionMemory {
  memory_id: string;
  user_id: string;
  timestamp: string;
  user_content: string;
  assistant_content: string;
}

// ─── Effective Memory ─────────────────────────────────────────────────────────
export interface EffectiveMemory {
  effective_id: string;           // synthetic id: "eff_0", "eff_1" …
  topic: string;
  summary: string;
  facts: string[];
  memory_type: 'LongTermMemory' | 'UserMemory';
  tags: string[];
  confidence: number;
  ttl_days: number | null;
  source_memory_ids: string[];    // → Session.memory_id 列表
  time_range: { start: string; end: string };
  original_text_preview?: string;
}

// ─── Long-term Memory ─────────────────────────────────────────────────────────
export interface LongTermMemory {
  memory_id: string;
  user_id: string;
  topic: string;
  summary: string;
  facts: string[];
  memory_type: string;
  tags: string[];
  confidence: number;
  ttl_days: number | null;
  source_effective_ids: string[]; // session memory_id 列表（用于反查 effective）
  time_range_start: string | null;
  time_range_end: string | null;
  created_at: string;
  updated_at: string;
  is_deleted: number;
}

// ─── Links ────────────────────────────────────────────────────────────────────
export interface Links {
  session_to_effective: Record<string, string[]>; // session_id → effective_id[]
  effective_to_longterm: Record<string, string[]>; // effective_id → lt_id[]
}

// ─── Eval ─────────────────────────────────────────────────────────────────────
export interface EvidenceTurn {
  dia_id: string;
  speaker: string;
  text: string;
  session: string;
  date_time: string;
}

export interface EvalQuestion {
  idx: number;
  question: string;
  ground_truth: string;
  category: number;
  category_label: string;
  evidence_ids: string[];
  evidence_texts: EvidenceTurn[];
}

export interface RetrievedEffective {
  effective_id: string;
  topic: string;
  summary: string;
  facts: string[];
  time_range: { start: string; end: string };
  retrieval_score: number;
  source_sessions: SessionMemory[];
}

export interface EvalAnswerResult {
  question: string;
  retrieved_chain: RetrievedEffective[];
  total_effective: number;
}

export interface EvalQAResult {
  question: string;
  ground_truth: string;
  predicted: string;
  score: number;
}

export interface ComparisonReport {
  before_score: number;
  after_score: number;
  improvement: number;
  improvement_rate: number;
  before_results: EvalQAResult[];
  after_results: EvalQAResult[];
  best_addition: string;
  _filename: string;
}

// ─── Selection state ─────────────────────────────────────────────────────────
export type SelectedLayer = 'session' | 'effective' | 'longterm' | null;

export interface SelectionState {
  layer: SelectedLayer;
  id: string | null;
  /** 所有高亮的 session memory_id */
  highlightedSessionIds: Set<string>;
  /** 所有高亮的 effective_id */
  highlightedEffectiveIds: Set<string>;
  /** 所有高亮的 longterm memory_id */
  highlightedLongtermIds: Set<string>;
}

