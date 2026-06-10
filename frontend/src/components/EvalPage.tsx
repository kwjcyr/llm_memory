import { useEffect, useState, useCallback } from 'react';
import type { EvalQuestion, EvalAnswerResult, ComparisonReport } from '../types';
import { fetchEvalQuestions, fetchEvalAnswer, fetchEvalReport } from '../api';
import './EvalPage.css';

// ─── Category badge ────────────────────────────────────────────────────────────
const CATEGORY_COLORS: Record<number, string> = {
  1: '#3b82f6',   // Single-hop: blue
  2: '#8b5cf6',   // Temporal: purple
  3: '#f59e0b',   // Multi-hop: amber
  4: '#ef4444',   // Adversarial: red
  5: '#10b981',   // Open-ended: green
};

function CategoryBadge({ category, label }: { category: number; label: string }) {
  const color = CATEGORY_COLORS[category] ?? '#64748b';
  return (
    <span className="eval-category" style={{ background: color + '22', color, borderColor: color + '44' }}>
      {label}
    </span>
  );
}

// ─── Score pill ────────────────────────────────────────────────────────────────
function ScorePill({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.75 ? '#10b981' : score >= 0.25 ? '#f59e0b' : '#ef4444';
  return (
    <span className="score-pill" style={{ background: color + '22', color, borderColor: color + '44' }}>
      {pct}%
    </span>
  );
}

// ─── Evidence chain drawer ─────────────────────────────────────────────────────
function EvidenceDrawer({
  question, onClose,
}: { question: EvalQuestion; onClose: () => void }) {
  const [chain, setChain] = useState<EvalAnswerResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchEvalAnswer(question.question)
      .then(r => setChain(r))
      .finally(() => setLoading(false));
  }, [question.question]);

  return (
    <div className="eval-drawer">
      <div className="eval-drawer-header">
        <div className="eval-drawer-title-row">
          <CategoryBadge category={question.category} label={question.category_label} />
          <span className="eval-drawer-title">{question.question}</span>
        </div>
        <button className="eval-drawer-close" onClick={onClose}>✕</button>
      </div>

      {/* 标准答案 */}
      <div className="eval-answer-bar">
        <span className="eval-answer-label">Ground Truth</span>
        <span className="eval-answer-value">{question.ground_truth}</span>
      </div>

      {/* ① Effective Memory 检索结果（主要内容，放最前） */}
      <div className="eval-drawer-section">
        <div className="eval-drawer-label">
          🔍 从 Effective Memory 检索到的证据链
          {chain && <span className="eval-badge">共 {chain.retrieved_chain.length} 条命中 / {chain.total_effective} 条总计</span>}
        </div>
        {loading && (
          <div className="eval-loading-bar">
            <div className="eval-loading-spinner" />
            正在检索 Effective Memory…
          </div>
        )}
        {!loading && chain && chain.retrieved_chain.length === 0 && (
          <div className="eval-empty">未检索到相关 Effective Memory（关键词不匹配）</div>
        )}
        {!loading && chain && chain.retrieved_chain.map((eff, i) => (
          <div key={eff.effective_id} className="eval-chain-card">
            <div className="eval-chain-header">
              <span className="eval-chain-rank">#{i + 1}</span>
              <span className="eval-chain-score">相关分 {eff.retrieval_score}</span>
              {(eff.time_range?.start || eff.time_range?.end) && (
                <span className="eval-chain-time">
                  {eff.time_range.start}{eff.time_range.end && eff.time_range.end !== eff.time_range.start ? ` → ${eff.time_range.end}` : ''}
                </span>
              )}
            </div>
            <div className="eval-chain-topic">{eff.topic}</div>
            <div className="eval-chain-summary">{eff.summary}</div>
            {eff.facts.length > 0 && (
              <ul className="eval-chain-facts">
                {eff.facts.map((f, j) => <li key={j}>{f}</li>)}
              </ul>
            )}
            {/* 该 effective 关联的原始 session 对话 */}
            {eff.source_sessions.length > 0 && (
              <details className="eval-chain-sessions">
                <summary>📩 关联 Session 原文 ({eff.source_sessions.length} 条)</summary>
                <div className="eval-chain-sessions-list">
                  {eff.source_sessions.map(s => (
                    <div key={s.memory_id} className="eval-session-turn">
                      <span className="eval-session-time">{s.timestamp?.slice(0, 10)}</span>
                      <span className="eval-session-text">{s.user_content}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}
      </div>

      {/* ② LocoMo 原始证据（折叠，作为参考对照） */}
      <div className="eval-drawer-section">
        <details className="eval-locomo-details">
          <summary className="eval-drawer-label eval-locomo-summary">
            📌 LocoMo 原始标注证据（参考）
            <span className="eval-badge">{question.evidence_ids.join('  ')}</span>
          </summary>
          <div className="eval-evidence-list" style={{ marginTop: 8 }}>
            {question.evidence_texts.map(ev => (
              <div key={ev.dia_id} className="eval-evidence-turn">
                <span className="eval-evidence-meta">
                  {ev.dia_id} · {ev.session} · {ev.date_time}
                </span>
                <span className={`eval-evidence-speaker speaker-${ev.speaker.toLowerCase().slice(0, 3)}`}>
                  {ev.speaker}
                </span>
                <span className="eval-evidence-text">{ev.text}</span>
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}

// ─── Report panel ──────────────────────────────────────────────────────────────
function ReportPanel({ reports }: { reports: ComparisonReport[] }) {
  const [activeReport, setActiveReport] = useState<ComparisonReport | null>(
    reports.length > 0 ? reports[reports.length - 1] : null,
  );
  const [tab, setTab] = useState<'before' | 'after'>('after');

  if (reports.length === 0) return <div className="eval-empty">暂无 Harness 评测报告</div>;

  const results = tab === 'before' ? activeReport?.before_results : activeReport?.after_results;

  return (
    <div className="eval-report">
      <div className="eval-report-header">
        <div className="eval-report-select-row">
          <select
            className="eval-report-select"
            value={activeReport?._filename ?? ''}
            onChange={e => setActiveReport(reports.find(r => r._filename === e.target.value) ?? null)}
          >
            {reports.map(r => (
              <option key={r._filename} value={r._filename}>{r._filename}</option>
            ))}
          </select>
        </div>
        {activeReport && (
          <div className="eval-report-stats">
            <div className="eval-stat-block">
              <span className="eval-stat-label">Before</span>
              <ScorePill score={activeReport.before_score} />
            </div>
            <div className="eval-arrow">→</div>
            <div className="eval-stat-block">
              <span className="eval-stat-label">After</span>
              <ScorePill score={activeReport.after_score} />
            </div>
            <div className="eval-improvement">
              +{(activeReport.improvement_rate).toFixed(1)}% 🎉
            </div>
          </div>
        )}
      </div>

      {activeReport && (
        <>
          <div className="eval-tab-row">
            <button
              className={`eval-tab-btn ${tab === 'before' ? 'active' : ''}`}
              onClick={() => setTab('before')}
            >Before ({activeReport.before_results.length}条)</button>
            <button
              className={`eval-tab-btn ${tab === 'after' ? 'active' : ''}`}
              onClick={() => setTab('after')}
            >After ({activeReport.after_results.length}条)</button>
          </div>
          <div className="eval-qa-list">
            {results?.map((r, i) => (
              <div key={i} className="eval-qa-row">
                <ScorePill score={r.score} />
                <div className="eval-qa-content">
                  <div className="eval-qa-q">{r.question}</div>
                  <div className="eval-qa-gt">Ground truth: <em>{r.ground_truth}</em></div>
                  <div className="eval-qa-pred">Predicted: {r.predicted}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Main EvalPage ─────────────────────────────────────────────────────────────
export default function EvalPage() {
  const [questions, setQuestions] = useState<EvalQuestion[]>([]);
  const [reports, setReports] = useState<ComparisonReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<EvalQuestion | null>(null);
  const [activeTab, setActiveTab] = useState<'questions' | 'report'>('questions');
  const [filterCat, setFilterCat] = useState<number>(0);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const qRes = await fetchEvalQuestions('caroline');
        if (!cancelled) setQuestions(qRes.data ?? []);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
      try {
        const rRes = await fetchEvalReport();
        if (!cancelled) setReports(rRes.reports ?? []);
      } catch (_) { /* report 失败不阻塞 */ }
      if (!cancelled) setLoading(false);
    })();
    return () => { cancelled = true; };
  }, []);

  const filtered = questions.filter(q => {
    if (filterCat && q.category !== filterCat) return false;
    if (searchText && !q.question.toLowerCase().includes(searchText.toLowerCase())) return false;
    return true;
  });

  const handleSelect = useCallback((q: EvalQuestion) => {
    setSelected(prev => (prev?.idx === q.idx ? null : q));
  }, []);

  const CATS = [
    { v: 0, label: 'All' },
    { v: 1, label: 'Single-hop' },
    { v: 2, label: 'Temporal' },
    { v: 3, label: 'Multi-hop' },
    { v: 4, label: 'Adversarial' },
    { v: 5, label: 'Open-ended' },
  ];

  return (
    <div className="eval-page">
      {/* ── Sidebar: question list ── */}
      <div className="eval-sidebar">
        <div className="eval-sidebar-header">
          <div className="eval-main-tabs">
            <button
              className={`eval-main-tab ${activeTab === 'questions' ? 'active' : ''}`}
              onClick={() => setActiveTab('questions')}
            >📋 Questions</button>
            <button
              className={`eval-main-tab ${activeTab === 'report' ? 'active' : ''}`}
              onClick={() => setActiveTab('report')}
            >📊 Report</button>
          </div>
        </div>

        {activeTab === 'questions' && (
          <>
            <div className="eval-filters">
              <input
                className="eval-search"
                placeholder="搜索问题…"
                value={searchText}
                onChange={e => setSearchText(e.target.value)}
              />
              <div className="eval-cat-filters">
                {CATS.map(c => (
                  <button
                    key={c.v}
                    className={`eval-cat-btn ${filterCat === c.v ? 'active' : ''}`}
                    style={c.v ? { borderColor: CATEGORY_COLORS[c.v] + '66', color: filterCat === c.v ? CATEGORY_COLORS[c.v] : undefined } : {}}
                    onClick={() => setFilterCat(c.v)}
                  >{c.label}</button>
                ))}
              </div>
              <div className="eval-count">{filtered.length} / {questions.length} 条</div>
            </div>

            {loading && <div className="eval-loading">Loading questions…</div>}
            {!loading && error && (
              <div className="eval-empty" style={{ color: '#f87171', padding: '12px 14px' }}>
                ❌ 加载失败: {error}
              </div>
            )}

            <div className="eval-question-list">
              {filtered.map(q => (
                <div
                  key={q.idx}
                  className={`eval-question-row ${selected?.idx === q.idx ? 'selected' : ''}`}
                  onClick={() => handleSelect(q)}
                >
                  <div className="eval-question-row-top">
                    <CategoryBadge category={q.category} label={q.category_label} />
                    <span className="eval-evidence-tags">
                      {q.evidence_ids.map(e => (
                        <span key={e} className="eval-dia-tag">{e}</span>
                      ))}
                    </span>
                  </div>
                  <div className="eval-question-text">{q.question}</div>
                  <div className="eval-question-gt">✔ {q.ground_truth}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {activeTab === 'report' && (
          <div className="eval-report-tab">
            <ReportPanel reports={reports} />
          </div>
        )}
      </div>

      {/* ── Main: evidence drawer ── */}
      <div className="eval-main">
        {!selected ? (
          <div className="eval-placeholder">
            <div className="eval-placeholder-icon">🔍</div>
            <div className="eval-placeholder-text">点击左侧问题，查看证据链</div>
            <div className="eval-placeholder-sub">
              系统将展示：LocoMo 原始证据对话 → 检索到的 Effective Memory → 关联 Session 原文
            </div>
          </div>
        ) : (
          <EvidenceDrawer question={selected} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  );
}

