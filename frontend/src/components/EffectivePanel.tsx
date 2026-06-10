import type { EffectiveMemory, SelectionState } from '../types';
import './Panel.css';

interface Props {
  effectives: EffectiveMemory[];
  selection: SelectionState;
  onSelect: (id: string) => void;
}

const TYPE_COLORS: Record<string, string> = {
  LongTermMemory: '#6366f1',
  UserMemory: '#0ea5e9',
};

export function EffectivePanel({ effectives, selection, onSelect }: Props) {
  return (
    <div className="panel panel-effective">
      <div className="panel-header">
        <span className="panel-icon">✨</span>
        <span className="panel-title">Effective Memory</span>
        <span className="panel-count">{effectives.length}</span>
      </div>
      <div className="panel-body">
        {effectives.map(e => {
          const isSelected = selection.layer === 'effective' && selection.id === e.effective_id;
          const isHighlighted = selection.highlightedEffectiveIds.has(e.effective_id);
          const typeColor = TYPE_COLORS[e.memory_type] ?? '#64748b';
          const isActive = isSelected || isHighlighted;
          return (
            <div
              key={e.effective_id}
              data-card-id={e.effective_id}
              className={`card card-effective ${isSelected ? 'selected' : ''} ${isHighlighted && !isSelected ? 'highlighted' : ''}`}
              onClick={() => onSelect(e.effective_id)}
              ref={isActive ? (el => el?.scrollIntoView({ block: 'nearest' })) : undefined}
            >
              <div className="card-topic">{e.topic}</div>

              <div className="card-meta-row">
                <span className="type-badge" style={{ background: typeColor }}>{e.memory_type}</span>
                <span className="conf-badge">conf {(e.confidence * 100).toFixed(0)}%</span>
                <span className="src-badge">← {e.source_memory_ids.length} sessions</span>
              </div>

              <div className="card-summary">{e.summary.slice(0, 160)}…</div>

              <div className="card-facts-preview">
                {e.facts.slice(0, 2).map((f, i) => (
                  <div key={i} className="fact-item">• {f.slice(0, 80)}{f.length > 80 ? '…' : ''}</div>
                ))}
                {e.facts.length > 2 && <div className="fact-more">+{e.facts.length - 2} more</div>}
              </div>

              <div className="card-tags">
                {e.tags.slice(0, 5).map(t => <span key={t} className="tag">{t}</span>)}
              </div>

              {e.time_range && (
                <div className="card-time-range">
                  {e.time_range.start?.slice(0, 10)}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

