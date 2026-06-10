import { useState } from 'react';
import type { LongTermMemory, SelectionState } from '../types';
import './Panel.css';

interface Props {
  memories: LongTermMemory[];
  selection: SelectionState;
  onSelect: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onUpdate: (mem: LongTermMemory, patch: Partial<LongTermMemory>) => Promise<void>;
}

function ConfBar({ value }: { value: number }) {
  return (
    <div className="conf-bar-wrap" title={`Confidence: ${(value * 100).toFixed(0)}%`}>
      <div className="conf-bar-fill" style={{ width: `${value * 100}%`, background: value > 0.8 ? '#22c55e' : value > 0.5 ? '#f59e0b' : '#ef4444' }} />
    </div>
  );
}

function EditModal({ mem, onSave, onClose }: { mem: LongTermMemory; onSave: (patch: Partial<LongTermMemory>) => void; onClose: () => void }) {
  const [topic, setTopic] = useState(mem.topic);
  const [summary, setSummary] = useState(mem.summary);
  const [factsText, setFactsText] = useState(mem.facts.join('\n'));
  const [tagsText, setTagsText] = useState(mem.tags.join(', '));

  const handleSave = () => {
    onSave({
      topic: topic.trim() || mem.topic,
      summary: summary.trim() || mem.summary,
      facts: factsText.split('\n').map(f => f.trim()).filter(Boolean),
      tags: tagsText.split(',').map(t => t.trim()).filter(Boolean),
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h3>Edit Long-term Memory</h3>
        <label>Topic
          <input value={topic} onChange={e => setTopic(e.target.value)} />
        </label>
        <label>Summary
          <textarea rows={4} value={summary} onChange={e => setSummary(e.target.value)} />
        </label>
        <label>Facts (one per line)
          <textarea rows={6} value={factsText} onChange={e => setFactsText(e.target.value)} />
        </label>
        <label>Tags (comma-separated)
          <input value={tagsText} onChange={e => setTagsText(e.target.value)} />
        </label>
        <div className="modal-actions">
          <button className="btn-save" onClick={handleSave}>Save</button>
          <button className="btn-cancel" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

export function LongtermPanel({ memories, selection, onSelect, onDelete, onUpdate }: Props) {
  const [editing, setEditing] = useState<LongTermMemory | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleDelete = async (id: string) => {
    await onDelete(id);
    setConfirmDelete(null);
  };

  const handleSave = async (patch: Partial<LongTermMemory>) => {
    if (!editing) return;
    await onUpdate(editing, patch);
    setEditing(null);
  };

  return (
    <div className="panel panel-longterm">
      <div className="panel-header">
        <span className="panel-icon">🏛️</span>
        <span className="panel-title">Long-term Memory</span>
        <span className="panel-count">{memories.length}</span>
      </div>
      <div className="panel-body">
        {memories.map(m => {
          const isSelected = selection.layer === 'longterm' && selection.id === m.memory_id;
          const isHighlighted = selection.highlightedLongtermIds.has(m.memory_id);
          const isActive = isSelected || isHighlighted;
          return (
            <div
              key={m.memory_id}
              data-card-id={m.memory_id}
              className={`card card-longterm ${isSelected ? 'selected' : ''} ${isHighlighted && !isSelected ? 'highlighted' : ''}`}
              onClick={() => onSelect(m.memory_id)}
              ref={isActive ? (el => el?.scrollIntoView({ block: 'nearest' })) : undefined}
            >
              <div className="card-topic">{m.topic}</div>

              <ConfBar value={m.confidence} />

              <div className="card-meta-row">
                <span className="type-badge" style={{ background: m.memory_type === 'LongTermMemory' ? '#6366f1' : '#0ea5e9' }}>
                  {m.memory_type}
                </span>
                <span className="src-badge">← {m.source_effective_ids.length} sessions</span>
                {m.time_range_start && (
                  <span className="card-time-range">{m.time_range_start.slice(0, 10)}</span>
                )}
              </div>

              <div className="card-summary">{m.summary.slice(0, 140)}…</div>

              <div className="card-facts-preview">
                {m.facts.slice(0, 2).map((f, i) => (
                  <div key={i} className="fact-item">• {f.slice(0, 80)}{f.length > 80 ? '…' : ''}</div>
                ))}
                {m.facts.length > 2 && <div className="fact-more">+{m.facts.length - 2} more</div>}
              </div>

              <div className="card-tags">
                {m.tags.slice(0, 5).map(t => <span key={t} className="tag">{t}</span>)}
              </div>

              {/* CRUD 操作按钮 */}
              <div className="card-actions" onClick={e => e.stopPropagation()}>
                <button className="btn-edit" onClick={() => setEditing(m)} title="Edit">✏️</button>
                {confirmDelete === m.memory_id ? (
                  <>
                    <button className="btn-confirm-del" onClick={() => handleDelete(m.memory_id)}>Confirm</button>
                    <button className="btn-cancel-del" onClick={() => setConfirmDelete(null)}>Cancel</button>
                  </>
                ) : (
                  <button className="btn-delete" onClick={() => setConfirmDelete(m.memory_id)} title="Delete">🗑️</button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {editing && (
        <EditModal
          mem={editing}
          onSave={handleSave}
          onClose={() => setEditing(null)}
        />
      )}
    </div>
  );
}

