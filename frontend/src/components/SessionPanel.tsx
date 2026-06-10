import type { SessionMemory, SelectionState } from '../types';
import './Panel.css';

interface Props {
  sessions: SessionMemory[];
  selection: SelectionState;
  onSelect: (id: string) => void;
}

function formatTime(ts: string) {
  try { return new Date(ts).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); }
  catch { return ts; }
}

export function SessionPanel({ sessions, selection, onSelect }: Props) {
  return (
    <div className="panel panel-session">
      <div className="panel-header">
        <span className="panel-icon">💬</span>
        <span className="panel-title">Session Memory</span>
        <span className="panel-count">{sessions.length}</span>
      </div>
      <div className="panel-body">
        {sessions.map(s => {
          const isSelected = selection.layer === 'session' && selection.id === s.memory_id;
          const isHighlighted = selection.highlightedSessionIds.has(s.memory_id);
          const isActive = isSelected || isHighlighted;
          return (
            <div
              key={s.memory_id}
              data-card-id={s.memory_id}
              className={`card card-session ${isSelected ? 'selected' : ''} ${isHighlighted && !isSelected ? 'highlighted' : ''}`}
              onClick={() => onSelect(s.memory_id)}
              ref={isActive ? (el => el?.scrollIntoView({ block: 'nearest' })) : undefined}
            >
              <div className="card-time">{formatTime(s.timestamp)}</div>
              <div className="card-user-msg">
                <span className="role-badge role-user">U</span>
                <span className="msg-text">{s.user_content.slice(0, 120)}{s.user_content.length > 120 ? '…' : ''}</span>
              </div>
              {s.assistant_content && (
                <div className="card-assistant-msg">
                  <span className="role-badge role-assistant">A</span>
                  <span className="msg-text">{s.assistant_content.slice(0, 100)}{s.assistant_content.length > 100 ? '…' : ''}</span>
                </div>
              )}
              <div className="card-id">#{s.memory_id.slice(0, 8)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

