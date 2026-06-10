import { useEffect, useReducer, useState, useCallback, useRef } from 'react';
import type { SessionMemory, EffectiveMemory, LongTermMemory, Links, SelectionState } from './types';
import {
  fetchSessions, fetchEffectives, fetchLongtermemories, fetchLinks,
  patchLongterm, deleteLongterm,
} from './api';
import { SessionPanel } from './components/SessionPanel';
import { EffectivePanel } from './components/EffectivePanel';
import { LongtermPanel } from './components/LongtermPanel';
import { ConnectionLines } from './components/ConnectionLines';
import EvalPage from './components/EvalPage';
import './App.css';

// ─── state ────────────────────────────────────────────────────────────────────

interface AppState {
  sessions: SessionMemory[];
  effectives: EffectiveMemory[];
  longtermemories: LongTermMemory[];
  links: Links;
  loading: boolean;
  error: string | null;
  selection: SelectionState;
}

const EMPTY_LINKS: Links = { session_to_effective: {}, effective_to_longterm: {} };
const EMPTY_SEL: SelectionState = {
  layer: null, id: null,
  highlightedSessionIds: new Set(),
  highlightedEffectiveIds: new Set(),
  highlightedLongtermIds: new Set(),
};

type Action =
  | { type: 'LOADED'; sessions: SessionMemory[]; effectives: EffectiveMemory[]; longtermemories: LongTermMemory[]; links: Links }
  | { type: 'ERROR'; msg: string }
  | { type: 'SELECT'; layer: SelectionState['layer']; id: string; links: Links; sessions: SessionMemory[]; effectives: EffectiveMemory[]; longtermemories: LongTermMemory[] }
  | { type: 'DESELECT' }
  | { type: 'DELETE_LT'; id: string }
  | { type: 'UPDATE_LT'; mem: LongTermMemory };

function computeSelection(
  layer: SelectionState['layer'],
  id: string,
  links: Links,
  _sessions: SessionMemory[],
  effectives: EffectiveMemory[],
  longtermemories: LongTermMemory[],
): SelectionState {
  const sessionIds = new Set<string>();
  const effectiveIds = new Set<string>();
  const longtermIds = new Set<string>();

  if (layer === 'session') {
    sessionIds.add(id);
    // session → effective
    const effIds = links.session_to_effective[id] ?? [];
    effIds.forEach(eid => {
      effectiveIds.add(eid);
      // effective → longterm
      (links.effective_to_longterm[eid] ?? []).forEach(lid => longtermIds.add(lid));
    });
  } else if (layer === 'effective') {
    effectiveIds.add(id);
    // effective → sessions
    const eff = effectives.find(e => e.effective_id === id);
    (eff?.source_memory_ids ?? []).forEach(sid => sessionIds.add(sid));
    // effective → longterm
    (links.effective_to_longterm[id] ?? []).forEach(lid => longtermIds.add(lid));
  } else if (layer === 'longterm') {
    longtermIds.add(id);
    const lt = longtermemories.find(m => m.memory_id === id);
    // longterm → session ids it references
    const ltSessionIds = new Set(lt?.source_effective_ids ?? []);
    // find effectives that share session ids
    effectives.forEach(eff => {
      const shared = eff.source_memory_ids.some(sid => ltSessionIds.has(sid));
      if (shared) {
        effectiveIds.add(eff.effective_id);
        eff.source_memory_ids.forEach(sid => sessionIds.add(sid));
      }
    });
  }

  return { layer, id, highlightedSessionIds: sessionIds, highlightedEffectiveIds: effectiveIds, highlightedLongtermIds: longtermIds };
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'LOADED':
      return { ...state, loading: false, sessions: action.sessions, effectives: action.effectives, longtermemories: action.longtermemories, links: action.links };
    case 'ERROR':
      return { ...state, loading: false, error: action.msg };
    case 'SELECT':
      return {
        ...state,
        selection: computeSelection(action.layer, action.id, action.links, action.sessions, action.effectives, action.longtermemories),
      };
    case 'DESELECT':
      return { ...state, selection: EMPTY_SEL };
    case 'DELETE_LT':
      return { ...state, longtermemories: state.longtermemories.filter(m => m.memory_id !== action.id), selection: EMPTY_SEL };
    case 'UPDATE_LT':
      return { ...state, longtermemories: state.longtermemories.map(m => m.memory_id === action.mem.memory_id ? action.mem : m) };
    default:
      return state;
  }
}

// ─── App ──────────────────────────────────────────────────────────────────────

// ─── 可用的 User/Group 列表 ────────────────────────────────────────────────────

const AVAILABLE_USERS = [
  { value: 'caroline', label: '👤 Caroline (group_0)', group: 'group_0_caroline' },
  { value: 'gina', label: '👤 Gina (group_1)', group: 'group_1_gina' },
  { value: 'melanie', label: '👤 Melanie (group_0)', group: 'group_0_melanie' },
  { value: 'jon', label: '👤 Jon (group_1)', group: 'group_1_jon' },
];

export default function App() {
  const [state, dispatch] = useReducer(reducer, {
    sessions: [], effectives: [], longtermemories: [], links: EMPTY_LINKS,
    loading: true, error: null, selection: EMPTY_SEL,
  });

  const [userId, setUserId] = useState('caroline');
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');
  const [appTab, setAppTab] = useState<'explorer' | 'eval'>('explorer');
  const columnsRef = useRef<HTMLElement>(null);

  useEffect(() => {
    dispatch({ type: 'LOADED', sessions: [], effectives: [], longtermemories: [], links: EMPTY_LINKS });
    Promise.all([
      fetchSessions(userId, order),
      fetchEffectives(order),  // TODO: 如果未来按 user 过滤 effective，需要传 userId
      fetchLongtermemories(userId, order),
      fetchLinks(userId),
    ]).then(([sessions, effectives, longtermemories, links]) => {
      dispatch({ type: 'LOADED', sessions, effectives, longtermemories, links });
    }).catch(e => dispatch({ type: 'ERROR', msg: String(e) }));
  }, [userId, order]);  // ✅ 切换 User 时重新加载

  const handleSelect = useCallback((layer: SelectionState['layer'], id: string) => {
    if (state.selection.id === id && state.selection.layer === layer) {
      dispatch({ type: 'DESELECT' });
    } else {
      dispatch({ type: 'SELECT', layer, id, links: state.links, sessions: state.sessions, effectives: state.effectives, longtermemories: state.longtermemories });
    }
  }, [state]);

  const handleDeleteLt = useCallback(async (id: string) => {
    await deleteLongterm(id);
    dispatch({ type: 'DELETE_LT', id });
  }, []);

  const handleUpdateLt = useCallback(async (mem: LongTermMemory, patch: Partial<LongTermMemory>) => {
    await patchLongterm(mem.memory_id, patch);
    dispatch({ type: 'UPDATE_LT', mem: { ...mem, ...patch } });
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 Memory Explorer</h1>
        <div className="app-tabs">
          <button
            className={`app-tab ${appTab === 'explorer' ? 'active' : ''}`}
            onClick={() => setAppTab('explorer')}
          >🗂 Memory</button>
          <button
            className={`app-tab ${appTab === 'eval' ? 'active' : ''}`}
            onClick={() => setAppTab('eval')}
          >📋 Eval</button>
        </div>
        <span className="user-selector">
          <label htmlFor="user-select">User:</label>
          <select
            id="user-select"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            title="切换用户/组"
          >
            {AVAILABLE_USERS.map(u => (
              <option key={u.value} value={u.value}>{u.label}</option>
            ))}
          </select>
        </span>

        {appTab === 'explorer' && (
          <>
            <div className="header-stats">
              <span className="stat">Session <b>{state.sessions.length}</b></span>
              <span className="arrow">→</span>
              <span className="stat">Effective <b>{state.effectives.length}</b></span>
              <span className="arrow">→</span>
              <span className="stat">Long-term <b>{state.longtermemories.length}</b></span>
            </div>
            <div className="order-toggle">
              <button
                className={`order-btn ${order === 'asc' ? 'active' : ''}`}
                onClick={() => setOrder('asc')}
                title="时间升序（最早在上）"
              >↑ 升序</button>
              <button
                className={`order-btn ${order === 'desc' ? 'active' : ''}`}
                onClick={() => setOrder('desc')}
                title="时间降序（最新在上）"
              >↓ 降序</button>
            </div>
            {state.selection.id && (
              <button className="clear-btn" onClick={() => dispatch({ type: 'DESELECT' })}>✕ Clear selection</button>
            )}
          </>
        )}
      </header>

      {appTab === 'explorer' ? (
        state.loading ? (
          <div className="loading">Loading memories…</div>
        ) : state.error ? (
          <div className="error">Error: {state.error}<br /><small>Make sure the backend is running: python server.py</small></div>
        ) : (
        <main className="columns" ref={columnsRef as React.RefObject<HTMLElement>}>
          <ConnectionLines selection={state.selection} containerRef={columnsRef} />
          <SessionPanel
            sessions={state.sessions}
            selection={state.selection}
            onSelect={(id) => handleSelect('session', id)}
          />
          <EffectivePanel
            effectives={state.effectives}
            selection={state.selection}
            onSelect={(id) => handleSelect('effective', id)}
          />
          <LongtermPanel
            memories={state.longtermemories}
            selection={state.selection}
            onSelect={(id) => handleSelect('longterm', id)}
            onDelete={handleDeleteLt}
            onUpdate={handleUpdateLt}
          />
          </main>
        ))
        : (
        <main className="eval-container">
          <EvalPage />
        </main>
      )}
    </div>
  );
}
