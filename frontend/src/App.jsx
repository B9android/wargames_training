/**
 * App.jsx — top-level wargame interface (E9.1 / E9.2).
 *
 * Panels:
 *  - GameCanvas     : WebGL/Canvas 2D map renderer
 *  - UnitPanel      : Blue / Red battalion stats
 *  - FormationSelector : Choose formation for next order
 *  - OrderQueue     : Queued orders
 *  - ScenarioEditor : Drag-and-drop scenario setup
 *  - ReplayViewer   : Step-through replay
 *  - COAPanel       : AI-Assisted Course of Action planning (E9.2)
 */

import React, { useEffect, useReducer, useRef, useState } from 'react';
import useGameSocket from './hooks/useGameSocket.js';
import GameCanvas from './components/GameCanvas.jsx';
import UnitPanel from './components/UnitPanel.jsx';
import FormationSelector from './components/FormationSelector.jsx';
import OrderQueue from './components/OrderQueue.jsx';
import ScenarioEditor from './components/ScenarioEditor.jsx';
import ReplayViewer from './components/ReplayViewer.jsx';
import COAPanel from './components/COAPanel.jsx';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8765';
const COA_API_BASE = import.meta.env.VITE_COA_API_URL || 'http://localhost:5000';

const SCENARIOS = ['open_field', 'mountain_pass', 'last_stand'];

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

/** @typedef {'lobby'|'playing'|'episode_end'|'replay'|'editor'|'coa_planner'} AppMode */

const initialState = {
  /** @type {AppMode} */
  mode: 'lobby',
  scenario: 'open_field',
  difficulty: 5,
  /** @type {object|null} Latest frame from the server. */
  frame: null,
  /** @type {string|null} Episode outcome. */
  outcome: null,
  /** @type {object[]} Queued orders before submission. */
  orderQueue: [],
  /** @type {string} Currently selected formation. */
  formation: 'LINE',
  /** @type {object|null} Loaded replay metadata. */
  replayMeta: null,
  /** Current replay frame index. */
  replayIndex: 0,
  /** Total replay frames. */
  replayTotal: 0,
  /** @type {object|null} Current replay frame dict. */
  replayFrame: null,
  /** @type {object|null} COA selected by the human planner. */
  activeCoa: null,
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_MODE':
      return { ...state, mode: action.mode };
    case 'SET_SCENARIO':
      return { ...state, scenario: action.scenario };
    case 'SET_DIFFICULTY':
      return { ...state, difficulty: action.difficulty };
    case 'SET_FRAME':
      return { ...state, frame: action.frame };
    case 'EPISODE_END':
      return { ...state, mode: 'episode_end', outcome: action.outcome };
    case 'SET_FORMATION':
      return { ...state, formation: action.formation };
    case 'ENQUEUE_ORDER':
      return { ...state, orderQueue: [...state.orderQueue, action.order] };
    case 'CLEAR_ORDERS':
      return { ...state, orderQueue: [] };
    case 'REPLAY_LOADED':
      return { ...state, mode: 'replay', replayMeta: action.meta, replayIndex: 0, replayTotal: action.total, replayFrame: null };
    case 'REPLAY_FRAME':
      return { ...state, replayFrame: action.frame, replayIndex: action.index };
    case 'REPLAY_DONE':
      return { ...state };  // keep current frame; let the viewer disable play
    case 'ACTIVATE_COA':
      return { ...state, activeCoa: action.coa };
    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { send, lastMessage, readyState } = useGameSocket(WS_URL);

  // ── Handle incoming server messages ──────────────────────────────────────
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage;

    switch (msg.type) {
      case 'frame':
        dispatch({ type: 'SET_FRAME', frame: msg });
        break;

      case 'episode_end':
        dispatch({ type: 'EPISODE_END', outcome: msg.outcome });
        break;

      case 'replay_export':
        // Trigger JSON download in the browser.
        _downloadJSON(msg.replay, `replay_${Date.now()}.json`);
        break;

      case 'replay_loaded':
        dispatch({ type: 'REPLAY_LOADED', meta: msg.meta || {}, total: msg.total });
        break;

      case 'replay_frame':
        dispatch({ type: 'REPLAY_FRAME', frame: msg.frame, index: msg.index });
        break;

      case 'replay_done':
        dispatch({ type: 'REPLAY_DONE' });
        break;

      case 'error':
        console.error('[server error]', msg.message);
        break;

      default:
        break;
    }
  }, [lastMessage]);

  // ── Keyboard controls (game mode) ────────────────────────────────────────
  const keysRef = useRef({});

  useEffect(() => {
    if (state.mode !== 'playing') return;

    const onKeyDown = (e) => { keysRef.current[e.code] = true; };
    const onKeyUp = (e) => { keysRef.current[e.code] = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    const interval = setInterval(() => {
      const keys = keysRef.current;
      const move = keys['KeyW'] || keys['ArrowUp'] ? 1 : keys['KeyS'] || keys['ArrowDown'] ? -1 : 0;
      const rotate = keys['KeyA'] || keys['ArrowLeft'] ? 1 : keys['KeyD'] || keys['ArrowRight'] ? -1 : 0;
      const fire = keys['Space'] ? 1 : 0;
      if (move !== 0 || rotate !== 0 || fire !== 0) {
        // Include the currently selected formation so the order is
        // self-describing (formation changes are queued alongside movement).
        send({ type: 'action', move, rotate, fire, formation: state.formation });
      }
    }, 100);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      clearInterval(interval);
    };
  }, [state.mode, state.formation, send]);

  // ── Order queue: canvas click enqueues a "move-here" order ───────────────
  const handleMapClick = (normX, normY) => {
    if (state.mode !== 'playing') return;
    dispatch({
      type: 'ENQUEUE_ORDER',
      order: { move: normX > 0.5 ? 1 : -1, rotate: 0, fire: 0, formation: state.formation },
    });
  };

  // ── Action helpers ────────────────────────────────────────────────────────
  const startGame = () => {
    send({ type: 'start', scenario: state.scenario, difficulty: state.difficulty });
    dispatch({ type: 'SET_MODE', mode: 'playing' });
    dispatch({ type: 'SET_FRAME', frame: null });
  };

  const resetGame = () => {
    send({ type: 'reset' });
    dispatch({ type: 'SET_MODE', mode: 'playing' });
    dispatch({ type: 'SET_FRAME', frame: null });
  };

  const exportReplay = () => {
    send({ type: 'export_replay' });
  };

  const handleReplayFileLoad = (replayData) => {
    send({ type: 'load_replay', replay: replayData });
  };

  const replayStep = () => {
    send({ type: 'replay_step' });
  };

  const replaySeek = (index) => {
    send({ type: 'replay_seek', index });
  };

  const openEditor = () => {
    dispatch({ type: 'SET_MODE', mode: 'editor' });
  };

  const openCoaPlanner = () => {
    dispatch({ type: 'SET_MODE', mode: 'coa_planner' });
  };

  const handleCoaActivate = (coa) => {
    dispatch({ type: 'ACTIVATE_COA', coa });
    // Send the selected COA strategy to the game server so it can influence
    // AI behaviour; the server may choose to honour it or ignore it.
    send({ type: 'activate_coa', label: coa.label, seed: coa.seed });
    dispatch({ type: 'SET_MODE', mode: 'lobby' });
  };

  const loadScenarioFromEditor = (scenarioCfg) => {
    send({ type: 'load_scenario', ...scenarioCfg });
    dispatch({ type: 'SET_MODE', mode: 'playing' });
  };

  // ── Render ────────────────────────────────────────────────────────────────
  const wsStatus = readyState === 1 ? '🟢 Connected' : readyState === 0 ? '🟡 Connecting…' : '🔴 Disconnected';

  return (
    <div style={styles.root}>
      {/* ── Header ── */}
      <header style={styles.header}>
        <span style={styles.title}>⚔ Wargames Training</span>
        <span style={styles.wsStatus}>{wsStatus}</span>
      </header>

      {/* ── Lobby ── */}
      {state.mode === 'lobby' && (
        <div style={styles.lobby}>
          <h2 style={{ marginBottom: 20 }}>Select Scenario</h2>
          <div style={styles.row}>
            <label>Scenario: </label>
            <select value={state.scenario} onChange={e => dispatch({ type: 'SET_SCENARIO', scenario: e.target.value })} style={styles.select}>
              {SCENARIOS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div style={{ ...styles.row, marginTop: 12 }}>
            <label>Difficulty: </label>
            <input type="range" min={1} max={5} value={state.difficulty} onChange={e => dispatch({ type: 'SET_DIFFICULTY', difficulty: Number(e.target.value) })} style={{ flex: 1 }} />
            <span style={{ marginLeft: 8 }}>{state.difficulty}</span>
          </div>
          <div style={{ ...styles.row, marginTop: 20, gap: 12 }}>
            <button onClick={startGame} style={styles.btn}>▶ Start Game</button>
            <button onClick={openEditor} style={styles.btnSecondary}>🗺 Scenario Editor</button>
            <button onClick={openCoaPlanner} style={styles.btnSecondary}>🎯 COA Planner</button>
          </div>
          <div style={{ marginTop: 16 }}>
            <ReplayViewer
              onLoad={handleReplayFileLoad}
              onStep={replayStep}
              onSeek={replaySeek}
              replayFrame={null}
              replayIndex={0}
              replayTotal={0}
              mode="load_only"
            />
          </div>
          {state.activeCoa && (
            <div style={{ marginTop: 12, padding: '8px 12px', background: '#0f3460', borderRadius: 8, border: '1px solid #e94560', fontSize: 12 }}>
              🎯 Active COA: <strong>{state.activeCoa.label.replace(/_/g, ' ')}</strong>
              {' '}(composite: {(state.activeCoa.score.composite * 100).toFixed(1)})
            </div>
          )}
        </div>
      )}

      {/* ── Playing ── */}
      {(state.mode === 'playing' || state.mode === 'episode_end') && (
        <div style={styles.gameLayout}>
          {/* Left: map canvas */}
          <div style={styles.canvasWrapper}>
            <GameCanvas frame={state.frame} onMapClick={handleMapClick} />
            <div style={styles.controls}>
              <span style={{ fontSize: 12, opacity: 0.7 }}>WASD / Arrows to move · Space to fire · Click map to queue move order</span>
            </div>
          </div>

          {/* Right: side panel */}
          <div style={styles.sidePanel}>
            <UnitPanel frame={state.frame} />
            <FormationSelector
              current={state.formation}
              onChange={f => dispatch({ type: 'SET_FORMATION', formation: f })}
            />
            <OrderQueue
              orders={state.orderQueue}
              onSubmit={() => {
                state.orderQueue.forEach(o => send({ type: 'action', ...o }));
                dispatch({ type: 'CLEAR_ORDERS' });
              }}
              onClear={() => dispatch({ type: 'CLEAR_ORDERS' })}
            />

            {state.mode === 'episode_end' && (
              <div style={styles.outcomeBox}>
                <h3>Episode Ended</h3>
                <p style={{ margin: '8px 0' }}>Outcome: <strong>{state.outcome}</strong></p>
                <div style={styles.row}>
                  <button onClick={resetGame} style={styles.btn}>🔄 Replay</button>
                  <button onClick={exportReplay} style={styles.btnSecondary}>💾 Export Replay</button>
                  <button onClick={() => dispatch({ type: 'SET_MODE', mode: 'lobby' })} style={styles.btnSecondary}>🏠 Lobby</button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Replay Viewer ── */}
      {state.mode === 'replay' && (
        <div style={styles.gameLayout}>
          <div style={styles.canvasWrapper}>
            <GameCanvas frame={state.replayFrame} />
          </div>
          <div style={styles.sidePanel}>
            <UnitPanel frame={state.replayFrame} />
            <ReplayViewer
              onLoad={handleReplayFileLoad}
              onStep={replayStep}
              onSeek={replaySeek}
              replayFrame={state.replayFrame}
              replayIndex={state.replayIndex}
              replayTotal={state.replayTotal}
              mode="playback"
            />
            <button onClick={() => dispatch({ type: 'SET_MODE', mode: 'lobby' })} style={{ ...styles.btnSecondary, marginTop: 12 }}>🏠 Lobby</button>
          </div>
        </div>
      )}

      {/* ── Scenario Editor ── */}
      {state.mode === 'editor' && (
        <ScenarioEditor
          onLaunch={loadScenarioFromEditor}
          onCancel={() => dispatch({ type: 'SET_MODE', mode: 'lobby' })}
        />
      )}

      {/* ── COA Planner (E9.2) ── */}
      {state.mode === 'coa_planner' && (
        <div style={styles.coaPlannerLayout}>
          <div style={styles.coaPlannerHeader}>
            <span style={{ fontWeight: 700, color: '#e94560', fontSize: 18 }}>🎯 COA Planning Tool</span>
            <button
              onClick={() => dispatch({ type: 'SET_MODE', mode: 'lobby' })}
              style={styles.btnSecondary}
            >
              🏠 Lobby
            </button>
          </div>
          <COAPanel
            apiBase={COA_API_BASE}
            envKwargs={{ n_divisions: 3 }}
            onActivate={handleCoaActivate}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Styles (inline — no CSS build step required)
// ---------------------------------------------------------------------------

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    background: '#1a1a2e',
    color: '#e0e0e0',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 24px',
    background: '#16213e',
    borderBottom: '1px solid #0f3460',
  },
  title: {
    fontSize: 20,
    fontWeight: 700,
    color: '#e94560',
  },
  wsStatus: {
    fontSize: 13,
    opacity: 0.8,
  },
  lobby: {
    maxWidth: 480,
    margin: '80px auto',
    padding: 32,
    background: '#16213e',
    borderRadius: 12,
    border: '1px solid #0f3460',
  },
  gameLayout: {
    display: 'flex',
    flex: 1,
    gap: 16,
    padding: 16,
    height: 'calc(100vh - 56px)',
  },
  canvasWrapper: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  controls: {
    textAlign: 'center',
    padding: '4px 0',
  },
  sidePanel: {
    width: 280,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    overflowY: 'auto',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  select: {
    flex: 1,
    padding: '6px 10px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 6,
  },
  btn: {
    padding: '8px 18px',
    background: '#e94560',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
  },
  btnSecondary: {
    padding: '8px 18px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 6,
    cursor: 'pointer',
  },
  outcomeBox: {
    padding: 16,
    background: '#0f3460',
    borderRadius: 8,
    border: '1px solid #e94560',
  },
  coaPlannerLayout: {
    flex: 1,
    maxWidth: 720,
    margin: '24px auto',
    padding: 24,
    background: '#16213e',
    borderRadius: 12,
    border: '1px solid #0f3460',
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    overflowY: 'auto',
    width: '100%',
  },
  coaPlannerHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
};
