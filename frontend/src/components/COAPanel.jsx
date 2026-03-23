/**
 * COAPanel.jsx — Course of Action planning panel (E9.2).
 *
 * Displays a ranked list of AI-generated corps-level COAs, allows the human
 * planner to inspect scores and explanations, select a COA for activation,
 * and request re-evaluation after modifying strategy parameters.
 *
 * Props:
 *   apiBase     {string}         Base URL of the COA REST API (e.g. "http://localhost:5000")
 *   envKwargs   {object}         CorpsEnv kwargs forwarded to the backend (e.g. {n_divisions: 3})
 *   onActivate  {function}       Called with a CorpsCourseOfAction dict when the planner
 *                                selects a COA to execute.
 */

import React, { useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Horizontal progress bar with optional label. */
function Bar({ value, color, label }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div style={styles.barRow}>
      {label && <span style={styles.barLabel}>{label}</span>}
      <div style={styles.barTrack}>
        <div style={{ ...styles.barFill, width: `${pct}%`, background: color || '#4caf50' }} />
      </div>
      <span style={styles.barPct}>{Math.round(pct)}%</span>
    </div>
  );
}

/** Single COA card in the ranked list. */
function COACard({ coa, selected, onSelect, onExplain, onModify }) {
  const s = coa.score;
  const rankColor = coa.rank === 1 ? '#ffd700' : coa.rank === 2 ? '#c0c0c0' : coa.rank === 3 ? '#cd7f32' : '#e0e0e0';

  return (
    <div
      style={{ ...styles.card, borderColor: selected ? '#e94560' : '#0f3460', cursor: 'pointer' }}
      onClick={() => onSelect(coa)}
      role="button"
      aria-pressed={selected}
    >
      <div style={styles.cardHeader}>
        <span style={{ ...styles.rankBadge, color: rankColor }}>#{coa.rank}</span>
        <span style={styles.coaLabel}>{coa.label.replace(/_/g, ' ')}</span>
        <span style={styles.composite}>{(s.composite * 100).toFixed(1)}</span>
      </div>

      <Bar value={s.win_rate}    color="#4caf50" label="Win" />
      <Bar value={s.red_casualties} color="#e94560" label="Red ☠" />
      <Bar value={s.objective_completion} color="#2196f3" label="Obj" />
      <Bar value={s.supply_efficiency}    color="#ff9800" label="Supply" />

      <div style={styles.cardFooter}>
        <button
          style={styles.smallBtn}
          onClick={e => { e.stopPropagation(); onExplain(coa); }}
        >
          💡 Explain
        </button>
        <button
          style={styles.smallBtn}
          onClick={e => { e.stopPropagation(); onModify(coa); }}
        >
          ✏️ Modify
        </button>
      </div>
    </div>
  );
}

/** Explanation panel for a selected COA. */
function ExplanationPanel({ explanation, onClose }) {
  if (!explanation) return null;
  return (
    <div style={styles.explPanel}>
      <div style={styles.explHeader}>
        <span>💡 Explanation: <strong>{explanation.coa_label.replace(/_/g, ' ')}</strong></span>
        <button onClick={onClose} style={styles.closeBtn}>✕</button>
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Key Decisions</div>
        {explanation.key_decisions.map((d, i) => (
          <div key={i} style={styles.decision}>
            <span style={styles.decisionNum}>{i + 1}.</span>
            <span>{d}</span>
          </div>
        ))}
      </div>

      {explanation.winning_patterns.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Winning Command Patterns</div>
          {explanation.winning_patterns.map((pat, i) => (
            <div key={i} style={styles.pattern}>
              {pat.map((cmd, j) => (
                <span key={j} style={styles.cmdChip}>Div {j}: {cmd.replace(/_/g, ' ')}</span>
              ))}
            </div>
          ))}
        </div>
      )}

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Objective Reward by Phase</div>
        <div style={styles.timeline}>
          {['q1', 'q2', 'q3', 'q4'].map(q => {
            const val = explanation.objective_timeline?.[q] ?? 0;
            return (
              <div key={q} style={styles.timelineItem}>
                <span style={styles.timelineLabel}>{q.toUpperCase()}</span>
                <div style={styles.timelineBarTrack}>
                  <div style={{
                    ...styles.timelineBarFill,
                    width: `${Math.min(100, Math.abs(val) * 100)}%`,
                    background: val >= 0 ? '#4caf50' : '#e94560',
                  }} />
                </div>
                <span style={styles.timelineValue}>{val.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** COA modification form. */
function ModifyPanel({ coa, corpStrategies, onSubmit, onClose, loading }) {
  const [strategyOverride, setStrategyOverride] = useState(coa.label);
  const [nRollouts, setNRollouts] = useState(5);

  const handleSubmit = () => {
    onSubmit(coa, {
      strategy_override: strategyOverride !== coa.label ? strategyOverride : null,
      n_rollouts: nRollouts,
    });
  };

  return (
    <div style={styles.explPanel}>
      <div style={styles.explHeader}>
        <span>✏️ Modify COA: <strong>{coa.label.replace(/_/g, ' ')}</strong></span>
        <button onClick={onClose} style={styles.closeBtn}>✕</button>
      </div>

      <div style={styles.section}>
        <label style={styles.fieldLabel}>Strategy Override</label>
        <select
          value={strategyOverride}
          onChange={e => setStrategyOverride(e.target.value)}
          style={styles.select}
        >
          {corpStrategies.map(s => (
            <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      <div style={styles.section}>
        <label style={styles.fieldLabel}>Re-simulation Rollouts</label>
        <input
          type="number"
          min={1}
          max={20}
          value={nRollouts}
          onChange={e => setNRollouts(Number(e.target.value))}
          style={styles.numInput}
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }}
      >
        {loading ? '⏳ Re-simulating…' : '▶ Re-simulate'}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main COAPanel component
// ---------------------------------------------------------------------------

const CORPS_STRATEGY_LABELS = [
  'full_advance', 'fortress_defense', 'left_envelopment', 'right_envelopment',
  'pincer_attack', 'fire_superiority', 'advance_and_fix', 'feint_and_assault',
  'strategic_withdrawal', 'rapid_exploitation',
];

export default function COAPanel({ apiBase = 'http://localhost:5000', envKwargs = {}, onActivate }) {
  const [coas, setCoas]             = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [selectedCoa, setSelectedCoa] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [modifyCoa, setModifyCoa]   = useState(null);
  const [modLoading, setModLoading] = useState(false);

  const [nRollouts, setNRollouts]   = useState(10);
  const [nCoas, setNCoas]           = useState(10);

  // ── Generate COAs ─────────────────────────────────────────────────────────
  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setExplanation(null);
    setModifyCoa(null);
    try {
      const res = await fetch(`${apiBase}/corps/coas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_rollouts: nRollouts,
          n_coas: nCoas,
          env_kwargs: envKwargs,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Unknown error');
      } else {
        setCoas(data.coas || []);
      }
    } catch (e) {
      setError(`Network error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [apiBase, envKwargs, nRollouts, nCoas]);

  // ── Explain a COA ─────────────────────────────────────────────────────────
  const handleExplain = useCallback(async (coa) => {
    setExplanation(null);
    setModifyCoa(null);
    try {
      const res = await fetch(`${apiBase}/corps/coas/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coa, n_rollouts: 5, env_kwargs: envKwargs }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Explanation failed');
      } else {
        setExplanation(data.explanation);
      }
    } catch (e) {
      setError(`Network error: ${e.message}`);
    }
  }, [apiBase, envKwargs]);

  // ── Modify a COA ──────────────────────────────────────────────────────────
  const handleModifySubmit = useCallback(async (coa, modification) => {
    setModLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/corps/coas/modify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coa, modification, env_kwargs: envKwargs }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Modification failed');
      } else {
        // Replace or append the modified COA in the list.
        // A COA is identified by both label and seed; after modify_and_evaluate()
        // the label may change (strategy_override) and the seed is always
        // offset by +999 from the original, so we match by the *original* coa's
        // label+seed which was captured in the closure.
        const updated = data.coa;
        setCoas(prev => {
          const idx = prev.findIndex(c => c.label === coa.label && c.seed === coa.seed);
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = { ...updated, rank: idx + 1 };
            return copy;
          }
          return [updated, ...prev];
        });
        setModifyCoa(null);
      }
    } catch (e) {
      setError(`Network error: ${e.message}`);
    } finally {
      setModLoading(false);
    }
  }, [apiBase, envKwargs]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>
      <div style={styles.header}>
        <span style={styles.title}>🗺 COA Planner</span>
      </div>

      {/* Controls */}
      <div style={styles.controls}>
        <div style={styles.controlRow}>
          <label style={styles.controlLabel}>COAs</label>
          <input
            type="number" min={1} max={10} value={nCoas}
            onChange={e => setNCoas(Number(e.target.value))}
            style={styles.numInputSm}
          />
          <label style={styles.controlLabel}>Rollouts</label>
          <input
            type="number" min={1} max={50} value={nRollouts}
            onChange={e => setNRollouts(Number(e.target.value))}
            style={styles.numInputSm}
          />
          <button
            onClick={handleGenerate}
            disabled={loading}
            style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }}
          >
            {loading ? '⏳ Generating…' : '⚡ Generate'}
          </button>
        </div>
      </div>

      {error && <div style={styles.error}>⚠ {error}</div>}

      {/* COA List */}
      {coas.length > 0 && (
        <div style={styles.coaList}>
          {coas.map(coa => (
            <COACard
              key={`${coa.label}-${coa.seed}`}
              coa={coa}
              selected={selectedCoa?.label === coa.label && selectedCoa?.seed === coa.seed}
              onSelect={setSelectedCoa}
              onExplain={handleExplain}
              onModify={c => { setModifyCoa(c); setExplanation(null); }}
            />
          ))}
        </div>
      )}

      {/* Activate button */}
      {selectedCoa && onActivate && (
        <button
          onClick={() => onActivate(selectedCoa)}
          style={{ ...styles.btn, marginTop: 8, background: '#e94560', width: '100%' }}
        >
          ✅ Activate: {selectedCoa.label.replace(/_/g, ' ')}
        </button>
      )}

      {/* Explanation panel */}
      {explanation && (
        <ExplanationPanel
          explanation={explanation}
          onClose={() => setExplanation(null)}
        />
      )}

      {/* Modify panel */}
      {modifyCoa && (
        <ModifyPanel
          coa={modifyCoa}
          corpStrategies={CORPS_STRATEGY_LABELS}
          onSubmit={handleModifySubmit}
          onClose={() => setModifyCoa(null)}
          loading={modLoading}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Styles (inline — no CSS build step required)
// ---------------------------------------------------------------------------

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    padding: '8px 0',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 14,
    fontWeight: 700,
    color: '#e0e0e0',
  },
  controls: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  controlLabel: {
    fontSize: 11,
    color: '#a0a0b0',
  },
  numInputSm: {
    width: 44,
    padding: '3px 5px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #304070',
    borderRadius: 4,
    fontSize: 12,
  },
  btn: {
    padding: '6px 14px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 12,
  },
  error: {
    padding: '6px 10px',
    background: '#3a1030',
    color: '#ff6060',
    borderRadius: 6,
    fontSize: 12,
  },
  coaList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    maxHeight: 420,
    overflowY: 'auto',
  },
  card: {
    padding: '8px 10px',
    background: '#16213e',
    borderRadius: 8,
    border: '1px solid',
    borderColor: '#0f3460',
    userSelect: 'none',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  rankBadge: {
    fontSize: 13,
    fontWeight: 700,
    minWidth: 24,
  },
  coaLabel: {
    flex: 1,
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'capitalize',
  },
  composite: {
    fontSize: 13,
    fontWeight: 700,
    color: '#4caf50',
  },
  cardFooter: {
    display: 'flex',
    gap: 6,
    marginTop: 6,
  },
  smallBtn: {
    padding: '3px 8px',
    background: '#0f3460',
    color: '#b0b0c0',
    border: '1px solid #304070',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 11,
  },
  barRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    marginBottom: 3,
  },
  barLabel: {
    fontSize: 10,
    width: 40,
    color: '#909090',
    textAlign: 'right',
  },
  barTrack: {
    flex: 1,
    height: 6,
    background: '#1e3060',
    borderRadius: 3,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
    transition: 'width 0.4s ease',
  },
  barPct: {
    fontSize: 10,
    width: 28,
    color: '#909090',
    textAlign: 'right',
  },
  // Explanation styles
  explPanel: {
    padding: '10px 12px',
    background: '#0f2040',
    borderRadius: 8,
    border: '1px solid #2040a0',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  explHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: 13,
    color: '#e0e0e0',
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: '#909090',
    cursor: 'pointer',
    fontSize: 14,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 5,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: 700,
    color: '#8090b0',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  decision: {
    display: 'flex',
    gap: 6,
    fontSize: 12,
    color: '#d0d0d0',
    lineHeight: 1.4,
  },
  decisionNum: {
    color: '#6080c0',
    fontWeight: 700,
    minWidth: 16,
  },
  pattern: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 4,
  },
  cmdChip: {
    padding: '2px 7px',
    background: '#1a3060',
    color: '#8090d0',
    borderRadius: 10,
    fontSize: 11,
  },
  timeline: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  timelineItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  timelineLabel: {
    fontSize: 10,
    color: '#8090b0',
    width: 20,
  },
  timelineBarTrack: {
    flex: 1,
    height: 8,
    background: '#1e3060',
    borderRadius: 4,
    overflow: 'hidden',
  },
  timelineBarFill: {
    height: '100%',
    borderRadius: 4,
    transition: 'width 0.4s ease',
  },
  timelineValue: {
    fontSize: 10,
    color: '#8090b0',
    width: 44,
    textAlign: 'right',
  },
  // Modify panel
  fieldLabel: {
    fontSize: 11,
    color: '#8090b0',
    marginBottom: 3,
  },
  select: {
    width: '100%',
    padding: '5px 8px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #304070',
    borderRadius: 5,
    fontSize: 12,
  },
  numInput: {
    width: 70,
    padding: '4px 6px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #304070',
    borderRadius: 5,
    fontSize: 12,
  },
};
