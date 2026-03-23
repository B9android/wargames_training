/**
 * UnitPanel.jsx
 *
 * Displays Blue and Red battalion statistics from the latest game frame.
 *
 * Props:
 *   frame  {object|null}  Latest frame dict from the server.
 */

import React from 'react';

function Stat({ label, value, color }) {
  return (
    <div style={styles.statRow}>
      <span style={styles.statLabel}>{label}</span>
      <span style={{ ...styles.statValue, color: color || '#e0e0e0' }}>{value}</span>
    </div>
  );
}

function Bar({ value, color }) {
  const pct = Math.max(0, Math.min(1, value ?? 0)) * 100;
  return (
    <div style={styles.barTrack}>
      <div style={{ ...styles.barFill, width: `${pct}%`, background: color }} />
    </div>
  );
}

function BattalionCard({ title, unit, color }) {
  if (!unit) return null;
  return (
    <div style={{ ...styles.card, borderColor: color }}>
      <div style={{ ...styles.cardTitle, color }}>{title}</div>
      <Stat label="Strength" value={`${(unit.strength * 100).toFixed(1)}%`} color={color} />
      <Bar value={unit.strength} color={color} />
      <Stat label="Morale" value={unit.morale != null ? `${(unit.morale * 100).toFixed(1)}%` : '—'} />
      {unit.morale != null && <Bar value={unit.morale} color="#f0a500" />}
      <Stat label="Position" value={`(${unit.x?.toFixed(0)}, ${unit.y?.toFixed(0)})`} />
      <Stat label="Heading" value={`${((unit.theta ?? 0) * 180 / Math.PI).toFixed(1)}°`} />
      {unit.routed && <div style={styles.routedBadge}>ROUTED</div>}
    </div>
  );
}

export default function UnitPanel({ frame }) {
  if (!frame) {
    return <div style={styles.container}><div style={{ opacity: 0.5 }}>No active game</div></div>;
  }
  return (
    <div style={styles.container}>
      <BattalionCard title="🔵 Blue (You)" unit={frame.blue} color="#4a90e2" />
      <BattalionCard title="🔴 Red (AI)" unit={frame.red} color="#e94560" />
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  card: {
    background: '#16213e',
    border: '1px solid',
    borderRadius: 8,
    padding: '10px 12px',
  },
  cardTitle: {
    fontWeight: 700,
    fontSize: 14,
    marginBottom: 8,
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12,
    marginTop: 4,
  },
  statLabel: {
    opacity: 0.7,
  },
  statValue: {
    fontWeight: 600,
  },
  barTrack: {
    height: 5,
    background: '#0a1628',
    borderRadius: 3,
    marginTop: 3,
    marginBottom: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
    transition: 'width 0.2s ease',
  },
  routedBadge: {
    marginTop: 6,
    padding: '2px 6px',
    background: '#f0a500',
    color: '#1a1a2e',
    borderRadius: 4,
    fontWeight: 700,
    fontSize: 11,
    display: 'inline-block',
  },
};
