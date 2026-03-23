/**
 * FormationSelector.jsx
 *
 * Lets the human player choose a formation (LINE, COLUMN, SQUARE, SKIRMISH)
 * before queuing orders.
 *
 * Props:
 *   current   {string}    Currently selected formation key.
 *   onChange  {Function}  Called with the new formation key.
 */

import React from 'react';

const FORMATIONS = [
  { key: 'LINE',      label: 'Line',      icon: '═══', description: 'Max firepower, low mobility.' },
  { key: 'COLUMN',    label: 'Column',    icon: '↑↑↑', description: 'Fast movement, low fire.' },
  { key: 'SQUARE',    label: 'Square',    icon: '▣',   description: 'Cavalry defence bonus.' },
  { key: 'SKIRMISH',  label: 'Skirmish',  icon: '∷',   description: 'Extended screen, low cohesion.' },
];

export default function FormationSelector({ current, onChange }) {
  return (
    <div style={styles.container}>
      <div style={styles.title}>Formation</div>
      <div style={styles.grid}>
        {FORMATIONS.map(f => (
          <button
            key={f.key}
            onClick={() => onChange(f.key)}
            title={f.description}
            style={{
              ...styles.btn,
              ...(current === f.key ? styles.btnActive : {}),
            }}
          >
            <span style={styles.icon}>{f.icon}</span>
            <span style={styles.label}>{f.label}</span>
          </button>
        ))}
      </div>
      {current && (
        <div style={styles.desc}>
          {FORMATIONS.find(f => f.key === current)?.description}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    background: '#16213e',
    border: '1px solid #0f3460',
    borderRadius: 8,
    padding: '10px 12px',
  },
  title: {
    fontWeight: 700,
    fontSize: 13,
    marginBottom: 8,
    opacity: 0.8,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 6,
  },
  btn: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '6px 4px',
    background: '#0a1628',
    border: '1px solid #0f3460',
    borderRadius: 6,
    cursor: 'pointer',
    color: '#e0e0e0',
    gap: 2,
  },
  btnActive: {
    background: '#0f3460',
    border: '1px solid #e94560',
  },
  icon: {
    fontSize: 16,
  },
  label: {
    fontSize: 11,
  },
  desc: {
    marginTop: 6,
    fontSize: 11,
    opacity: 0.6,
    fontStyle: 'italic',
  },
};
