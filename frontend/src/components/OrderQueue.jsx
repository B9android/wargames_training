/**
 * OrderQueue.jsx
 *
 * Displays queued orders before they are submitted to the server.
 * Orders are accumulated in App state and submitted in bulk.
 *
 * Props:
 *   orders    {object[]}   List of pending order objects.
 *   onSubmit  {Function}   Called when the user submits the queue.
 *   onClear   {Function}   Called when the user clears the queue.
 */

import React from 'react';

function formatOrder(order, idx) {
  const parts = [];
  if (order.move != null && order.move !== 0) parts.push(`Move ${order.move > 0 ? '▲' : '▼'}`);
  if (order.rotate != null && order.rotate !== 0) parts.push(`Turn ${order.rotate > 0 ? '↺' : '↻'}`);
  if (order.fire != null && order.fire > 0) parts.push('🔥 Fire');
  return `${idx + 1}. ${parts.join(' · ') || 'Idle'}`;
}

export default function OrderQueue({ orders, onSubmit, onClear }) {
  return (
    <div style={styles.container}>
      <div style={styles.title}>Order Queue ({orders.length})</div>
      {orders.length === 0 ? (
        <div style={styles.empty}>No pending orders</div>
      ) : (
        <ul style={styles.list}>
          {orders.map((o, i) => (
            <li key={i} style={styles.item}>{formatOrder(o, i)}</li>
          ))}
        </ul>
      )}
      <div style={styles.actions}>
        <button onClick={onSubmit} disabled={orders.length === 0} style={styles.btn}>
          ▶ Submit
        </button>
        <button onClick={onClear} disabled={orders.length === 0} style={styles.btnSecondary}>
          ✕ Clear
        </button>
      </div>
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
    marginBottom: 6,
    opacity: 0.8,
  },
  empty: {
    fontSize: 12,
    opacity: 0.4,
    fontStyle: 'italic',
    marginBottom: 8,
  },
  list: {
    listStyle: 'none',
    marginBottom: 8,
    maxHeight: 80,
    overflowY: 'auto',
  },
  item: {
    fontSize: 12,
    padding: '2px 0',
    borderBottom: '1px solid rgba(255,255,255,0.05)',
  },
  actions: {
    display: 'flex',
    gap: 6,
  },
  btn: {
    flex: 1,
    padding: '5px 0',
    background: '#e94560',
    color: '#fff',
    border: 'none',
    borderRadius: 5,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
  },
  btnSecondary: {
    flex: 1,
    padding: '5px 0',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 5,
    cursor: 'pointer',
    fontSize: 12,
  },
};
