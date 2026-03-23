/**
 * ScenarioEditor.jsx
 *
 * Drag-and-drop scenario builder for E9.1.
 *
 * Features:
 *  - Choose built-in scenario as a starting point.
 *  - Adjust difficulty slider.
 *  - Click on the map preview to place Blue / Red starting positions.
 *  - Configure weather condition (Clear, Overcast, Rain, Fog, Snow).
 *  - Configure terrain type (Flat, Hills).
 *  - Launch the custom scenario.
 *
 * Props:
 *   onLaunch  {Function}  Called with the scenario config object.
 *   onCancel  {Function}  Called when the user cancels.
 */

import React, { useRef, useState, useEffect } from 'react';

const WEATHER_OPTIONS = ['CLEAR', 'OVERCAST', 'RAIN', 'FOG', 'SNOW'];
const MAP_W = 400;
const MAP_H = 400;

export default function ScenarioEditor({ onLaunch, onCancel }) {
  const [scenario, setScenario] = useState('open_field');
  const [difficulty, setDifficulty] = useState(5);
  const [weather, setWeather] = useState('CLEAR');
  const [randomizeTerrain, setRandomizeTerrain] = useState(false);
  const [bluePos, setBluePos] = useState({ x: 200, y: 200 });
  const [redPos, setRedPos] = useState({ x: 800, y: 800 });
  const [placing, setPlacing] = useState(null); // 'blue' | 'red' | null

  const canvasRef = useRef(null);

  // Draw map preview whenever positions change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, MAP_W, MAP_H);

    // Background.
    ctx.fillStyle = randomizeTerrain ? '#1a2e1a' : '#0a1628';
    ctx.fillRect(0, 0, MAP_W, MAP_H);

    // Grid.
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    for (let x = 0; x <= MAP_W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAP_H); ctx.stroke(); }
    for (let y = 0; y <= MAP_H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(MAP_W, y); ctx.stroke(); }

    // Units (world-space 0–1000 → canvas 0–400).
    const toCanvas = (wx, wy) => ({ cx: (wx / 1000) * MAP_W, cy: (1 - wy / 1000) * MAP_H });

    const b = toCanvas(bluePos.x, bluePos.y);
    ctx.fillStyle = '#4a90e2';
    ctx.beginPath(); ctx.arc(b.cx, b.cy, 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('B', b.cx, b.cy);

    const r = toCanvas(redPos.x, redPos.y);
    ctx.fillStyle = '#e94560';
    ctx.beginPath(); ctx.arc(r.cx, r.cy, 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText('R', r.cx, r.cy);

    // Placing hint.
    if (placing) {
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(`Click to place ${placing === 'blue' ? 'Blue' : 'Red'}`, MAP_W / 2, 8);
    }
  }, [bluePos, redPos, randomizeTerrain, placing]);

  const handleCanvasClick = (e) => {
    if (!placing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    // Convert canvas coords → world coords.
    const wx = (cx / MAP_W) * 1000;
    const wy = (1 - cy / MAP_H) * 1000;
    const pos = { x: Math.round(wx), y: Math.round(wy) };
    if (placing === 'blue') setBluePos(pos);
    else setRedPos(pos);
    setPlacing(null);
  };

  const handleLaunch = () => {
    onLaunch({
      scenario,
      difficulty,
      weather_condition: weather,
      randomize_terrain: randomizeTerrain,
      blue_pos: bluePos,
      red_pos: redPos,
    });
  };

  return (
    <div style={styles.root}>
      <h2 style={styles.heading}>🗺 Scenario Editor</h2>
      <div style={styles.layout}>
        {/* Map preview */}
        <div>
          <canvas
            ref={canvasRef}
            width={MAP_W}
            height={MAP_H}
            onClick={handleCanvasClick}
            style={{
              border: '1px solid #0f3460',
              borderRadius: 8,
              cursor: placing ? 'crosshair' : 'default',
            }}
          />
          <div style={styles.placeBtns}>
            <button
              onClick={() => setPlacing(placing === 'blue' ? null : 'blue')}
              style={{ ...styles.btn, background: placing === 'blue' ? '#4a90e2' : '#0f3460' }}
            >
              📍 Place Blue
            </button>
            <button
              onClick={() => setPlacing(placing === 'red' ? null : 'red')}
              style={{ ...styles.btn, background: placing === 'red' ? '#e94560' : '#0f3460' }}
            >
              📍 Place Red
            </button>
          </div>
          <div style={{ fontSize: 11, opacity: 0.5, marginTop: 4 }}>
            Blue: ({bluePos.x}, {bluePos.y}) · Red: ({redPos.x}, {redPos.y})
          </div>
        </div>

        {/* Config panel */}
        <div style={styles.config}>
          <label style={styles.fieldLabel}>Base Scenario</label>
          <select value={scenario} onChange={e => setScenario(e.target.value)} style={styles.select}>
            <option value="open_field">Open Field</option>
            <option value="mountain_pass">Mountain Pass</option>
            <option value="last_stand">Last Stand</option>
          </select>

          <label style={styles.fieldLabel}>Difficulty: {difficulty}</label>
          <input
            type="range" min={1} max={5} value={difficulty}
            onChange={e => setDifficulty(Number(e.target.value))}
            style={{ width: '100%' }}
          />

          <label style={styles.fieldLabel}>Weather</label>
          <select value={weather} onChange={e => setWeather(e.target.value)} style={styles.select}>
            {WEATHER_OPTIONS.map(w => <option key={w} value={w}>{w}</option>)}
          </select>

          <label style={styles.checkRow}>
            <input type="checkbox" checked={randomizeTerrain} onChange={e => setRandomizeTerrain(e.target.checked)} />
            <span style={{ marginLeft: 6 }}>Randomize terrain</span>
          </label>

          <div style={{ display: 'flex', gap: 10, marginTop: 24 }}>
            <button onClick={handleLaunch} style={styles.launchBtn}>▶ Launch</button>
            <button onClick={onCancel} style={styles.cancelBtn}>✕ Cancel</button>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  root: {
    padding: 24,
    maxWidth: 880,
    margin: '0 auto',
  },
  heading: {
    marginBottom: 20,
    color: '#e94560',
  },
  layout: {
    display: 'flex',
    gap: 32,
    flexWrap: 'wrap',
  },
  config: {
    flex: 1,
    minWidth: 220,
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  fieldLabel: {
    fontSize: 13,
    fontWeight: 600,
    opacity: 0.8,
    marginTop: 8,
  },
  select: {
    padding: '6px 10px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 6,
    width: '100%',
  },
  checkRow: {
    display: 'flex',
    alignItems: 'center',
    fontSize: 13,
    marginTop: 8,
    cursor: 'pointer',
  },
  placeBtns: {
    display: 'flex',
    gap: 8,
    marginTop: 8,
  },
  btn: {
    flex: 1,
    padding: '6px 0',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 12,
  },
  launchBtn: {
    flex: 1,
    padding: '10px 0',
    background: '#e94560',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 700,
    fontSize: 14,
  },
  cancelBtn: {
    flex: 1,
    padding: '10px 0',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 14,
  },
};
