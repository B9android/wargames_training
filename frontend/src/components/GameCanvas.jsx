/**
 * GameCanvas.jsx
 *
 * WebGL-accelerated (Canvas 2D fallback) map renderer.
 *
 * Renders:
 *  - Terrain elevation grid (heat-map shading)
 *  - Blue battalion (blue rectangle + direction arrow)
 *  - Red battalion  (red rectangle + direction arrow)
 *  - Strength / morale bars
 *  - Step counter overlay
 *
 * Props:
 *   frame  {object|null}  Frame dict from the server, or null (shows placeholder).
 */

import React, { useEffect, useRef } from 'react';

const CANVAS_W = 640;
const CANVAS_H = 640;

/** Draw one battalion rectangle + heading arrow on a Canvas 2D context. */
function drawBattalion(ctx, unit, mapW, mapH, color) {
  if (!unit || unit.strength <= 0) return;

  const px = (unit.x / mapW) * CANVAS_W;
  const py = (1 - unit.y / mapH) * CANVAS_H;  // Y is flipped (screen coords)
  const theta = unit.theta ?? 0;

  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-theta);  // negate because canvas Y is flipped

  // Battalion rectangle (represents a 1D line segment).
  const rectW = 40;
  const rectH = 12;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.fillRect(-rectW / 2, -rectH / 2, rectW, rectH);

  // Direction arrow.
  ctx.globalAlpha = 1.0;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(rectW / 2 + 8, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(rectW / 2 + 8, 0);
  ctx.lineTo(rectW / 2 + 2, -4);
  ctx.moveTo(rectW / 2 + 8, 0);
  ctx.lineTo(rectW / 2 + 2, 4);
  ctx.stroke();

  ctx.restore();

  // Strength bar (below unit).
  const barW = 44;
  const barH = 5;
  const bx = px - barW / 2;
  const by = py + 10;
  ctx.fillStyle = '#333';
  ctx.fillRect(bx, by, barW, barH);
  ctx.fillStyle = color;
  ctx.fillRect(bx, by, barW * Math.max(0, unit.strength), barH);
}

/** Draw terrain elevation heat-map from a 2-D grid. */
function drawTerrain(ctx, grid) {
  if (!grid || grid.length === 0) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = CANVAS_W / cols;
  const cellH = CANVAS_H / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = grid[r][c];                   // 0–1 normalised elevation
      const g = Math.round(30 + v * 60);      // green channel: 30–90
      const b = Math.round(20 + v * 30);      // blue channel: 20–50
      ctx.fillStyle = `rgb(20,${g},${b})`;
      ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

export default function GameCanvas({ frame }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    if (!frame) {
      // Placeholder.
      ctx.fillStyle = '#0a1628';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.fillStyle = '#e0e0e0';
      ctx.font = '18px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting game start…', CANVAS_W / 2, CANVAS_H / 2);
      return;
    }

    const mapW = frame.map?.width ?? 1000;
    const mapH = frame.map?.height ?? 1000;

    // Background.
    ctx.fillStyle = '#0a1628';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Terrain.
    drawTerrain(ctx, frame.terrain_summary);

    // Grid lines (subtle).
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= CANVAS_W; x += CANVAS_W / 10) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CANVAS_H); ctx.stroke();
    }
    for (let y = 0; y <= CANVAS_H; y += CANVAS_H / 10) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CANVAS_W, y); ctx.stroke();
    }

    // Units.
    drawBattalion(ctx, frame.blue, mapW, mapH, '#4a90e2');
    drawBattalion(ctx, frame.red, mapW, mapH, '#e94560');

    // Step counter.
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(4, 4, 90, 22);
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Step: ${frame.step ?? 0}`, 10, 19);

    // Routed indicators.
    if (frame.blue?.routed) {
      ctx.fillStyle = 'rgba(255,255,0,0.25)';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText('ROUTED', 10, 40);
    }
  }, [frame]);

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      style={{ width: '100%', height: 'auto', borderRadius: 8, border: '1px solid #0f3460' }}
    />
  );
}
