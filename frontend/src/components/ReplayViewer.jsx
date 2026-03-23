/**
 * ReplayViewer.jsx
 *
 * Step-through replay player with annotation support.
 *
 * Modes:
 *  "load_only"  — only shows the file-upload control (used in lobby).
 *  "playback"   — shows full playback controls.
 *
 * Props:
 *   onLoad       {Function}  Called with parsed replay JSON when a file is loaded.
 *   onStep       {Function}  Advance replay by one frame.
 *   onSeek       {Function}  Jump to a specific frame index.
 *   replayFrame  {object|null}
 *   replayIndex  {number}
 *   replayTotal  {number}
 *   mode         {"load_only"|"playback"}
 */

import React, { useRef, useState } from 'react';

export default function ReplayViewer({
  onLoad,
  onStep,
  onSeek,
  replayFrame,
  replayIndex,
  replayTotal,
  mode = 'playback',
}) {
  const fileRef = useRef(null);
  const [annotation, setAnnotation] = useState('');
  const [annotations, setAnnotations] = useState([]);
  const [playing, setPlaying] = useState(false);
  const playIntervalRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        onLoad(data);
      } catch {
        alert('Invalid replay file (must be JSON).');
      }
    };
    reader.readAsText(file);
  };

  const handlePlay = () => {
    if (playing) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
      setPlaying(false);
    } else {
      setPlaying(true);
      playIntervalRef.current = setInterval(() => {
        onStep();
      }, 150);
    }
  };

  // Clear the play timer on unmount to prevent leaks.
  useEffect(() => {
    return () => {
      if (playIntervalRef.current != null) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, []);

  const handleSeek = (e) => {
    const idx = Number(e.target.value);
    onSeek(idx);
  };

  const addAnnotation = () => {
    if (!annotation.trim()) return;
    setAnnotations(prev => [...prev, { frame: replayIndex, text: annotation.trim() }]);
    setAnnotation('');
  };

  // Load-only mode (lobby).
  if (mode === 'load_only') {
    return (
      <div style={styles.container}>
        <div style={styles.title}>📼 Load Replay</div>
        <input
          ref={fileRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <button onClick={() => fileRef.current?.click()} style={styles.btn}>
          📂 Open Replay File…
        </button>
      </div>
    );
  }

  // Full playback mode.
  return (
    <div style={styles.container}>
      <div style={styles.title}>
        📼 Replay {replayTotal > 0 ? `(${replayIndex + 1} / ${replayTotal})` : ''}
      </div>

      {/* Seek slider */}
      {replayTotal > 0 && (
        <input
          type="range"
          min={0}
          max={replayTotal - 1}
          value={replayIndex}
          onChange={handleSeek}
          style={{ width: '100%', marginBottom: 8 }}
        />
      )}

      {/* Controls */}
      <div style={styles.controls}>
        <button onClick={() => onSeek(0)} style={styles.iconBtn} title="Rewind">⏮</button>
        <button onClick={() => onSeek(Math.max(0, replayIndex - 1))} style={styles.iconBtn} title="Step back">◀</button>
        <button onClick={handlePlay} style={{ ...styles.iconBtn, minWidth: 44 }} title={playing ? 'Pause' : 'Play'}>
          {playing ? '⏸' : '▶'}
        </button>
        <button onClick={onStep} style={styles.iconBtn} title="Step forward">▶|</button>
        <button onClick={() => onSeek(replayTotal - 1)} style={styles.iconBtn} title="Fast-forward">⏭</button>
      </div>

      {/* Annotation */}
      <div style={{ marginTop: 8 }}>
        <div style={styles.fieldLabel}>Add annotation at frame {replayIndex}</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <input
            value={annotation}
            onChange={e => setAnnotation(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addAnnotation()}
            placeholder="Note…"
            style={styles.input}
          />
          <button onClick={addAnnotation} style={styles.btn}>+</button>
        </div>
        {annotations.length > 0 && (
          <ul style={styles.annotList}>
            {annotations.map((a, i) => (
              <li key={i} style={styles.annotItem}>
                <span style={{ opacity: 0.5 }}>#{a.frame}</span> {a.text}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Load another file */}
      <input
        ref={fileRef}
        type="file"
        accept=".json"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <button onClick={() => fileRef.current?.click()} style={{ ...styles.btn, marginTop: 8 }}>
        📂 Load Another…
      </button>
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
  },
  controls: {
    display: 'flex',
    gap: 4,
    justifyContent: 'center',
  },
  iconBtn: {
    padding: '4px 8px',
    background: '#0f3460',
    color: '#e0e0e0',
    border: '1px solid #e94560',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 14,
  },
  btn: {
    padding: '5px 10px',
    background: '#e94560',
    color: '#fff',
    border: 'none',
    borderRadius: 5,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 600,
  },
  fieldLabel: {
    fontSize: 11,
    opacity: 0.6,
    marginBottom: 4,
  },
  input: {
    flex: 1,
    padding: '4px 8px',
    background: '#0a1628',
    color: '#e0e0e0',
    border: '1px solid #0f3460',
    borderRadius: 4,
    fontSize: 12,
  },
  annotList: {
    listStyle: 'none',
    marginTop: 6,
    maxHeight: 80,
    overflowY: 'auto',
  },
  annotItem: {
    fontSize: 11,
    padding: '2px 0',
    borderBottom: '1px solid rgba(255,255,255,0.05)',
  },
};
