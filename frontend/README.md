# Wargames Frontend (E9.1)

Interactive web-based wargame interface — React + Canvas 2D map renderer.

## Prerequisites

- Node.js ≥ 18
- The Python game server running on `ws://localhost:8765`

## Quick start

```bash
cd frontend
npm install
npm start        # dev server on http://localhost:3000
```

## Production build

```bash
npm run build    # output in frontend/dist/
```

## Architecture

```
Browser (React)
  │
  ├─ useGameSocket (WebSocket hook) ─── ws://localhost:8765
  │       │
  │       └─ server/game_server.py
  │               │
  │               ├─ envs/human_env.py (BattalionEnv)
  │               └─ http://localhost:8080 (ONNX policy server, optional)
  │
  └─ Components
       ├─ GameCanvas.jsx      (Canvas 2D map renderer)
       ├─ UnitPanel.jsx       (Blue / Red stats)
       ├─ FormationSelector   (LINE / COLUMN / SQUARE / SKIRMISH)
       ├─ OrderQueue.jsx      (pending orders)
       ├─ ScenarioEditor.jsx  (drag-and-drop placement, weather/terrain)
       └─ ReplayViewer.jsx    (step-through, annotation, JSON export/import)
```

## Environment variables

| Variable      | Default                  | Description                        |
|---------------|--------------------------|------------------------------------|
| `VITE_WS_URL` | `ws://localhost:8765`    | Game server WebSocket URL          |

## WebSocket message protocol

See [`server/game_server.py`](../server/game_server.py) for full protocol docs.

### Human → Server

| Message type    | Required fields                      | Description              |
|-----------------|--------------------------------------|--------------------------|
| `start`         | `scenario`, `difficulty`             | Begin new episode        |
| `action`        | `move`, `rotate`, `fire`             | Submit one step action   |
| `reset`         | —                                    | Restart current episode  |
| `load_scenario` | `scenario`, `difficulty`             | Load custom scenario     |
| `export_replay` | —                                    | Request replay download  |
| `load_replay`   | `replay`                             | Load replay for playback |
| `replay_step`   | —                                    | Advance replay one frame |
| `replay_seek`   | `index`                              | Jump to replay frame     |

### Server → Human

| Message type    | Description                                         |
|-----------------|-----------------------------------------------------|
| `frame`         | Current game-state frame (per sim step)             |
| `episode_end`   | Episode finished (`outcome`, `step`)                |
| `replay_export` | Full replay blob for download                       |
| `replay_loaded` | Replay loaded confirmation (`total` frame count)    |
| `replay_frame`  | One replay frame (`frame`, `index`, `total`)        |
| `error`         | Server-side error message                           |
