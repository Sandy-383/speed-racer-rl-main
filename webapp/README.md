# Speed Racer RL — Python Web App

An interactive browser-based visualizer for the Speed Racer RL project.
Watch a trained Deep Q-Network agent drive autonomously around a 2D racing track in real time, with live LIDAR rays, Q-value charts, and lap telemetry — all in your browser.

---

## What It Does

The web app reimplements the original C++ replay system entirely in Python and streams it to a browser over WebSockets.

- Loads trained `.pt` model checkpoints from `sampleModels/`
- Runs the physics simulation and DQN inference on the server (Python + PyTorch)
- Sends frame data to the browser 30 times per second via Socket.IO
- Renders the track, car, LIDAR rays, checkpoints, and HUD on an HTML5 Canvas

---

## Project Structure

```
webapp/
├── app.py               # Flask + Socket.IO server, game session management
├── simulator.py         # Physics engine, LIDAR raycasting, checkpoint logic
├── model_loader.py      # PyTorch DQN model loading (multi-strategy)
├── convert_models.py    # One-shot utility to convert C++ .pt files
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Main UI (Jinja2 template)
└── static/
    ├── css/style.css    # Dark theme, glassmorphism UI
    └── js/game.js       # Canvas rendering, Socket.IO client
```

---

## Requirements

- Python 3.9 or newer
- pip

---

## Installation

```bash
cd webapp
pip install -r requirements.txt
```

---

## Running

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## If Models Fail to Load

The `.pt` files were saved by C++ LibTorch, which uses a different serialization format than Python PyTorch.
If the app shows a load error, run the conversion utility once:

```bash
python convert_models.py
```

This creates Python-compatible copies prefixed with `py_` in `sampleModels/`.
Restart the app and select one of the `py_` models from the dropdown.

---

## UI Overview

| Area | Description |
|---|---|
| **Canvas** | Live 900×900 race track with car, LIDAR rays, and checkpoints |
| **Speed Gauge** | Arc gauge showing current speed (0–300 units/s) |
| **Race Info** | Current lap, lap time, best lap time |
| **Q-Values** | Live bar chart of the neural network's 7 output values |
| **Lap Times** | Per-lap history with best lap highlighted in gold |
| **Model Selector** | Switch between trained checkpoints |

---

## Controls

| Button | Action |
|---|---|
| **Start Race** | Load selected model and begin simulation |
| **Restart** | Reset car to start position |
| **LIDAR: ON/OFF** | Toggle LIDAR ray visualization |

---

## LIDAR Visualization

The agent perceives the track through 18 raycasts cast from the car:

- **Orange rays (13)** — Short range (200 px), span ±90°. Return a danger value `= 1 / ((dist/50) + 0.1)`, clamped to [0, 1]. Used for immediate wall avoidance.
- **Cyan rays (5)** — Long range (900 px), span ±30°. Return normalized distance `dist / 900`. Used for anticipating upcoming corners.

---

## Sample Models

| Model | Training Stage | Expected Behavior |
|---|---|---|
| `model_episode_50.pt` | Very early | Mostly random, frequent crashes |
| `model_episode_100.pt` | Early | Starts following the track loosely |
| `model_episode_200.pt` | Mid | Occasional lap completions |
| `model_episode_500.pt` | Intermediate | Consistent laps, higher speed attempts |
| `model_episode_1000.pt` | Trained | Smooth, reliable multi-lap racing |
| `best_time.pt` | Speed-optimized | Fastest recorded finish, aggressive lines |

**Demo tip:** Show `episode_50` first, then switch to `best_time` to demonstrate how far the agent has learned.

---

## How It Works

```
Browser                          Server (Python)
  │                                    │
  │── socket.emit('start', model) ────▶│  Load .pt model (PyTorch)
  │                                    │  Start 30 FPS game loop thread
  │                                    │
  │                                    │  Every frame:
  │                                    │   1. Build 23-dim state vector
  │                                    │   2. DQN forward pass → 7 Q-values
  │                                    │   3. Select argmax action
  │                                    │   4. Step physics (speed, angle, position)
  │                                    │   5. Cast 18 LIDAR rays
  │                                    │   6. Check checkpoint crossings
  │                                    │
  │◀── socket.on('frame', data) ──────│  Emit frame data (position, angle,
  │                                    │   speed, LIDAR hits, Q-values, lap info)
  │
  │  Canvas renders:
  │   - Track image
  │   - Checkpoint lines (color-coded)
  │   - LIDAR rays with glow effect
  │   - Car sprite (rotated to heading)
  │   - Speed gauge + Q-value bars
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `flask-socketio` | WebSocket communication |
| `torch` | DQN model inference |
| `pillow` | Track image pixel access for LIDAR |
| `numpy` | Fast pixel array operations |
