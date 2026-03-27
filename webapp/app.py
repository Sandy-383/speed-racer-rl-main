from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
import threading
import time
import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODELS_DIR = os.path.join(BASE_DIR, 'sampleModels')
TRACK_IMG  = os.path.join(ASSETS_DIR, 'raceTrackFullyWalled.png')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'speedracer-rl-2026'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

_sessions = {}


def _get_models():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, '*.pt')))
    return [{'name': os.path.basename(p), 'path': p} for p in paths]


@app.route('/')
def index():
    return render_template('index.html', models=_get_models())


@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_from_directory(ASSETS_DIR, filename)


class _Session:
    FPS = 30

    def __init__(self, sid, model_path):
        from simulator import RacingSimulator
        from model_loader import load_model

        self.sid = sid
        self.show_lidar = True
        self.running = False
        self._thread = None

        self.sim = RacingSimulator(TRACK_IMG)
        self.model, self.err = load_model(model_path)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        from model_loader import predict
        dt = 1.0 / self.FPS

        while self.running:
            t0 = time.perf_counter()

            action, q_vals = 6, [0.0] * 7
            if self.model and not self.sim.finished:
                state = self.sim.get_state()
                action, q_vals = predict(self.model, state)
                self.sim.step(action, dt)

            sh, lh = [], []
            if self.show_lidar:
                sh, lh = self.sim.get_lidar_hits()

            data = {
                'x':          round(self.sim.x, 1),
                'y':          round(self.sim.y, 1),
                'angle':      round(self.sim.angle, 4),
                'speed':      round(abs(self.sim.speed), 1),
                'lap':        max(0, self.sim.current_lap),
                'lap_time':   round(self.sim.lap_time, 2),
                'best_lap':   round(self.sim.best_lap, 2) if self.sim.best_lap < 99999 else None,
                'finished':   self.sim.finished,
                'next_cp':    self.sim.next_cp,
                'cp_crossed': self.sim.cp_crossed,
                'lap_times':  [round(t, 2) for t in self.sim.lap_times],
                'action':     action,
                'q_vals':     [round(q, 3) for q in q_vals],
                'steps':      self.sim.steps,
                'wall_hits':  self.sim.wall_hits,
                'show_lidar': self.show_lidar,
                'short_hits': sh,
                'long_hits':  lh,
            }
            socketio.emit('frame', data, room=self.sid)

            elapsed = time.perf_counter() - t0
            wait = dt - elapsed
            if wait > 0:
                time.sleep(wait)


@socketio.on('connect')
def on_connect():
    print(f'[+] Connected: {request.sid}')


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    if sid in _sessions:
        _sessions[sid].stop()
        del _sessions[sid]
    print(f'[-] Disconnected: {sid}')


@socketio.on('start')
def on_start(data):
    sid = request.sid
    if sid in _sessions:
        _sessions[sid].stop()

    sess = _Session(sid, data.get('model', ''))
    _sessions[sid] = sess

    if sess.model is None:
        emit('load_error', {'msg': sess.err or 'Unknown error loading model.'})
        return

    sess.start()
    emit('ready', {'model': data.get('model', '')})


@socketio.on('restart')
def on_restart():
    sid = request.sid
    if sid in _sessions:
        _sessions[sid].sim.reset()


@socketio.on('toggle_lidar')
def on_toggle_lidar():
    sid = request.sid
    if sid in _sessions:
        _sessions[sid].show_lidar = not _sessions[sid].show_lidar


if __name__ == '__main__':
    print('╔══════════════════════════════════════╗')
    print('║      Speed Racer RL  –  Web App      ║')
    print('╠══════════════════════════════════════╣')
    print('║  Open  http://localhost:5000          ║')
    print('╚══════════════════════════════════════╝')
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
