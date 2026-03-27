/* game.js – Speed Racer RL Web Visualizer */
const socket = io();

// ── DOM refs ──────────────────────────────────────────────────
const canvas       = document.getElementById('track');
const ctx          = canvas.getContext('2d');
const overlay      = document.getElementById('overlay');
const finishOvl    = document.getElementById('finish-overlay');
const btnStart     = document.getElementById('btn-start');
const btnRestart   = document.getElementById('btn-restart');
const btnLidar     = document.getElementById('btn-lidar');
const modelSel     = document.getElementById('model-sel');
const statusDot    = document.getElementById('status-dot');
const statusLabel  = document.getElementById('status-label');
const speedReadout = document.getElementById('speed-readout');
const gaugeCanvas  = document.getElementById('gauge-canvas');
const gCtx         = gaugeCanvas.getContext('2d');
const errorToast   = document.getElementById('error-toast');

// info cells
const elLap     = document.getElementById('info-lap');
const elTime    = document.getElementById('info-time');
const elBest    = document.getElementById('info-best');
const elSteps   = document.getElementById('info-steps');
const elWalls   = document.getElementById('info-walls');
const elAction  = document.getElementById('info-action');
const lapList   = document.getElementById('lap-list');

// ── Assets ───────────────────────────────────────────────────
const trackImg = new Image();
const carImg   = new Image();
let assetsReady = 0;
trackImg.onload = carImg.onload = () => { assetsReady++; if (assetsReady === 2) drawIdle(); };
trackImg.src = '/assets/raceTrackFullyWalled.png';
carImg.src   = '/assets/racecarTransparent.png';

// ── Constants ─────────────────────────────────────────────────
const NATIVE = 900;   // internal canvas resolution
const CHECKPOINTS = [
  [[450,35],[450,150]],
  [[719,260],[850,260]],
  [[850,665],[723,665]],
  [[523,482],[625,517]],
  [[409,438],[295,413]],
  [[160,730],[220,815]],
  [[138,600],[49,600]],
  [[138,205],[49,205]],
];
const Q_LABELS = ['↑ Fwd','↓ Rev','← Left','→ Right','↖ F+L','↗ F+R','○ Coast'];
const ACTION_NAMES = ['FORWARD','REVERSE','TURN LEFT','TURN RIGHT','FWD+LEFT','FWD+RIGHT','COAST'];
const CAR_SCALE = 0.15;

let gameState = null;
let lidarOn   = true;

// ── Scale helper (canvas CSS vs native) ──────────────────────
function scale() { return canvas.width / NATIVE; } // always 1 since we CSS-scale the element

// ── Gauge ─────────────────────────────────────────────────────
function drawGauge(speed) {
  const W = gaugeCanvas.width, H = gaugeCanvas.height;
  const cx = W / 2, cy = H * 0.62;
  const r  = W * 0.38;
  const startA = Math.PI * 0.75, endA = Math.PI * 2.25;
  const pct = Math.min(1, speed / 300);
  gCtx.clearRect(0, 0, W, H);

  // Background arc
  gCtx.beginPath();
  gCtx.arc(cx, cy, r, startA, endA);
  gCtx.strokeStyle = 'rgba(255,255,255,0.07)';
  gCtx.lineWidth = 10;
  gCtx.lineCap = 'round';
  gCtx.stroke();

  // Value arc
  if (pct > 0) {
    const grad = gCtx.createLinearGradient(cx - r, cy, cx + r, cy);
    grad.addColorStop(0,   '#00d4ff');
    grad.addColorStop(0.6, '#ff6b35');
    grad.addColorStop(1,   '#ff2244');
    gCtx.beginPath();
    gCtx.arc(cx, cy, r, startA, startA + (endA - startA) * pct);
    gCtx.strokeStyle = grad;
    gCtx.lineWidth = 10;
    gCtx.lineCap = 'round';
    gCtx.shadowBlur = 14;
    gCtx.shadowColor = pct > 0.7 ? '#ff6b35' : '#00d4ff';
    gCtx.stroke();
    gCtx.shadowBlur = 0;
  }

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const a = startA + (endA - startA) * (i / 10);
    const inner = r - (i % 5 === 0 ? 16 : 10);
    gCtx.beginPath();
    gCtx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner);
    gCtx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
    gCtx.strokeStyle = i % 5 === 0 ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)';
    gCtx.lineWidth = i % 5 === 0 ? 2 : 1;
    gCtx.stroke();
  }
}

// ── Q-Values bars ──────────────────────────────────────────────
function updateQBars(qVals, action) {
  const min = Math.min(...qVals), max = Math.max(...qVals);
  const range = max - min || 1;
  document.querySelectorAll('.q-row').forEach((row, i) => {
    const fill = row.querySelector('.q-bar-fill');
    const val  = row.querySelector('.q-val');
    const pct  = ((qVals[i] - min) / range * 100).toFixed(0);
    fill.style.width = pct + '%';
    fill.classList.toggle('active', i === action);
    val.textContent = qVals[i].toFixed(2);
  });
}

// ── Lap list ──────────────────────────────────────────────────
function updateLapList(lapTimes, bestLap) {
  if (lapTimes.length === 0) {
    lapList.innerHTML = '<div class="lap-placeholder">No laps yet</div>';
    return;
  }
  lapList.innerHTML = lapTimes.map((t, i) => {
    const isBest = bestLap !== null && Math.abs(t - bestLap) < 0.01;
    return `<div class="lap-entry${isBest ? ' best' : ''}">
      <span class="lap-num">LAP ${i + 1}</span>
      <span class="lap-t">${t.toFixed(2)}s${isBest ? ' ★' : ''}</span>
    </div>`;
  }).join('');
}

// ── Canvas drawing ─────────────────────────────────────────────
function drawIdle() {
  ctx.clearRect(0, 0, NATIVE, NATIVE);
  if (assetsReady >= 1) ctx.drawImage(trackImg, 0, 0, NATIVE, NATIVE);
  // dim it
  ctx.fillStyle = 'rgba(8,11,18,0.55)';
  ctx.fillRect(0, 0, NATIVE, NATIVE);
  drawGauge(0);
  speedReadout.innerHTML = '0<span>UNITS/S</span>';
}

function drawFrame(d) {
  // Track
  ctx.clearRect(0, 0, NATIVE, NATIVE);
  ctx.drawImage(trackImg, 0, 0, NATIVE, NATIVE);

  // Checkpoints
  CHECKPOINTS.forEach(([[x1, y1], [x2, y2]], i) => {
    let color;
    if (i === 0)              color = '#ff2244';       // finish line (red)
    if (d.cp_crossed[i])      color = '#39ff14';       // crossed (green)
    if (i === d.next_cp)      color = '#00d4ff';       // next target (cyan)
    if (i === 0 && !d.cp_crossed[0] && d.next_cp !== 0) color = '#ff2244';

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color || '#ffdd44';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Label
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    ctx.fillStyle = color || '#ffdd44';
    ctx.font = 'bold 13px Inter';
    ctx.fillText(i === 0 ? 'F' : i, mx - 5, my - 6);
  });

  // LIDAR rays
  if (d.show_lidar) {
    // Short range (orange)
    ctx.save();
    ctx.globalAlpha = 0.5;
    d.short_hits.forEach(h => {
      ctx.beginPath();
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(h.x, h.y);
      ctx.strokeStyle = '#ff6b35';
      ctx.lineWidth = 1;
      ctx.shadowBlur = 5;
      ctx.shadowColor = '#ff6b35';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(h.x, h.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ff6b35';
      ctx.fill();
    });
    ctx.globalAlpha = 0.35;
    // Long range (cyan)
    d.long_hits.forEach(h => {
      ctx.beginPath();
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(h.x, h.y);
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1;
      ctx.shadowBlur = 5;
      ctx.shadowColor = '#00d4ff';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(h.x, h.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#00d4ff';
      ctx.fill();
    });
    ctx.restore();
  }

  // Car
  if (assetsReady >= 2 && carImg.complete) {
    const cw = carImg.naturalWidth  * CAR_SCALE;
    const ch = carImg.naturalHeight * CAR_SCALE;
    ctx.save();
    ctx.translate(d.x, d.y);
    ctx.rotate(d.angle);
    ctx.drawImage(carImg, -cw / 2, -ch / 2, cw, ch);
    ctx.restore();
  } else {
    // Fallback: colored rectangle
    ctx.save();
    ctx.translate(d.x, d.y);
    ctx.rotate(d.angle);
    ctx.fillStyle = '#00d4ff';
    ctx.shadowBlur = 12;
    ctx.shadowColor = '#00d4ff';
    ctx.fillRect(-10, -6, 20, 12);
    ctx.restore();
  }

  // HUD – current action
  ctx.save();
  ctx.fillStyle = 'rgba(8,11,18,0.7)';
  ctx.beginPath();
  ctx.roundRect(10, 10, 160, 32, 6);
  ctx.fill();
  ctx.fillStyle = '#00d4ff';
  ctx.font = 'bold 12px Orbitron, monospace';
  ctx.fillText('AI: ' + ACTION_NAMES[d.action], 20, 31);
  ctx.restore();
}

// ── UI updates ────────────────────────────────────────────────
function updateUI(d) {
  drawGauge(d.speed);
  speedReadout.innerHTML = `${Math.round(d.speed)}<span>UNITS/S</span>`;
  elLap.textContent    = `${d.lap} / 3`;
  elTime.textContent   = d.lap_time.toFixed(2) + 's';
  elBest.textContent   = d.best_lap !== null ? d.best_lap.toFixed(2) + 's' : '—';
  elSteps.textContent  = d.steps;
  elWalls.textContent  = d.wall_hits;
  elAction.textContent = ACTION_NAMES[d.action];
  updateQBars(d.q_vals, d.action);
  updateLapList(d.lap_times, d.best_lap);
}

// ── Socket events ─────────────────────────────────────────────
socket.on('connect', () => {
  statusDot.classList.add('live');
  statusLabel.textContent = 'Connected';
});

socket.on('disconnect', () => {
  statusDot.classList.remove('live');
  statusLabel.textContent = 'Disconnected';
  btnStart.disabled   = false;
  btnRestart.disabled = true;
});

socket.on('ready', () => {
  overlay.classList.add('hidden');
  finishOvl.classList.remove('show');
  btnStart.disabled   = false;
  btnRestart.disabled = false;
  statusLabel.textContent = 'Racing…';
});

socket.on('frame', d => {
  gameState = d;
  drawFrame(d);
  updateUI(d);

  if (d.finished && !finishOvl.classList.contains('show')) {
    finishOvl.classList.add('show');
    document.getElementById('finish-total').textContent =
      d.lap_times.reduce((a, b) => a + b, 0).toFixed(2) + 's';
    const lapsEl = document.getElementById('finish-laps');
    lapsEl.innerHTML = d.lap_times.map((t, i) =>
      `<div class="lap-chip"><span>LAP ${i+1}</span><strong>${t.toFixed(2)}s</strong></div>`
    ).join('');
    statusLabel.textContent = 'Finished!';
  }
});

socket.on('load_error', data => {
  showError('Model load failed: ' + data.msg + ' — Try running convert_models.py first.');
  btnStart.disabled = false;
  statusLabel.textContent = 'Error';
});

// ── Controls ──────────────────────────────────────────────────
btnStart.addEventListener('click', () => {
  const model = modelSel.value;
  if (!model) { showError('Please select a model first.'); return; }
  btnStart.disabled = true;
  statusLabel.textContent = 'Loading…';
  finishOvl.classList.remove('show');
  socket.emit('start', { model });
});

btnRestart.addEventListener('click', () => {
  finishOvl.classList.remove('show');
  socket.emit('restart');
  statusLabel.textContent = 'Racing…';
});

btnLidar.addEventListener('click', () => {
  lidarOn = !lidarOn;
  btnLidar.textContent = lidarOn ? 'LIDAR: ON' : 'LIDAR: OFF';
  btnLidar.style.borderColor = lidarOn ? 'var(--cyan)' : '';
  btnLidar.style.color       = lidarOn ? 'var(--cyan)' : '';
  socket.emit('toggle_lidar');
});

// ── Error toast ───────────────────────────────────────────────
function showError(msg) {
  errorToast.textContent = msg;
  errorToast.classList.add('show');
  setTimeout(() => errorToast.classList.remove('show'), 5000);
}
