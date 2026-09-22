"""Standalone smooth-pursuit ENGAGEMENT prototype (Step 1/2).

Purpose: find out, with real numbers, whether we can tell that a student is
genuinely TRYING to follow a moving dot (intention, not accuracy) using only the
features FrontCamWorker already emits - NO new dependencies, NO per-user setup.

How it works (Pursuits, Vidal/Bulling/Gellersen UbiComp 2013):
  - One dot ORBITS the screen center in a LARGE ellipse, driven by wall-clock so
    it is always moving. Reference signal = the orbit offset (cos, sin).
  - Every frame we buffer the last CORR_WIN samples of the orbit offset and of
    the head pose.
  - Engagement per axis = SIGNED Pearson corr of head vs. the orbit offset:
        ex = corr(yaw,   dx)     (head must move the SAME way as the dot)
        ey = corr(pitch, dy)
    Positive is required (moving WITH the dot); correlation is scale/offset
    invariant, so no calibration and staring can't fake it.
  - ANTI-FLUKE guard: the axis' head signal must actually have MOVED (std over
    the window > its MOVE_MIN); a flat/frozen signal scores 0.
  - SUSTAIN accumulator: +1 per good frame (ex>=R and ey>=R), -1 per bad frame,
    floored at 0. Confirm at CONFIRM_TARGET. A short lucky burst can't reach it;
    genuine following sails past it.

Tuned from two data runs:
  - X and Y both come from HEAD pose (yaw, pitch). The eye signals (h_ratio,
    v_openness) are weak/noisy and gave random motion extra ways to false-fire,
    so they are dropped from scoring (still shown for reference).
  - Requiring the correct SIGN + a sustained accumulator rejects both a frozen
    stare and vigorous random head-sweeping ("looking away"), while genuine
    head-following confirms in a few seconds.

Run from integration/app:
    ../.venv/Scripts/python.exe pursuit_test.py            (camera 0)
    ../.venv/Scripts/python.exe pursuit_test.py 1          (camera index 1)

Everything is appended to pursuit_log.txt so the numbers don't scroll away.
Press 1/2/3 to mark FOLLOW / STARE / LOOK-AWAY trials.  Esc quits.
"""
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget

from workers.front_cam_worker import FrontCamWorker

LOG_PATH = Path(__file__).with_name('pursuit_log.txt')
TRIALS = {Qt.Key_1: 'FOLLOW', Qt.Key_2: 'STARE', Qt.Key_3: 'LOOK-AWAY'}

# --- Tuning knobs (edit + rerun; the HUD prints live values) ---
ORBIT_R = 0.25         # orbit radius as a fraction of screen (LARGE motion)
ORBIT_SEC = 3.5        # seconds per revolution (raise if too fast to follow)
CORR_WIN = 28          # samples in the correlation window (~2 s at ~14 fps)
FOLLOW_R = 0.50        # per-axis POSITIVE corr threshold (intention, not accuracy)
CONFIRM_TARGET = 16    # sustain accumulator target (+1 good / -1 bad, floor 0)
GAP_SEC = 0.5          # clear buffers if features stop for this long (face lost)

# Anti-fluke minimum std per HEAD axis, in degrees: the head must actually have
# moved this much over the window or that axis scores 0.
MOVE_MIN = {'yaw': 0.8, 'pitch': 1.6}


def corr(xs, ys):
    """SIGNED Pearson corr, guarded against too-few samples / zero variance."""
    if len(xs) < 8:
        return 0.0
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class PursuitTest(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle('Pursuit engagement test')
        self.frame = None
        self.t0 = time.time()
        self.last_feat = 0.0
        self.progress = 0.0          # sustain accumulator
        self.confirms = 0
        self._at_target = False      # edge flag so one hold = one confirm
        self._last_print = 0.0
        self.trial = '-'
        self._log = open(LOG_PATH, 'w', buffering=1)
        self._log.write(f'# pursuit_log  {datetime.now():%Y-%m-%d %H:%M:%S}  '
                        f'ORBIT_R={ORBIT_R} ORBIT_SEC={ORBIT_SEC} CORR_WIN={CORR_WIN} '
                        f'FOLLOW_R={FOLLOW_R} TARGET={CONFIRM_TARGET} MOVE_MIN={MOVE_MIN}\n')

        # Rolling buffers: reference offset + head pose (+ eye signals for display).
        self.dx = deque(maxlen=CORR_WIN)
        self.dy = deque(maxlen=CORR_WIN)
        self.sig = {k: deque(maxlen=CORR_WIN) for k in
                    ('yaw', 'pitch', 'h_ratio', 'v_openness')}
        self.hud = {}

        self.worker = FrontCamWorker(camera_index=camera_index, detect=False)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.features_ready.connect(self._on_features)
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)   # 60 fps repaint
        self._timer.start(16)

    def _offset(self):
        ang = 2 * math.pi * (time.time() - self.t0) / ORBIT_SEC
        return math.cos(ang), math.sin(ang)

    def _clear(self):
        self.dx.clear(); self.dy.clear()
        for d in self.sig.values():
            d.clear()
        self.progress = 0.0
        self._at_target = False

    def _on_frame(self, qimg):
        self.frame = qimg

    def _on_features(self, feats):
        now = time.time()
        if now - self.last_feat > GAP_SEC:     # face lost: window is stale
            self._clear()
        self.last_feat = now

        ox, oy = self._offset()
        self.dx.append(ox)
        self.dy.append(oy)
        for k in self.sig:
            self.sig[k].append(feats[k])

        self._score()

    def _score(self):
        sd = {k: (np.std(self.sig[k]) if self.sig[k] else 0.0) for k in self.sig}
        cyaw = corr(self.sig['yaw'], self.dx)      # signed
        cpit = corr(self.sig['pitch'], self.dy)    # signed

        # Head must move WITH the dot (positive) AND have actually moved.
        ex = cyaw if sd['yaw'] > MOVE_MIN['yaw'] else 0.0
        ey = cpit if sd['pitch'] > MOVE_MIN['pitch'] else 0.0
        good = ex >= FOLLOW_R and ey >= FOLLOW_R

        self.progress = min(CONFIRM_TARGET, max(0.0, self.progress + (1 if good else -1)))
        if self.progress >= CONFIRM_TARGET:
            if not self._at_target:
                self._at_target = True
                self.confirms += 1
        else:
            self._at_target = False

        # Eye signals kept only for the reference readout.
        chr_ = corr(self.sig['h_ratio'], self.dx)
        cvo = corr(self.sig['v_openness'], self.dy)
        self.hud = dict(ex=ex, ey=ey, sd=sd, chr=chr_, cvo=cvo)
        self._maybe_print()

    def _maybe_print(self):
        now = time.time()
        if now - self._last_print < 0.3:
            return
        self._last_print = now
        h = self.hud
        line = (
            f'[{self.trial:9}] ex={h["ex"]:+.2f} ey={h["ey"]:+.2f} '
            f'prog={self.progress:.0f}/{CONFIRM_TARGET} conf={self.confirms} | '
            f'yaw sd={h["sd"]["yaw"]:.2f}  pit sd={h["sd"]["pitch"]:.2f}  '
            f'[ref hr c={h["chr"]:+.2f} sd={h["sd"]["h_ratio"]:.3f}  '
            f'vo c={h["cvo"]:+.2f} sd={h["sd"]["v_openness"]:.3f}]')
        print(line, flush=True)
        self._log.write(line + '\n')

    # --- drawing ---
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        W, H = self.width(), self.height()

        if self.frame is not None:
            img = self.frame.scaled(self.size(), Qt.KeepAspectRatio)
            fx, fy = (W - img.width()) // 2, (H - img.height()) // 2
            p.setOpacity(0.25)
            p.drawImage(fx, fy, img)
            p.setOpacity(1.0)

        ox, oy = self._offset()
        cx = int((0.5 + ORBIT_R * ox) * W)
        cy = int((0.5 + ORBIT_R * oy) * H)
        live = time.time() - self.last_feat < GAP_SEC
        confirmed = self.progress >= CONFIRM_TARGET
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#22c55e') if confirmed else QColor('#ffffff') if live else QColor('#ef4444'))
        p.drawEllipse(QPointF(cx, cy), 20, 20)

        # Progress ring fills with the sustain accumulator.
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor('#94a3b8'), 4))
        p.drawEllipse(QPointF(cx, cy), 34, 34)
        if self.progress:
            p.setPen(QPen(QColor('#22c55e'), 5))
            span = int(360 * self.progress / CONFIRM_TARGET)
            p.drawArc(QRectF(cx - 34, cy - 34, 68, 68), 90 * 16, -span * 16)

        self._draw_hud(p)
        p.end()

    def _draw_hud(self, p):
        h = self.hud
        if not h:
            return
        p.setFont(QFont('Consolas', 14))
        good = h['ex'] >= FOLLOW_R and h['ey'] >= FOLLOW_R
        lines = [
            f'ex(yaw)={h["ex"]:+.2f}   ey(pitch)={h["ey"]:+.2f}   (need >= {FOLLOW_R}, positive)',
            f'progress {self.progress:.0f}/{CONFIRM_TARGET}   confirms={self.confirms}   '
            f'{"GOOD" if good else ""}',
            '',
            f'head yaw   sd={h["sd"]["yaw"]:.2f}  (need > {MOVE_MIN["yaw"]})',
            f'head pitch sd={h["sd"]["pitch"]:.2f}  (need > {MOVE_MIN["pitch"]})',
            '',
            'reference only (not scored):',
            f'  h_ratio    c={h["chr"]:+.2f}  sd={h["sd"]["h_ratio"]:.3f}',
            f'  v_openness c={h["cvo"]:+.2f}  sd={h["sd"]["v_openness"]:.3f}',
            '',
            f'ORBIT_R={ORBIT_R}  ORBIT_SEC={ORBIT_SEC}  CORR_WIN={CORR_WIN}',
            f'trial: {self.trial}   (1=FOLLOW  2=STARE  3=LOOK-AWAY)',
            'follow the dot with your HEAD and eyes.  Esc to quit.',
        ]
        y = 40
        for ln in lines:
            p.setPen(QColor('#22c55e') if (good and ln.startswith('ex(')) else QColor('#e2e8f0'))
            p.drawText(20, y, ln)
            y += 26

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() in TRIALS:
            self.trial = TRIALS[event.key()]
            self._clear()                       # clean window for the new trial
            self._log.write(f'\n--- {self.trial} ---\n')

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait()
        self._log.close()
        super().closeEvent(event)


def main():
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = QApplication(sys.argv)
    win = PursuitTest(camera_index=cam)
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
