"""Calibration-FREE live gaze marker - hit-the-dot demo.

The professor wants a rough "you're looking about here" dot on screen with NO
calibration, and to advance the calibration by moving that marker onto each dot.
This is inherently imprecise: with no per-student calibration the app can only
GUESS a screen position from head pose (yaw/pitch) + iris position (h_ratio/
v_openness) with fixed gains, so the marker drifts and lags. To make it usable we
(1) auto-baseline for 1 s at start, (2) let you tune sensitivity live, and (3)
use a GENEROUS hit-radius with a short dwell so a rough marker can still land it.

Run from integration/app:
    ../.venv/Scripts/python.exe gaze_marker_test.py           (camera 0)
    ../.venv/Scripts/python.exe gaze_marker_test.py 1         (camera index 1)

Controls:
    Left / Right : horizontal sensitivity down / up
    Down / Up    : vertical sensitivity down / up
    B            : re-baseline (look at the centre, then press B)
    R            : restart the dot sequence
    Esc          : quit
"""
import sys
import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget

from workers.front_cam_worker import FrontCamWorker

# Dot sequence (centre, then corners, then edges). .1/.9 instead of .05/.95 so a
# rough marker can actually reach them.
DOTS = [(.5, .5), (.1, .1), (.9, .1), (.1, .9), (.9, .9),
        (.5, .1), (.1, .5), (.9, .5), (.5, .9)]

BASELINE_SEC = 1.0     # auto-baseline window at start (look at centre)
EMA_A = 0.25           # smoothing: lower = smoother/laggier, higher = jumpier
W_EYE = 150.0          # eye vs head blend (see notes in the previous version)
SENS_X = 0.025         # master horizontal sensitivity (tune live)
SENS_Y = 0.030         # master vertical sensitivity (tune live)

HIT_R = 0.12           # how close (fraction of screen) counts as "on the dot"
HIT_DWELL = 0.6        # seconds the marker must stay on the dot to advance


class GazeMarker(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle('Gaze marker - hit the dot (uncalibrated)')
        self.sens_x = SENS_X
        self.sens_y = SENS_Y
        self.frame = None
        self.gaze = None                 # smoothed (x, y) in 0..1, or None
        self._base = deque()
        self._base_until = time.time() + BASELINE_SEC
        self.ref = None                  # baseline (yaw0, pitch0, h0, v0)
        self.i = 0                        # current dot index
        self.dwell = 0.0                  # time held on the current dot
        self.done = False
        self._last = time.time()

        self.worker = FrontCamWorker(camera_index=camera_index, detect=False)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.features_ready.connect(self._on_features)
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _on_frame(self, qimg):
        self.frame = qimg

    def _on_features(self, feats):
        vals = (feats['yaw'], feats['pitch'], feats['h_ratio'], feats['v_openness'])
        if self.ref is None:
            self._base.append(vals)
            if time.time() >= self._base_until and self._base:
                n = len(self._base)
                self.ref = tuple(sum(v[i] for v in self._base) / n for i in range(4))
            return

        yaw0, pitch0, h0, v0 = self.ref
        raw_x = (feats['yaw'] - yaw0) + W_EYE * (feats['h_ratio'] - h0)
        raw_y = (feats['pitch'] - pitch0) - W_EYE * (feats['v_openness'] - v0)
        x = min(1.0, max(0.0, 0.5 + self.sens_x * raw_x))
        y = min(1.0, max(0.0, 0.5 + self.sens_y * raw_y))
        if self.gaze is None:
            self.gaze = (x, y)
        else:
            gx, gy = self.gaze
            self.gaze = (gx * (1 - EMA_A) + x * EMA_A, gy * (1 - EMA_A) + y * EMA_A)

    def _tick(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        # Advance when the marker sits on the current dot long enough.
        if not self.done and self.ref is not None and self.gaze is not None:
            tx, ty = DOTS[self.i]
            dist = ((self.gaze[0] - tx) ** 2 + (self.gaze[1] - ty) ** 2) ** 0.5
            if dist < HIT_R:
                self.dwell += dt
                if self.dwell >= HIT_DWELL:
                    self.i += 1
                    self.dwell = 0.0
                    if self.i >= len(DOTS):
                        self.done = True
            else:
                self.dwell = max(0.0, self.dwell - dt)   # decay when off the dot
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        W, H = self.width(), self.height()

        if self.frame is not None:
            img = self.frame.scaled(self.size(), Qt.KeepAspectRatio)
            fx, fy = (W - img.width()) // 2, (H - img.height()) // 2
            p.setOpacity(0.2)
            p.drawImage(fx, fy, img)
            p.setOpacity(1.0)

        if self.ref is None:
            p.setPen(QColor('#fbbf24'))
            p.setFont(QFont('Arial', 22))
            p.drawText(self.rect(), Qt.AlignCenter, 'Look at the centre...\n(baselining)')
            p.end()
            return

        if self.done:
            p.setPen(QColor('#22c55e'))
            p.setFont(QFont('Arial', 26, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignCenter, 'Done!\nR to restart, Esc to quit')
        else:
            # current target dot + hit ring + dwell fill
            tx, ty = DOTS[self.i][0] * W, DOTS[self.i][1] * H
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(tx, ty), 16, 16)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor('#334155'), 2))     # generous hit zone (faint)
            p.drawEllipse(QPointF(tx, ty), HIT_R * min(W, H), HIT_R * min(W, H))
            p.setPen(QPen(QColor('#94a3b8'), 4))     # dwell ring
            p.drawEllipse(QPointF(tx, ty), 30, 30)
            if self.dwell:
                p.setPen(QPen(QColor('#22c55e'), 6))
                span = int(360 * min(self.dwell, HIT_DWELL) / HIT_DWELL)
                p.drawArc(QRectF(tx - 30, ty - 30, 60, 60), 90 * 16, -span * 16)

            # the gaze marker
            if self.gaze is not None:
                gx, gy = self.gaze[0] * W, self.gaze[1] * H
                p.setBrush(QColor(34, 197, 94, 90))
                p.setPen(QPen(QColor('#22c55e'), 3))
                p.drawEllipse(QPointF(gx, gy), 24, 24)
                p.setPen(QPen(QColor('#22c55e'), 2))
                p.drawLine(int(gx - 32), int(gy), int(gx + 32), int(gy))
                p.drawLine(int(gx), int(gy - 32), int(gx), int(gy + 32))

        # disclaimer + live values
        p.setPen(QColor('#f87171'))
        p.setFont(QFont('Arial', 15, QFont.Bold))
        p.drawText(16, 30, 'UNCALIBRATED rough estimate - accuracy is very low')
        p.setPen(QColor('#e2e8f0'))
        p.setFont(QFont('Consolas', 13))
        p.drawText(16, 54, f'dot {min(self.i + 1, len(DOTS))}/{len(DOTS)}   '
                           f'sens_x={self.sens_x:.4f}  sens_y={self.sens_y:.4f}   '
                           f'(arrows tune, B baseline, R restart, Esc quit)')
        p.end()

    def _rebaseline(self):
        self.ref = None
        self.gaze = None
        self._base = deque()
        self._base_until = time.time() + BASELINE_SEC

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_B:
            self._rebaseline()
        elif k == Qt.Key_R:
            self.i = 0
            self.dwell = 0.0
            self.done = False
        elif k == Qt.Key_Right:
            self.sens_x += 0.002
        elif k == Qt.Key_Left:
            self.sens_x = max(0.0, self.sens_x - 0.002)
        elif k == Qt.Key_Up:
            self.sens_y += 0.002
        elif k == Qt.Key_Down:
            self.sens_y = max(0.0, self.sens_y - 0.002)
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            print(f'sens_x={self.sens_x:.4f}  sens_y={self.sens_y:.4f}', flush=True)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)


def main():
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = QApplication(sys.argv)
    win = GazeMarker(camera_index=cam)
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
