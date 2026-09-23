"""Click-the-dot + gaze-consistency calibration test (calibration-free).

Advance rule: you must CLICK the dot AND your rough gaze must be near it at the
click. Clicking proves engagement + gives a clean sample trigger; the gaze check
stops "click while looking away". Dot order is shuffled and jittered so you can't
run on memory. Uncalibrated -> rough by design (see the earlier notes).

Run:  ../.venv/Scripts/python.exe gaze_marker_test.py [cam_index]
      ../.venv/Scripts/python.exe gaze_marker_test.py --selftest
Keys: arrows tune sensitivity, B re-baseline, R restart, Esc quit.
"""
import random
import sys
import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget

BASELINE_SEC = 1.0
GAZE_EMA = 0.25
W_EYE = 150.0
SENS_X = 0.025
SENS_Y = 0.030
HIT_R = 0.08        # mouse click must land this close to the dot
CONSIST_R = 0.20    # rough gaze must be this close to the dot (generous: it's uncalibrated)

BASE_DOTS = [(.5, .5), (.1, .1), (.9, .1), (.1, .9), (.9, .9),
             (.5, .1), (.1, .5), (.9, .5), (.5, .9)]


def _near(p, dot, r):
    return p is not None and ((p[0] - dot[0]) ** 2 + (p[1] - dot[1]) ** 2) ** 0.5 < r


def valid_click(click, gaze, dot):
    # Accept only if the click AND the rough gaze are both near the dot.
    return _near(click, dot, HIT_R) and _near(gaze, dot, CONSIST_R)


def _new_sequence():
    pts = BASE_DOTS[:]
    random.shuffle(pts)                                   # no memorised order
    return [(min(.94, max(.06, x + random.uniform(-.04, .04))),
             min(.94, max(.06, y + random.uniform(-.04, .04)))) for x, y in pts]


class GazeClick(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle('Click the dot while looking at it (uncalibrated)')
        self.sens_x, self.sens_y = SENS_X, SENS_Y
        self.frame = None
        self.gaze = None
        self._base = deque()
        self._base_until = time.time() + BASELINE_SEC
        self.ref = None
        self.dots = _new_sequence()
        self.i = 0
        self.samples = 0
        self.done = False
        self.flash = None                                 # (color, msg, until)

        from workers.front_cam_worker import FrontCamWorker
        self.worker = FrontCamWorker(camera_index=camera_index, detect=False)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.features_ready.connect(self._on_features)
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    def _on_frame(self, qimg):
        self.frame = qimg

    def _on_features(self, feats):
        vals = (feats['yaw'], feats['pitch'], feats['h_ratio'], feats['v_openness'])
        if self.ref is None:
            self._base.append(vals)
            if time.time() >= self._base_until and self._base:
                n = len(self._base)
                self.ref = tuple(sum(v[k] for v in self._base) / n for k in range(4))
            return
        yaw0, pitch0, h0, v0 = self.ref
        x = min(1.0, max(0.0, 0.5 + self.sens_x * ((feats['yaw'] - yaw0) + W_EYE * (feats['h_ratio'] - h0))))
        y = min(1.0, max(0.0, 0.5 + self.sens_y * ((feats['pitch'] - pitch0) - W_EYE * (feats['v_openness'] - v0))))
        self.gaze = (x, y) if self.gaze is None else \
            (self.gaze[0] * (1 - GAZE_EMA) + x * GAZE_EMA, self.gaze[1] * (1 - GAZE_EMA) + y * GAZE_EMA)

    def _set_flash(self, color, msg):
        self.flash = (color, msg, time.time() + 0.9)

    def mousePressEvent(self, e):
        if self.ref is None or self.done:
            return
        click = (e.position().x() / self.width(), e.position().y() / self.height())
        dot = self.dots[self.i]
        if valid_click(click, self.gaze, dot):
            self.samples += 1                             # record the sample here (target = dot)
            self._set_flash('#22c55e', 'looked + clicked')
            self.i += 1
            self.done = self.i >= len(self.dots)
        elif not _near(click, dot, HIT_R):
            self._set_flash('#ef4444', 'click ON the dot')
        else:
            self._set_flash('#ef4444', 'clicked, but your gaze is off - look at the dot')

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        W, H = self.width(), self.height()
        if self.frame is not None:
            img = self.frame.scaled(self.size(), Qt.KeepAspectRatio)
            p.setOpacity(0.2)
            p.drawImage((W - img.width()) // 2, (H - img.height()) // 2, img)
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
            tx, ty = self.dots[self.i]
            cx, cy = int(tx * W), int(ty * H)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(cx, cy), 16, 16)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor('#334155'), 2))
            p.drawEllipse(QPointF(cx, cy), HIT_R * min(W, H), HIT_R * min(W, H))
            if self.gaze is not None:                     # rough gaze marker
                gx, gy = self.gaze[0] * W, self.gaze[1] * H
                p.setBrush(QColor(34, 197, 94, 90))
                p.setPen(QPen(QColor('#22c55e'), 3))
                p.drawEllipse(QPointF(gx, gy), 20, 20)
                p.setPen(QPen(QColor('#22c55e'), 2))
                p.drawLine(int(gx - 28), int(gy), int(gx + 28), int(gy))
                p.drawLine(int(gx), int(gy - 28), int(gx), int(gy + 28))

        if self.flash and time.time() < self.flash[2]:
            p.setPen(QColor(self.flash[0]))
            p.setFont(QFont('Arial', 20, QFont.Bold))
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignBottom, self.flash[1] + '\n')

        p.setPen(QColor('#f87171'))
        p.setFont(QFont('Arial', 15, QFont.Bold))
        p.drawText(16, 30, 'UNCALIBRATED rough estimate - accuracy is very low')
        p.setPen(QColor('#e2e8f0'))
        p.setFont(QFont('Consolas', 13))
        p.drawText(16, 54, f'dot {min(self.i + 1, len(self.dots))}/{len(self.dots)}   accepted={self.samples}   '
                           f'sens_x={self.sens_x:.4f} sens_y={self.sens_y:.4f}   (arrows/B/R/Esc)')
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
            self.dots = _new_sequence()
            self.i, self.samples, self.done = 0, 0, False
        elif k == Qt.Key_Right:
            self.sens_x += 0.002
        elif k == Qt.Key_Left:
            self.sens_x = max(0.0, self.sens_x - 0.002)
        elif k == Qt.Key_Up:
            self.sens_y += 0.002
        elif k == Qt.Key_Down:
            self.sens_y = max(0.0, self.sens_y - 0.002)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)


def _selftest():
    d = (0.5, 0.5)
    assert valid_click((0.5, 0.5), (0.55, 0.52), d)          # looked + clicked
    assert not valid_click((0.5, 0.5), (0.9, 0.9), d)        # clicked, looked away
    assert not valid_click((0.9, 0.1), (0.5, 0.5), d)        # looked, clicked off
    assert not valid_click((0.5, 0.5), None, d)              # no gaze yet
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        _selftest()
        return
    cam = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    app = QApplication(sys.argv)
    win = GazeClick(camera_index=cam)
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
