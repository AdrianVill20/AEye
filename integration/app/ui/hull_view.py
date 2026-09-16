"""Live picture of the Graham scan screen area.

    grey dots      one point per calibration dot (what the Graham scan gets)
    blue outline   the convex hull the Graham scan found around them
    green area     the hull grown to the screen edge = "eyes on the screen"
    moving dot     the student's eyes right now (green inside, red outside),
                   with a short trail of the last 2 seconds

Fed by the FrontCamWorker that Live Tracking already runs (Windows will not
give one camera to two readers).
"""

from collections import deque

from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from cheat.calibration_store import load_sessions
from cheat.cheat_detector import CheatDetector, graham_scan
from cheat.train_cheat_model import newest_dots


class HullView(QWidget):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet('background: #0f172a;')
        self.area = CheatDetector()
        self.dots = []                    # Graham scan input
        self.scan = []                    # Graham scan output, before growing
        self.trail = deque(maxlen=60)     # last ~2 s of eye points
        self.off = False
        timer = QTimer(self)
        timer.timeout.connect(self.update)    # repaint 15x a second, not per frame
        timer.start(66)

    def set_user(self, user_id):
        self.area = CheatDetector.load(user_id)
        self.trail.clear()
        self.dots = newest_dots(load_sessions(user_id), self.area)[0] if self.area.ready else []
        self.scan = graham_scan(self.dots) if len(self.dots) >= 3 else []

    def on_features(self, feats):
        if self.area.ready:
            self.trail.append(self.area.point(feats['h_ratio'], feats['v_openness'],
                                              feats['yaw'], feats['pitch']))
            self.off = feats.get('off_screen', False)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#0f172a'))
        p.setPen(QColor('#cbd5e1'))
        if not self.area.ready:
            p.drawText(self.rect(), Qt.AlignCenter,
                       'No screen area yet - calibrate, press Train Model, then Start Tracking.')
            p.end()
            return

        # Fit the area (plus room around it) into the widget. Eye points further
        # out are drawn on the border, so one wild frame can't shrink the picture.
        xs = [q[0] for q in self.area.hull]
        ys = [q[1] for q in self.area.hull]
        pad_x, pad_y = (max(xs) - min(xs)) * 0.6, (max(ys) - min(ys)) * 0.6
        x0, x1, y0, y1 = min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y
        w, h = self.width(), self.height()

        def to_screen(q):
            x = (q[0] - x0) / (x1 - x0) * w
            y = h - (q[1] - y0) / (y1 - y0) * h         # more open eyes = up
            return QPointF(min(max(x, 8), w - 8), min(max(y, 8), h - 8))

        # green area = grown hull
        p.setPen(QPen(QColor('#22c55e'), 2, Qt.DashLine))
        p.setBrush(QColor(34, 197, 94, 40))
        p.drawPolygon(QPolygonF([to_screen(q) for q in self.area.hull]))
        # blue outline = what the Graham scan found
        p.setPen(QPen(QColor('#60a5fa'), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([to_screen(q) for q in self.scan]))
        # grey dots = Graham scan input
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#94a3b8'))
        for q in self.dots:
            p.drawEllipse(to_screen(q), 5, 5)
        # live eyes + trail
        if self.trail:
            color = QColor('#ef4444') if self.off else QColor('#22c55e')
            p.setPen(QPen(color, 1))
            p.drawPolyline(QPolygonF([to_screen(q) for q in self.trail]))
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(to_screen(self.trail[-1]), 9, 9)

        p.setPen(QColor('#cbd5e1'))
        status = 'OFF SCREEN' if self.off else 'on screen'
        p.drawText(10, 20, f'Graham scan: {len(self.dots)} dots -> {len(self.scan)} corners   |   eyes: {status}')
        p.drawText(10, h - 10, 'left  <-  eyes  ->  right          up = eyes more open (looking up)')
        p.end()
