# Live picture of the Graham scan screen area.

from collections import deque

from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from cheat.calibration_store import load_sessions
from cheat.cheat_detector import CheatDetector, graham_scan
from cheat.train_cheat_model import average_dots


class HullView(QWidget):
    # Draws the dots, the outline and where the eyes are.

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet('background: #0f172a;')
        self.area = CheatDetector()
        self.dots = []                    # graham scan input
        self.scan = []                    # graham scan output
        self.trail = deque(maxlen=60)     # last 2 seconds of eyes
        self.off = False
        timer = QTimer(self)
        timer.timeout.connect(self.update)    # redraw 15 times a second
        timer.start(66)

    def set_user(self, user_id):
        # Load this student's screen area.
        self.area = CheatDetector.load(user_id)
        self.trail.clear()
        self.dots = average_dots(load_sessions(user_id)) if self.area.ready else []
        self.scan = graham_scan(self.dots) if len(self.dots) >= 3 else []

    def on_features(self, feats):
        # Add the newest eye point.
        if self.area.ready:
            self.trail.append((feats['h_ratio'], feats['v_openness']))
            self.off = feats.get('off_screen', False)

    def paintEvent(self, event):
        # Draw everything.
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#0f172a'))
        p.setPen(QColor('#cbd5e1'))
        if not self.area.ready:
            p.drawText(self.rect(), Qt.AlignCenter,
                       'No screen area yet - calibrate, press Train Model, then Start Tracking.')
            p.end()
            return

        # zoom to fit the area
        xs = [q[0] for q in self.area.hull]
        ys = [q[1] for q in self.area.hull]
        pad_x, pad_y = (max(xs) - min(xs)) * 0.6, (max(ys) - min(ys)) * 0.6
        x0, x1, y0, y1 = min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y
        w, h = self.width(), self.height()

        def to_screen(q):
            # Eye values to pixels.
            x = (q[0] - x0) / (x1 - x0) * w
            y = h - (q[1] - y0) / (y1 - y0) * h         # iris higher = up
            return QPointF(min(max(x, 8), w - 8), min(max(y, 8), h - 8))

        # green = screen area
        p.setPen(QPen(QColor('#22c55e'), 2, Qt.DashLine))
        p.setBrush(QColor(34, 197, 94, 40))
        p.drawPolygon(QPolygonF([to_screen(q) for q in self.area.hull]))
        # blue = graham scan outline
        p.setPen(QPen(QColor('#60a5fa'), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([to_screen(q) for q in self.scan]))
        # grey = calibration dots
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#94a3b8'))
        for q in self.dots:
            p.drawEllipse(to_screen(q), 5, 5)
        # eyes now: green inside, red outside
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
        p.drawText(10, h - 10, 'left  <-  eyes  ->  right          up = iris higher (looking up)')
        p.end()
