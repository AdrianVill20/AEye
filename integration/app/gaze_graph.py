"""Live z-score graph of h_ratio, yaw and pitch.

Read-only. Fed by the FrontCamWorker DetectionView already starts, because
Windows will not give camera 0 to two cv2.VideoCaptures at once.
"""

import math
import time
from collections import deque

from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

import cheat_detector
import calibration_store

PLOTTED = ('h_ratio', 'yaw', 'pitch')
KEYS = [f for f in cheat_detector.FEATURES if f in PLOTTED]
LABELS = {'h_ratio': 'eyegaze', 'yaw': 'head sideways', 'pitch': 'head up/down'}
COLORS = {'h_ratio': QColor(30, 60, 160),
          'yaw': QColor(160, 40, 40),
          'pitch': QColor(30, 110, 60)}

# Smallest movement we count as "one unit" on the graph. Calibration stds are
# tiny (the student holds still), so dividing by them made a small tilt spike
# off the chart. Floor them with these instead.
MIN_STD = {'h_ratio': 0.08, 'yaw': 8.0, 'pitch': 8.0, 'roll': 8.0}

WINDOW_SECONDS = 30
MAXLEN = 900             # 30 s at 30 fps
Z_LIMIT = 6.0
BAND = 2.0


def floor_stds(stds):
    """Never divide by a std smaller than MIN_STD, so small head movement
    stays small on the graph."""
    return [max(sd, MIN_STD[k]) for sd, k in zip(stds, KEYS)]


def load_baseline(user_id):
    """Return (means, stds, source) for KEYS, or None. Never raises."""
    idx = [cheat_detector.FEATURES.index(k) for k in KEYS]

    try:
        import joblib
        scaler = joblib.load(cheat_detector.user_model_path(user_id))['scaler']
        means = [float(scaler.mean_[i]) for i in idx]
        stds = [float(scaler.scale_[i]) for i in idx]
        return means, floor_stds(stds), 'model'
    except Exception:
        pass

    try:
        samples = calibration_store.load(user_id)['samples']
        means, stds = [], []
        for k in KEYS:
            vals = [float(s[k]) for s in samples]
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            means.append(m)
            stds.append(math.sqrt(var))
        return means, floor_stds(stds), 'calibration'
    except Exception:
        return None


try:
    from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
    HAVE_QTCHARTS = True
except ImportError:
    HAVE_QTCHARTS = False


class PainterPlot(QWidget):
    """Fallback plot drawn with QPainter."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self._times = []
        self._values = {k: [] for k in KEYS}
        self._now = time.time()

    def redraw(self, times, values):
        self._now = time.time()
        self._times = list(times)
        self._values = {k: list(v) for k, v in values.items()}
        self.update()

    def _ypix(self, z):
        h = self.height()
        return h / 2.0 - (z / Z_LIMIT) * (h / 2.0 - 4)

    def _xpix(self, t):
        w = self.width()
        return w + (t - self._now) / WINDOW_SECONDS * w

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(QPen(Qt.gray, 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.drawLine(QPointF(0, self._ypix(0)), QPointF(w, self._ypix(0)))

        p.setPen(QPen(Qt.gray, 1, Qt.DashLine))
        for z in (-BAND, BAND):
            p.drawLine(QPointF(0, self._ypix(z)), QPointF(w, self._ypix(z)))

        for k in KEYS:
            pts = QPolygonF([QPointF(self._xpix(t), self._ypix(v))
                             for t, v in zip(self._times, self._values[k])])
            p.setPen(QPen(COLORS[k], 1))
            p.drawPolyline(pts)

        x = 6
        for k in KEYS:
            p.setPen(QPen(COLORS[k], 2))
            p.drawLine(QPointF(x, 10), QPointF(x + 14, 10))
            p.drawText(x + 18, 14, LABELS[k])
            x += 60
        p.end()


if HAVE_QTCHARTS:

    class ChartPlot(QChartView):
        """Three lines plus the two dashed band lines."""

        def __init__(self):
            super().__init__()
            self.setMinimumHeight(150)
            chart = QChart()
            chart.legend().setAlignment(Qt.AlignBottom)

            self._series = {}
            for k in KEYS:
                s = QLineSeries()
                s.setName(LABELS[k])
                s.setPen(QPen(COLORS[k], 1))
                chart.addSeries(s)
                self._series[k] = s

            band = []
            for z in (-BAND, BAND):
                s = QLineSeries()
                s.append(-WINDOW_SECONDS, z)
                s.append(0.0, z)
                s.setPen(QPen(Qt.gray, 1, Qt.DashLine))
                chart.addSeries(s)
                for marker in chart.legend().markers(s):
                    marker.setVisible(False)
                band.append(s)

            ax = QValueAxis()
            ax.setRange(-WINDOW_SECONDS, 0)
            ax.setLabelFormat('%.0f')
            ay = QValueAxis()
            ay.setRange(-Z_LIMIT, Z_LIMIT)
            ay.setTickCount(7)
            chart.addAxis(ax, Qt.AlignBottom)
            chart.addAxis(ay, Qt.AlignLeft)
            for s in list(self._series.values()) + band:
                s.attachAxis(ax)
                s.attachAxis(ay)

            self.setChart(chart)

        def redraw(self, times, values):
            now = time.time()
            for k in KEYS:
                pts = [QPointF(t - now, v)
                       for t, v in zip(list(times), list(values[k]))]
                self._series[k].replace(pts)


class GazeGraph(QWidget):
    """Plot plus one status line. Connect features_ready to on_features."""

    def __init__(self):
        super().__init__()
        self.times = deque(maxlen=MAXLEN)
        self.values = {k: deque(maxlen=MAXLEN) for k in KEYS}
        self._means = None
        self._stds = None
        self._source = None

        self.plot = ChartPlot() if HAVE_QTCHARTS else PainterPlot()

        self.status = QLabel('No data yet.')
        self.status.setStyleSheet('color: gray; font-size: 11px;')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.status)

        # repaint on timer, not per frame
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(66)

    def set_user(self, user_id):
        """Load this student's baseline and clear the old lines."""
        self.times.clear()
        for d in self.values.values():
            d.clear()

        baseline = load_baseline(user_id)
        if baseline is None:
            self._means = self._stds = self._source = None
            self.status.setText('not calibrated - plotting raw values')
        else:
            self._means, self._stds, self._source = baseline
            self.status.setText(f'baseline from {self._source}; waiting for frames')

    def on_features(self, feats):
        self.times.append(time.time())
        for i, k in enumerate(KEYS):
            v = float(feats[k])
            if self._means is not None:
                std = self._stds[i] if self._stds[i] > 1e-9 else 1.0
                v = (v - self._means[i]) / std
            self.values[k].append(max(-Z_LIMIT, min(Z_LIMIT, v)))

    def _redraw(self):
        if not self.times or not self.isVisible():
            return
        self.plot.redraw(self.times, self.values)
        parts = [f'{LABELS[k]} {self.values[k][-1]:.1f}' for k in KEYS]
        prefix = '' if self._means is not None else 'not calibrated (raw)  '
        self.status.setText(prefix + '  '.join(parts))
