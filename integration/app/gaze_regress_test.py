"""WebGazer-style CALIBRATED gaze marker (Python port of the core idea).

Why our old marker was inaccurate: it used fixed gains and no fitting. WebGazer
is accurate because it FITS a per-user regression from click points. This does
the same: click each dot (looking at it) to collect (eye-features -> known dot)
pairs, fit a Ridge regression, then show the calibrated marker live. In test
mode every extra click keeps improving it (continuous self-calibration, like
WebGazer). Features come from MediaPipe (iris ratios + head pose) instead of raw
eye-patch pixels - fewer features, but per-user fitting is what buys the accuracy.

Run:  ../.venv/Scripts/python.exe gaze_regress_test.py [cam_index]
      ../.venv/Scripts/python.exe gaze_regress_test.py --selftest
Keys: R restart, Esc quit.
"""
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget

DOTS = [(.5, .5), (.08, .08), (.92, .08), (.08, .92), (.92, .92),
        (.5, .08), (.08, .5), (.92, .5), (.5, .92)]
HIT_R = 0.10          # click must land this close to the training dot


def fit_model(samples):
    # samples: list of (features, (x, y)). Ridge maps features -> screen (x, y).
    X = np.array([s[0] for s in samples], float)
    Y = np.array([s[1] for s in samples], float)
    return Ridge(alpha=1.0).fit(X, Y)


class GazeRegress(QWidget):

    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle('Calibrated gaze marker (WebGazer-style)')
        self.frame = None
        self.last_feats = None
        self.gaze = None
        self.model = None
        self.samples = []          # (features, (x, y)) pairs
        self.i = 0                 # current training dot
        self.err = []              # test-mode error samples (fraction of screen)

        from workers.front_cam_worker import FrontCamWorker
        self.worker = FrontCamWorker(camera_index=camera_index, detect=False)
        self.worker.frame_ready.connect(lambda q: setattr(self, 'frame', q))
        self.worker.features_ready.connect(self._on_features)
        self.worker.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    def _on_features(self, f):
        self.last_feats = [f['h_ratio'], f['v_openness'], f['yaw'], f['pitch']]
        if self.model is not None:
            x, y = self.model.predict([self.last_feats])[0]
            self.gaze = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    def mousePressEvent(self, e):
        if self.last_feats is None:
            return
        click = (e.position().x() / self.width(), e.position().y() / self.height())
        if self.model is None:                       # TRAIN: pair features with the known dot
            dot = DOTS[self.i]
            if ((click[0] - dot[0]) ** 2 + (click[1] - dot[1]) ** 2) ** 0.5 > HIT_R:
                return                               # must click on the shown dot
            self.samples.append((self.last_feats, dot))
            self.i += 1
            if self.i >= len(DOTS):
                self.model = fit_model(self.samples)
        else:                                        # TEST: measure error + keep improving
            if self.gaze is not None:
                self.err.append(((self.gaze[0] - click[0]) ** 2 + (self.gaze[1] - click[1]) ** 2) ** 0.5)
            self.samples.append((self.last_feats, click))
            self.model = fit_model(self.samples)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        W, H = self.width(), self.height()
        if self.frame is not None:
            img = self.frame.scaled(self.size(), Qt.KeepAspectRatio)
            p.setOpacity(0.2)
            p.drawImage((W - img.width()) // 2, (H - img.height()) // 2, img)
            p.setOpacity(1.0)

        if self.model is None:                       # training
            tx, ty = DOTS[self.i]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(tx * W, ty * H), 15, 15)
            p.setPen(QColor('#e2e8f0'))
            p.setFont(QFont('Arial', 20))
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop,
                       f'\nLook at the dot and CLICK it   ({self.i}/{len(DOTS)})')
        else:                                        # calibrated live marker
            p.setPen(QPen(QColor('#475569'), 2))
            p.setBrush(Qt.NoBrush)
            for rx, ry in DOTS:
                p.drawEllipse(QPointF(rx * W, ry * H), 8, 8)
            if self.gaze is not None:
                gx, gy = self.gaze[0] * W, self.gaze[1] * H
                p.setBrush(QColor(34, 197, 94, 110))
                p.setPen(QPen(QColor('#22c55e'), 3))
                p.drawEllipse(QPointF(gx, gy), 20, 20)
                p.setPen(QPen(QColor('#22c55e'), 2))
                p.drawLine(int(gx - 28), int(gy), int(gx + 28), int(gy))
                p.drawLine(int(gx), int(gy - 28), int(gx), int(gy + 28))
            mean_err = 100 * sum(self.err) / len(self.err) if self.err else None
            p.setPen(QColor('#e2e8f0'))
            p.setFont(QFont('Arial', 16))
            msg = 'Calibrated. Click where you look to test/improve.  R restart.'
            if mean_err is not None:
                msg += f'   mean error: {mean_err:.0f}% of screen ({len(self.err)} clicks)'
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, '\n' + msg)
        p.end()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_R:
            self.samples, self.i, self.model, self.gaze, self.err = [], 0, None, None, []

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)


def _selftest():
    # A truly linear mapping should be recovered almost exactly.
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(40):
        f = rng.normal(size=4)
        x = 0.5 + 0.1 * f[0] + 0.02 * f[2]
        y = 0.5 + 0.1 * f[1] + 0.02 * f[3]
        samples.append((list(f), (x, y)))
    m = fit_model(samples)
    (px, py), (tx, ty) = m.predict([samples[0][0]])[0], samples[0][1]
    assert abs(px - tx) < 0.03 and abs(py - ty) < 0.03, (px, py, tx, ty)
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        _selftest()
        return
    cam = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    app = QApplication(sys.argv)
    win = GazeRegress(camera_index=cam)
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
