"""Phone-as-camera setup dialog (Iriun Webcam).

Self-contained helper. Iriun makes a phone appear as an ordinary camera index
on the PC, so the rest of AEye opens it exactly like any webcam -- this dialog
shows the setup steps, lists the cameras by name, and gives a live preview so
the user can confirm the phone feed. After exec(), `selected_index` holds the
chosen camera index, or stays None if the user cancelled.

The preview opens the camera on its own background thread and releases it as
soon as the dialog closes, so the AEye worker can open the same index next.
"""

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QPushButton, QFrame)

# The picker always exposes at least these indexes so the usual 0/1 exist even
# when camera names cannot be read.
MIN_LOCAL_INDEXES = 4

INSTRUCTIONS = (
    "<b style='color:#b45309;'>Before you start:</b> open the Iriun PC client "
    "and connect your phone <b>before</b> the exam locks down, and keep it "
    "running in the background."
    "<br><br>"
    "<b>1. Install once</b>"
    "<br>&bull; <b>PC:</b> Iriun Webcam from iriun.com"
    "<br>&bull; <b>Phone:</b> Iriun Webcam from the App Store (iPhone) or "
    "Google Play (Android)"
    "<br><br>"
    "<b>2. Connect</b>"
    "<br>&bull; <b>USB cable (best for exams):</b> plug in with a <b>data</b> "
    "cable. iPhone: tap <b>Trust</b>. Android: enable USB debugging once "
    "(Settings &rarr; About phone &rarr; tap \"Build number\" 7&times; &rarr; "
    "Developer options &rarr; USB debugging)."
    "<br>&bull; <b>Wi-Fi:</b> same Wi-Fi as this laptop, then open the app."
    "<br><br>"
    "<b>3. Pick it</b>"
    "<br>Open the Iriun app, press <b>Refresh</b>, choose <b>\"Iriun Webcam\"</b> "
    "and confirm it in the preview."
    "<br><br>"
    "<span style='color:gray;'>Use a data cable, not charge-only; USB also "
    "charges the phone. The free version adds a small watermark.</span>"
)


def list_camera_names():
    """Friendly names of attached cameras, in index order. [] on any failure."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:
        return []


class _PreviewWorker(QThread):
    """Reads frames from one camera index for the dialog preview and releases
    the camera when stopped. Kept separate from the app's real workers."""

    frame = Signal(QImage)
    failed = Signal()

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self._index = index
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self.failed.emit()
            return
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.msleep(30)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # .copy() detaches the QImage from the numpy buffer before it crosses
            # the thread boundary.
            self.frame.emit(QImage(rgb.data, w, h, 3 * w,
                                   QImage.Format_RGB888).copy())
            self.msleep(30)
        cap.release()


class PhoneCameraDialog(QDialog):
    """Iriun setup steps + camera picker + live preview. Sets `selected_index`
    on accept."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Set up phone camera (Iriun Webcam)')
        self.setMinimumWidth(720)
        self.selected_index = None
        self._preview = None
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        heading = QLabel('Use your phone as a camera')
        heading.setStyleSheet('font-size: 17px; font-weight: bold;')
        outer.addWidget(heading)

        body = QHBoxLayout()
        body.setSpacing(18)

        steps = QLabel(INSTRUCTIONS)
        steps.setTextFormat(Qt.RichText)
        steps.setWordWrap(True)
        steps.setAlignment(Qt.AlignTop)
        steps.setFixedWidth(320)
        body.addWidget(steps)

        right = QVBoxLayout()
        right.setSpacing(10)
        self.preview = QLabel('Camera preview')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(360, 270)
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setStyleSheet(
            'background: #0f172a; color: #94a3b8; border-radius: 10px;')
        right.addWidget(self.preview)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel('Camera:'))
        self.cam_box = QComboBox()
        self.cam_box.currentIndexChanged.connect(self._on_cam_changed)
        cam_row.addWidget(self.cam_box, stretch=1)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setToolTip('Re-scan after opening the Iriun app')
        self.refresh_btn.clicked.connect(self._fill_cameras)
        cam_row.addWidget(self.refresh_btn)
        right.addLayout(cam_row)

        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color: gray;')
        right.addWidget(self.status)
        right.addStretch(1)
        body.addLayout(right, stretch=1)
        outer.addLayout(body)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        self.use_btn = QPushButton('Use this camera')
        self.use_btn.setDefault(True)
        self.use_btn.clicked.connect(self._accept)
        btns.addWidget(self.use_btn)
        outer.addLayout(btns)

        # Stops the preview (and releases the camera) however the dialog ends.
        self.finished.connect(self._stop_preview)
        self._fill_cameras()

    def _fill_cameras(self):
        self._stop_preview()
        self._loading = True
        self.cam_box.clear()
        names = list_camera_names()
        for i in range(max(len(names), MIN_LOCAL_INDEXES)):
            label = f'{i} - {names[i]}' if i < len(names) else str(i)
            self.cam_box.addItem(label, i)
        sel = 0
        for i, n in enumerate(names):        # jump straight to Iriun if present
            if 'iriun' in n.lower():
                sel = i
                break
        self.cam_box.setCurrentIndex(sel)
        self._loading = False
        self._start_preview(sel)

    def _on_cam_changed(self, _index):
        if not self._loading:
            self._start_preview(self.cam_box.currentData())

    def _start_preview(self, index):
        self._stop_preview()
        if index is None:
            return
        self.preview.setText('Opening camera ...')
        self.status.setText('')
        self._preview = _PreviewWorker(int(index))
        self._preview.frame.connect(self._show_frame)
        self._preview.failed.connect(self._preview_failed)
        self._preview.start()

    def _stop_preview(self, *args):
        if self._preview is not None:
            self._preview.stop()
            self._preview.wait()
            self._preview = None

    def _show_frame(self, img):
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _preview_failed(self):
        self.preview.setText('No preview')
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText('Could not open this camera. If it is your phone, '
                             'open the Iriun app first, then press Refresh.')

    def _accept(self):
        data = self.cam_box.currentData()
        self.selected_index = data if isinstance(data, int) else 0
        self.accept()
