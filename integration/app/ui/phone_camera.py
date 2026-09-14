"""Side-camera chooser + phone (Iriun Webcam) helper.

Cameras are shown by NAME with a live preview, so the student never deals with
"index 0/1/2" -- they see their phone's feed and click it. Iriun makes a phone
appear as an ordinary camera on the PC, so AEye still opens whatever the chooser
returns by its index.

`SideCameraDialog`: after exec(), `selected_index` (int) and `selected_label`
(str) hold the pick, or `selected_index` is None if cancelled.
`populate_camera_combo`: fills a dropdown with camera names (for the face cam).
"""

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QComboBox, QPushButton, QFrame, QListWidget,
                               QListWidgetItem)

# Always expose at least these indexes so 0/1 exist even when names can't be read.
MIN_LOCAL_INDEXES = 4

STEPS = (
    "<b>Using your phone?</b> Install <b>Iriun Webcam</b> on the PC (iriun.com) "
    "and on the phone (App Store / Google Play). Connect by <b>USB</b> (data "
    "cable; iPhone tap Trust, Android enable USB debugging) or Wi-Fi, open the "
    "app, and it appears below as <b>\"Iriun Webcam\"</b>."
)


def list_camera_names():
    """Friendly names of attached cameras, in index order. [] on any failure."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:
        return []


def populate_camera_combo(combo, default_index=0):
    """Fill a QComboBox with cameras as their name (itemData = index); falls
    back to 'Camera i' when names can't be read. Selects default_index."""
    combo.clear()
    names = list_camera_names()
    for i in range(max(len(names), MIN_LOCAL_INDEXES)):
        label = names[i] if i < len(names) else f'Camera {i}'
        combo.addItem(label, i)
    if 0 <= default_index < combo.count():
        combo.setCurrentIndex(default_index)
    return names


class CameraPreview(QThread):
    """Reads frames from one camera index for a live preview and releases the
    camera when stopped. Separate from the app's real workers."""

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
            # .copy() detaches the QImage from the numpy buffer before it
            # crosses the thread boundary.
            self.frame.emit(QImage(rgb.data, w, h, 3 * w,
                                   QImage.Format_RGB888).copy())
            self.msleep(30)
        cap.release()


class SideCameraDialog(QDialog):
    """Pick the side camera by name + live preview. `front_index` is the face
    camera, which can't be reused. Sets selected_index / selected_label."""

    def __init__(self, parent=None, front_index=None):
        super().__init__(parent)
        self.setWindowTitle('Choose side camera')
        self.setMinimumWidth(700)
        self.selected_index = None
        self.selected_label = ''
        self._front_index = front_index
        self._preview = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        heading = QLabel('Which camera watches you from the side?')
        heading.setStyleSheet('font-size: 16px; font-weight: bold;')
        outer.addWidget(heading)

        body = QHBoxLayout()
        body.setSpacing(16)
        self.preview = QLabel('Select a camera')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(360, 270)
        self.preview.setFrameShape(QFrame.StyledPanel)
        self.preview.setStyleSheet(
            'background: #0f172a; color: #94a3b8; border-radius: 10px;')
        body.addWidget(self.preview)

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(QLabel('Click a camera to preview it:'))
        self.cam_list = QListWidget()
        self.cam_list.currentRowChanged.connect(self._on_row)
        right.addWidget(self.cam_list, stretch=1)
        self.refresh_btn = QPushButton('Refresh list')
        self.refresh_btn.setToolTip('Re-scan after opening the Iriun app')
        self.refresh_btn.clicked.connect(self._fill)
        right.addWidget(self.refresh_btn)
        steps = QLabel(STEPS)
        steps.setTextFormat(Qt.RichText)
        steps.setWordWrap(True)
        steps.setStyleSheet('color: gray; font-size: 11px;')
        right.addWidget(steps)
        body.addLayout(right, stretch=1)
        outer.addLayout(body)

        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color: gray;')
        outer.addWidget(self.status)

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

        self.finished.connect(self._stop_preview)
        self._fill()

    def _fill(self):
        self._stop_preview()
        self.cam_list.clear()
        names = list_camera_names()
        sel = 0
        for i in range(max(len(names), MIN_LOCAL_INDEXES)):
            label = names[i] if i < len(names) else f'Camera {i}'
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, i)
            self.cam_list.addItem(item)
            if i < len(names) and 'iriun' in names[i].lower():
                sel = i          # jump to the phone if it's there
        self.cam_list.setCurrentRow(sel)

    def _current_index(self):
        item = self.cam_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _on_row(self, _row):
        index = self._current_index()
        self._start_preview(index)
        clash = index == self._front_index
        self.use_btn.setEnabled(not clash)
        if clash:
            self.status.setStyleSheet('color: #a00000;')
            self.status.setText("That's your face camera — pick a different one.")
        else:
            self.status.setStyleSheet('color: gray;')
            self.status.setText('')

    def _start_preview(self, index):
        self._stop_preview()
        if index is None:
            return
        self.preview.setText('Opening camera ...')
        self._preview = CameraPreview(int(index))
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
                            'open the Iriun app, then press Refresh list.')

    def _accept(self):
        index = self._current_index()
        if index is None:
            return
        self.selected_index = int(index)
        item = self.cam_list.currentItem()
        self.selected_label = item.text() if item else f'Camera {index}'
        self.accept()
