import sys
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QFrame, QComboBox, QMdiArea, QMdiSubWindow, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QInputDialog, QProgressBar
REPO_ROOT = Path(__file__).resolve().parent.parent
from gaze_worker import GazeWorker
from posture_worker import SideCameraWorker
from headpose_worker import HeadPoseWorker
from front_cam_worker import FrontCamWorker
from front_cam_logger import FrontCamLogWriter
import calibration_store
from cheat_logger import CheatEventLogger
from db_config import get_connection

class LoginView(QWidget):

    def __init__(self, on_login):
        super().__init__()
        self._on_login = on_login
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel('AEye — Sign In')
        title.setStyleSheet('font-size: 20px; font-weight: bold;')
        title.setAlignment(Qt.AlignCenter)
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText('School-issued ID')
        self.id_input.setMaximumWidth(300)
        self.pw_input = QLineEdit()
        self.pw_input.setPlaceholderText('Password')
        self.pw_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw_input.setMaximumWidth(300)
        self.student_radio = QRadioButton('Student')
        self.proctor_radio = QRadioButton('Proctor')
        self.student_radio.setChecked(True)
        role_group = QButtonGroup(self)
        role_group.addButton(self.student_radio)
        role_group.addButton(self.proctor_radio)
        roles = QHBoxLayout()
        roles.setAlignment(Qt.AlignCenter)
        roles.addWidget(self.student_radio)
        roles.addWidget(self.proctor_radio)
        self.error_label = QLabel('')
        self.error_label.setStyleSheet('color: red;')
        self.error_label.setAlignment(Qt.AlignCenter)
        sign_in_btn = QPushButton('Sign In')
        sign_in_btn.setMaximumWidth(300)
        sign_in_btn.clicked.connect(self._handle_sign_in)
        layout.addWidget(title)
        layout.addWidget(self.id_input, alignment=Qt.AlignCenter)
        layout.addWidget(self.pw_input, alignment=Qt.AlignCenter)
        layout.addLayout(roles)
        layout.addWidget(self.error_label)
        layout.addWidget(sign_in_btn, alignment=Qt.AlignCenter)

    def _handle_sign_in(self):
        role = 'student' if self.student_radio.isChecked() else 'proctor'
        self._on_login(self.id_input.text(), self.pw_input.text(), role)

    def show_error(self, message):
        self.error_label.setText(message)

class AnalysisTab(QWidget):

    def __init__(self, title, camera_desc, stats_fields, owner):
        super().__init__()
        root = QHBoxLayout(self)
        self.feed_area = QFrame()
        self.feed_area.setFrameShape(QFrame.Shape.Box)
        self.feed_area.setMinimumSize(320, 240)
        self.feed_area.setStyleSheet('background-color: #111; color: #ddd;')
        feed_layout = QVBoxLayout(self.feed_area)
        self.placeholder = QLabel(f'📷  {camera_desc}\n\nLive camera feed + ML overlay will be embedded here.\nProvided by: {owner}')
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        feed_layout.addWidget(self.placeholder)
        self._panel = QVBoxLayout()
        heading = QLabel(title)
        heading.setStyleSheet('font-weight: bold; font-size: 16px;')
        self._panel.addWidget(heading)
        self.stat_labels = {}
        for field in stats_fields:
            lbl = QLabel(f'{field}: --')
            self.stat_labels[field] = lbl
            self._panel.addWidget(lbl)
        self._panel.addStretch()
        owner_label = QLabel(f'Owner: {owner}')
        owner_label.setStyleSheet('color: gray;')
        self._panel.addWidget(owner_label)
        root.addWidget(self.feed_area, stretch=3)
        root.addLayout(self._panel, stretch=1)
        self._proc = None
        self._script = None
        self._cwd = None

    def set_feed_widget(self, widget):
        self.placeholder.hide()
        self.feed_area.layout().addWidget(widget)

    def enable_controls(self, on_start, on_stop):
        self._on_start = on_start
        self._on_stop = on_stop
        self._ctrl_btn = QPushButton('Start')
        self._ctrl_btn.clicked.connect(self._toggle_controls)
        self._panel.insertWidget(1, self._ctrl_btn)

    def _toggle_controls(self):
        if self._ctrl_btn.text() == 'Start':
            self._on_start()
            self._ctrl_btn.setText('Stop')
        else:
            self._on_stop()
            self._ctrl_btn.setText('Start')

    def update_stats(self, stats):
        for field, value in stats.items():
            if field in self.stat_labels:
                self.stat_labels[field].setText(f'{field}: {value}')

    def set_launcher(self, script, cwd):
        self._script = str(script)
        self._cwd = str(cwd)
        self._launch_btn = QPushButton('Start')
        self._launch_btn.clicked.connect(self._toggle_launch)
        self._panel.insertWidget(1, self._launch_btn)

    def _toggle_launch(self):
        win = self.window()
        if self._launch_btn.text() == 'Start':
            self._proc = subprocess.Popen([sys.executable, self._script], cwd=self._cwd)
            self._launch_btn.setText('Stop')
            win.showMinimized()
        else:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None
            self._launch_btn.setText('Start')
            win.showFullScreen()

    def stop_launch(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

class CameraSection(QWidget):
    """One camera view for the Posture tab.

    It has a dropdown to choose which camera index to use, a Start/Stop
    button, and the live posture feed. Each section runs its own
    SideCameraWorker, so the two sections are fully independent.
    """

    def __init__(self, title, worker_class, default_index=0, log_to_db=False):
        super().__init__()
        self.worker_class = worker_class   # GazeWorker or SideCameraWorker
        self.log_to_db = log_to_db         # only SideCameraWorker reads this
        self.worker = None
        self.log_writer = None                  # the running worker, created on Start

        layout = QVBoxLayout(self)

        # Title (e.g. "Camera 1")
        heading = QLabel(title)
        heading.setStyleSheet('font-weight: bold; font-size: 15px;')
        heading.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)

        # Controls row: "Camera index:" + dropdown + Start/Stop button
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Camera index:'))
        self.index_box = QComboBox()
        self.index_box.addItems(['0', '1', '2', '3'])   # which webcam to open
        self.index_box.setCurrentText(str(default_index))
        controls.addWidget(self.index_box)
        self.button = QPushButton('Start')
        self.button.clicked.connect(self._toggle)
        controls.addWidget(self.button)
        layout.addLayout(controls)

        # Video feed area
        self.video = QLabel('No feed yet')
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(280, 210)
        # Ignore the pixmap's own size so the feed fills the space and resizes
        # with the window instead of shrinking frame-by-frame.
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet('background-color: #111; color: #ddd;')
        layout.addWidget(self.video, stretch=1)

        # Small status line (landmark count, or "Camera N unavailable")
        self.status = QLabel('')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet('color: gray;')
        layout.addWidget(self.status)

    def _toggle(self):
        # Start if stopped, stop if running.
        if self.worker is None:
            self.start()
        else:
            self.stop()

    def start(self):
        camera_index = int(self.index_box.currentText())   # read the dropdown
        session = self.window().session                    # who is logged in
        user_id = session.user_id if session else 'test_user'

        # SideCameraWorker owns its own DB writer, switched on with log_to_db.
        # GazeWorker/FrontCamWorker don't accept that argument, so only pass it
        # to the worker class that supports it.
        worker_kwargs = dict(camera_index=camera_index, session_user_id=user_id)
        if self.worker_class is SideCameraWorker:
            worker_kwargs['log_to_db'] = self.log_to_db
        self.worker = self.worker_class(**worker_kwargs)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.stats_ready.connect(self._show_status)

        # Only FrontCamWorker emits record_ready (raw values for DB logging).
        # Other worker types (GazeWorker, SideCameraWorker) don't have this
        # signal, so guard with hasattr rather than connecting blindly.
        self.log_writer = None
        if hasattr(self.worker, 'record_ready'):
            self.log_writer = FrontCamLogWriter()
            self.log_writer.start()
            self.worker.record_ready.connect(self.log_writer.enqueue)

        self.worker.start()

        self.index_box.setEnabled(False)   # can't change camera while running
        self.button.setText('Stop')

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        if self.log_writer is not None:
            self.log_writer.stop()
            self.log_writer.wait()
            self.log_writer = None
        self.index_box.setEnabled(True)
        self.button.setText('Start')

    def _show_frame(self, qimg):
        # Scale the frame to fit the label, keeping aspect ratio.
        pixmap = QPixmap.fromImage(qimg)
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_status(self, stats):
        # Show every stat the worker sends, one per line. Works for both gaze
        # stats (Direction/Gaze ratio/Blink) and posture stats, and also shows
        # the "Camera N unavailable" message the worker sends when it can't open.
        lines = [f'{field}: {value}' for field, value in stats.items()]
        self.status.setText('\n'.join(lines))


class PostureTab(QWidget):
    """Individual posture test: one camera running posture detection, so posture
    can be tested on its own (separate from the combined Head + Posture view)."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.camera = CameraSection('Posture Camera', SideCameraWorker, default_index=0)
        layout.addWidget(self.camera)

    def stop_all(self):
        # Called when the app closes, to shut the camera down cleanly.
        self.camera.stop()


class GazePostureTab(QWidget):
    """Combined tab: eye gaze on camera 1, posture on camera 2.

    Reuses the same CameraSection widget as the Posture tab, just with a
    different worker on each side (GazeWorker vs SideCameraWorker).
    """

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self.camera1 = CameraSection('Eye Gaze — Camera 1', GazeWorker, default_index=0)
        self.camera2 = CameraSection('Posture — Camera 2', SideCameraWorker, default_index=1)
        layout.addWidget(self.camera1)
        layout.addWidget(self.camera2)

    def stop_all(self):
        self.camera1.stop()
        self.camera2.stop()


class ViewWindow(QMdiSubWindow):
    """An MDI sub-window that hides instead of closing, so the toolbar's
    buttons can reopen it later. (A normal close would remove it for good.)"""

    def closeEvent(self, event):
        event.ignore()   # don't actually close...
        self.hide()      # ...just hide it, so show() can bring it back


class FrontSideCamTab(QWidget):
    """Front cam (combined eye gaze + head pose) + side cam (posture)."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self.front = CameraSection('Front Cam — Gaze + Head Pose', FrontCamWorker, default_index=0)
        self.side = CameraSection('Side Cam — Posture', SideCameraWorker, default_index=1, log_to_db=True)
        layout.addWidget(self.front)
        layout.addWidget(self.side)

    def stop_all(self):
        self.front.stop()
        self.side.stop()


class CalibrationView(QWidget):
    """Step 1: record a short sample of the student's NORMAL behaviour while
    they read a passage, and save it to JSON (calibration_data/). That JSON is
    the seed the training script turns into the student's personal model."""

    def __init__(self, on_proceed=None):
        super().__init__()
        self._on_proceed = on_proceed
        self.worker = None
        self._samples = []
        self._user_id = None
        self._remaining = 0
        self._elapsed = 0

        self.setObjectName('calibRoot')
        self.setStyleSheet('''
            #calibRoot { background: #eef2f7; }
            #card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; }
            #brand { color: #4f46e5; font-size: 26px; font-weight: 800; }
            #subtitle { color: #9aa2af; font-size: 12px; letter-spacing: 2px; }
            #tagline { color: #374151; font-size: 15px; }
            #passage { background: #fffdf7; border: 1px solid #ece6d5; border-radius: 12px;
                       color: #2b2b2b; font-size: 17px; padding: 18px; }
            #previewCaption { color: #9aa2af; font-size: 11px; }
            #status { color: #6b7280; font-size: 13px; }
            QProgressBar#bar { border: none; background: #e9ecf3; border-radius: 8px;
                               height: 16px; text-align: center; }
            QProgressBar#bar::chunk { background: #4f46e5; border-radius: 8px; }
            QComboBox#dur { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 8px;
                            background: #fff; color: #374151; }
            QPushButton#primary { background: #4f46e5; color: white; border: none; border-radius: 10px;
                                  padding: 11px 22px; font-weight: 700; }
            QPushButton#primary:hover { background: #4338ca; }
            QPushButton#primary:disabled { background: #c7c9e6; }
            QPushButton#secondary { background: transparent; color: #4f46e5; border: 2px solid #4f46e5;
                                    border-radius: 10px; padding: 9px 18px; font-weight: 700; }
            QPushButton#secondary:hover { background: #eef0ff; }
            #cmd { background: #0f172a; color: #34d399; border-radius: 10px; padding: 12px;
                   font-family: Consolas, monospace; font-size: 12px; }
        ''')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        card = QFrame()
        card.setObjectName('card')
        card.setMaximumWidth(980)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(28, 24, 28, 24)
        card_l.setSpacing(16)

        brand = QLabel('\U0001F441️  AEye')
        brand.setObjectName('brand')
        subtitle = QLabel('PERSONAL CALIBRATION')
        subtitle.setObjectName('subtitle')
        header = QVBoxLayout()
        header.setSpacing(0)
        header.addWidget(brand)
        header.addWidget(subtitle)
        card_l.addLayout(header)

        tagline = QLabel(
            'Before your exam, AEye learns what <b>normal</b> looks like for '
            '<b>you</b> — so your own habits are never mistaken for cheating.')
        tagline.setObjectName('tagline')
        tagline.setWordWrap(True)
        card_l.addWidget(tagline)

        steps = QHBoxLayout()
        steps.setSpacing(12)
        steps.addWidget(self._step('1', 'Read naturally',
            'Read the passage at your own pace for a short while.'))
        steps.addWidget(self._step('2', 'We learn your normal',
            'AEye saves how you look at the screen — your baseline, not a generic one.'))
        steps.addWidget(self._step('3', 'Judged fairly',
            'Glances, fidgets and glasses become normal for you, so real focus is never flagged.'))
        card_l.addLayout(steps)

        body = QHBoxLayout()
        body.setSpacing(16)
        self.passage = QLabel(
            'Effective studying is less about the number of hours spent and more '
            'about the quality of attention during those hours. When you sit down '
            'to review, remove the distractions within reach, decide on a single '
            'goal for the session, and work in focused blocks separated by short '
            'breaks. As you read, your eyes move across the lines in small jumps, '
            'pausing briefly to take in groups of words. Try to keep your attention '
            'on the text in front of you, the way you would during a real '
            'examination. If your mind wanders, gently bring it back to the '
            'sentence you were on.')
        self.passage.setObjectName('passage')
        self.passage.setWordWrap(True)
        self.passage.setAlignment(Qt.AlignTop)
        body.addWidget(self.passage, stretch=3)

        preview_col = QVBoxLayout()
        preview_col.setSpacing(6)
        self.video = QLabel('Camera preview')
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(260, 200)
        self.video.setMaximumHeight(230)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet('background: #0f172a; border-radius: 10px; color: #cbd5e1;')
        caption = QLabel('You, right now')
        caption.setObjectName('previewCaption')
        caption.setAlignment(Qt.AlignCenter)
        preview_col.addWidget(self.video, stretch=1)
        preview_col.addWidget(caption)
        body.addLayout(preview_col, stretch=2)
        card_l.addLayout(body)

        self.progress = QProgressBar()
        self.progress.setObjectName('bar')
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        card_l.addWidget(self.progress)

        self.status = QLabel('Ready when you are — press Start Calibration to begin.')
        self.status.setObjectName('status')
        card_l.addWidget(self.status)

        controls = QHBoxLayout()
        dur_lbl = QLabel('Duration:')
        dur_lbl.setStyleSheet('color: #6b7280;')
        controls.addWidget(dur_lbl)
        self.duration_box = QComboBox()
        self.duration_box.setObjectName('dur')
        self.duration_box.addItems(['30', '60', '120'])
        self.duration_box.setCurrentText('120')
        controls.addWidget(self.duration_box)
        controls.addStretch(1)
        self.start_btn = QPushButton('Start Calibration')
        self.start_btn.setObjectName('primary')
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start)
        controls.addWidget(self.start_btn)
        self.proceed_btn = QPushButton('Proceed to Monitoring  ▸')
        self.proceed_btn.setObjectName('secondary')
        self.proceed_btn.setCursor(Qt.PointingHandCursor)
        self.proceed_btn.clicked.connect(self._proceed)
        controls.addWidget(self.proceed_btn)
        card_l.addLayout(controls)

        self.cmd_label = QLabel('')
        self.cmd_label.setObjectName('cmd')
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cmd_label.setVisible(False)
        card_l.addWidget(self.cmd_label)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    def _step(self, num, title, desc):
        frame = QFrame()
        frame.setStyleSheet(
            'QFrame { background: #f8f9fc; border: 1px solid #eceef3; border-radius: 10px; }')
        v = QVBoxLayout(frame)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(3)
        n = QLabel(num)
        n.setStyleSheet('color: #4f46e5; font-weight: 800; font-size: 16px; border: none;')
        t = QLabel(title)
        t.setStyleSheet('color: #111827; font-weight: 700; border: none;')
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet('color: #6b7280; font-size: 12px; border: none;')
        v.addWidget(n)
        v.addWidget(t)
        v.addWidget(d)
        return frame

    def _start(self):
        if self.worker is not None:
            return
        session = self.window().session
        self._user_id = session.user_id if session else 'test_user'
        self._samples = []
        # detect=False -> record only. We collect the raw features via
        # features_ready and save them to JSON; nothing is written to MySQL.
        self.worker = FrontCamWorker(camera_index=0, session_user_id=self._user_id, detect=False)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.features_ready.connect(self._collect)
        self.worker.start()
        self._remaining = int(self.duration_box.currentText())
        self._elapsed = 0
        self.progress.setRange(0, self._remaining)
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.duration_box.setEnabled(False)
        self.cmd_label.setVisible(False)
        self.status.setStyleSheet('color: #4f46e5; font-size: 13px; font-weight: 600;')
        self.status.setText(f'● Recording your normal behaviour…  {self._remaining}s left')
        self._timer.start()

    def _collect(self, feats):
        self._samples.append(feats)

    def _tick(self):
        self._elapsed += 1
        self.progress.setValue(self._elapsed)
        left = self._remaining - self._elapsed
        self.status.setText(f'● Recording your normal behaviour…  {max(left, 0)}s left')
        if self._elapsed >= self._remaining:
            self._finish()

    def _finish(self):
        self._timer.stop()
        self._stop_worker()
        self.start_btn.setEnabled(True)
        self.start_btn.setText('Re-record')
        self.duration_box.setEnabled(True)
        if len(self._samples) < 100:
            self.status.setStyleSheet('color: #b91c1c; font-size: 13px; font-weight: 600;')
            self.status.setText(
                f'Only {len(self._samples)} samples captured — keep your face in view, then Re-record.')
            return
        calibration_store.save(self._user_id, self._samples)
        self.status.setStyleSheet('color: #059669; font-size: 13px; font-weight: 600;')
        self.status.setText(
            f'✓  Saved {len(self._samples)} samples for “{self._user_id}”.  Train the model, then Proceed:')
        self.cmd_label.setText(
            f'& ../../app_video/.venv/Scripts/python.exe train_cheat_model.py --user {self._user_id}')
        self.cmd_label.setVisible(True)

    def _proceed(self):
        self._stop_worker()
        if self._on_proceed is not None:
            self._on_proceed()

    def _show_frame(self, qimg):
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def stop_all(self):
        if self._timer.isActive():
            self._timer.stop()
        self._stop_worker()


class DetectionView(QWidget):
    """Step 2: live tracking. The FRONT cam runs the personal cheat model
    (model + rule + 2s); the SIDE cam shows posture. Nothing touches MySQL
    unless a cheating episode is confirmed - then one row is written and the
    proctor can see it."""

    def __init__(self):
        super().__init__()
        self.front = None
        self.side = None
        self.logger = None
        self._count = 0

        layout = QVBoxLayout(self)
        heading = QLabel('Step 2 — Live Tracking')
        heading.setStyleSheet('font-weight: bold; font-size: 16px;')
        heading.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)

        feeds = QHBoxLayout()
        self.front_video = QLabel('Front camera')
        self.side_video = QLabel('Side camera')
        for lbl, cap in ((self.front_video, 'Front — Gaze + Head (detection)'),
                         (self.side_video, 'Side — Posture')):
            col = QVBoxLayout()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(320, 260)
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            lbl.setStyleSheet('background: #0f172a; color: #cbd5e1; border-radius: 8px;')
            caption = QLabel(cap)
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet('color: gray; font-size: 11px;')
            col.addWidget(lbl, stretch=1)
            col.addWidget(caption)
            feeds.addLayout(col)
        layout.addLayout(feeds, stretch=1)

        self.status = QLabel('Idle — press Start to begin tracking.')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet('color: gray;')
        layout.addWidget(self.status)

        self.button = QPushButton('Start Tracking')
        self.button.clicked.connect(self._toggle)
        layout.addWidget(self.button)

    def _toggle(self):
        if self.front is None:
            self._start()
        else:
            self.stop_all()

    def _start(self):
        session = self.window().session
        user_id = session.user_id if session else 'test_user'

        # Front cam: cheat detection (model + rule + 2s).
        self.front = FrontCamWorker(camera_index=0, session_user_id=user_id, detect=True)
        self.front.frame_ready.connect(self._show_front)
        self.front.cheat_detected.connect(self._on_cheat)

        # Side cam: posture feed only (log_to_db=False -> no continuous MySQL).
        self.side = SideCameraWorker(camera_index=1, session_user_id=user_id, log_to_db=False)
        self.side.frame_ready.connect(self._show_side)

        # MySQL writer - only used when a cheat actually fires.
        self.logger = CheatEventLogger()
        self.logger.start()

        self.front.start()
        self.side.start()

        self._count = 0
        self.status.setStyleSheet('color: #059669;')
        self.status.setText('● Tracking…  (0 flags)')
        self.button.setText('Stop Tracking')

    def _on_cheat(self, event):
        self.logger.enqueue(event)   # write to MySQL (only happens on a flag)
        self._count += 1
        ts = event['timestamp'].strftime('%H:%M:%S')
        self.status.setStyleSheet('color: #b91c1c; font-weight: 600;')
        self.status.setText(f'⚠  Cheating flagged at {ts}    (total this session: {self._count})')

    def _show_front(self, qimg):
        self.front_video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.front_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_side(self, qimg):
        self.side_video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.side_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def stop_all(self):
        if self.front is not None:
            self.front.stop()
            self.front.wait()
            self.front = None
        if self.side is not None:
            self.side.stop()
            self.side.wait()
            self.side = None
        if self.logger is not None:
            self.logger.stop()
            self.logger.wait()
            self.logger = None
        self.button.setText('Start Tracking')
        self.status.setStyleSheet('color: gray;')
        self.status.setText('Stopped.')


class ExamView(QWidget):
    """The cheat-detection flow in one place: calibrate (Step 1, saves JSON),
    then monitor live (Step 2, logs cheats to MySQL). Train in the terminal
    in between using the command the calibration screen shows."""

    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.calib = CalibrationView(on_proceed=self._go_detect)
        self.detect = DetectionView()
        self.stack.addWidget(self.calib)     # index 0 (shown first)
        self.stack.addWidget(self.detect)    # index 1
        root = QVBoxLayout(self)
        root.addWidget(self.stack)

    def _go_detect(self):
        self.stack.setCurrentIndex(1)

    def stop_all(self):
        self.calib.stop_all()
        self.detect.stop_all()


class AnalysisDashboard(QWidget):

    def __init__(self):
        super().__init__()
        # The one main view: Front + Side Cam, as a stepped flow -
        # calibration first, then live tracking.
        self.front_side_tab = ExamView()
        self.web_tab = None
        # --- MDI area: each analysis view is its own movable sub-window ---
        self.mdi = QMdiArea()

        # Every view, in the order its "open" button appears in the toolbar.
        view_list = [
            ('Front + Side Cam', self.front_side_tab),
        ]

        # One sub-window per view, plus one button that (re)opens that view.
        self.windows = {}          # name -> its sub-window, so we can reopen it
        toolbar = QHBoxLayout()
        for name, widget in view_list:
            sub = ViewWindow()      # closing only hides it, so it can come back
            sub.setWidget(widget)
            sub.setWindowTitle(name)
            self.mdi.addSubWindow(sub)
            sub.hide()              # start closed; open on demand via its button
            self.windows[name] = sub

            open_btn = QPushButton(name)
            # n=name gives each button its own name (needed inside a loop).
            open_btn.clicked.connect(lambda checked=False, n=name: self._open_window(n))
            toolbar.addWidget(open_btn)

        # The Web (exam) button opens FULL-SCREEN, separate from the MDI views.
        web_btn = QPushButton('Web')
        web_btn.clicked.connect(self._open_web)
        toolbar.addWidget(web_btn)

        # Arrange buttons, pushed to the right.
        toolbar.addStretch()
        tile_btn = QPushButton('Tile')
        tile_btn.clicked.connect(self.mdi.tileSubWindows)
        cascade_btn = QPushButton('Cascade')
        cascade_btn.clicked.connect(self.mdi.cascadeSubWindows)
        toolbar.addWidget(tile_btn)
        toolbar.addWidget(cascade_btn)

        # Power/exit button (far right). Quitting requires a password so a
        # student can't just close the exam window.
        exit_btn = QPushButton('X')   # ⏻ power symbol
        exit_btn.setToolTip('Exit AEye')
        exit_btn.setFixedWidth(40)
        exit_btn.setStyleSheet(
            'QPushButton { color: white; background-color: #c0392b;'
            ' font-size: 16px; font-weight: bold; border-radius: 4px;'
            ' padding: 4px; }'
            ' QPushButton:hover { background-color: #e74c3c; }'
        )
        exit_btn.clicked.connect(self._exit_with_password)
        toolbar.addWidget(exit_btn)

        # Toolbar on top, MDI area filling the rest.
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.mdi)

    def _open_window(self, name):
        # Open (or focus) a view, sized to ~80% of the current area so it fits
        # whatever screen size you're on.
        sub = self.windows[name]
        area = self.mdi.size()
        w, h = int(area.width() * 0.8), int(area.height() * 0.8)
        sub.resize(w, h)
        sub.move((area.width() - w) // 2, (area.height() - h) // 2)   # center it
        sub.show()
        self.mdi.setActiveSubWindow(sub)

    def _open_web(self):
        if self.web_tab is None:
            from web_tab import WebTab
            self.web_tab = WebTab()
        self.web_tab.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.web_tab.showFullScreen()

    def _exit_with_password(self):
        # Ask for the exit password. Only the correct password quits the app;
        # QApplication.quit() triggers the aboutToQuit cleanup (keyboard hook
        # removed, workers stopped) wired up in main.py.
        password, ok = QInputDialog.getText(
            self, 'Exit AEye', 'Enter password to quit:',
            QLineEdit.EchoMode.Password,
        )
        if not ok:
            return   # user cancelled the dialog
        if password == 'quit':
            QApplication.quit()
        else:
            QMessageBox.warning(self, 'Incorrect Password', 'Incorrect password.')

    def stop_all(self):
        self.front_side_tab.stop_all()

class ProctorView(QWidget):
    """Proctor Mode: the cheating events students' sessions wrote to MySQL,
    newest first. Refresh to pull the latest; filter by student ID."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        heading = QLabel('Proctor — Cheating Alerts')
        heading.setStyleSheet('font-weight: bold; font-size: 18px;')
        layout.addWidget(heading)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('Student:'))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText('filter by student ID (blank = everyone)')
        self.filter_input.returnPressed.connect(self.refresh)
        controls.addWidget(self.filter_input)
        refresh_btn = QPushButton('Refresh')
        refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(refresh_btn)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Student', 'Detected at'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table)

        self.status = QLabel('')
        self.status.setStyleSheet('color: gray;')
        layout.addWidget(self.status)

        self.refresh()

    def refresh(self):
        name = self.filter_input.text().strip()
        conn = get_connection()
        if conn is None:
            self.status.setText('Could not connect to the database.')
            return
        try:
            cur = conn.cursor()
            if name:
                cur.execute(
                    'SELECT session_user_id, detected_at FROM cheating_events '
                    'WHERE session_user_id = %s ORDER BY detected_at DESC', (name,))
            else:
                cur.execute(
                    'SELECT session_user_id, detected_at FROM cheating_events '
                    'ORDER BY detected_at DESC')
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as exc:
            self.status.setText(f'Query failed: {exc}')
            return

        self.table.setRowCount(len(rows))
        for r, (user, ts) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(user)))
            self.table.setItem(r, 1, QTableWidgetItem(str(ts)))
        self.status.setText(f'{len(rows)} alert(s).')
