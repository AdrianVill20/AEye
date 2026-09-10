from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QFrame, QComboBox, QMdiArea, QMdiSubWindow, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QInputDialog, QProgressBar, QCheckBox
from posture_worker import SideCameraWorker
from front_cam_worker import FrontCamWorker
from front_cam_logger import FrontCamLogWriter
import calibration_store
from cheat_logger import CheatEventLogger
from gaze_graph import GazeGraph
from db_config import get_connection
from phone_camera import PhoneCameraDialog


def _open_phone_setup(parent, combo):
    """Open the Iriun phone-camera dialog; on confirm, point `combo` at the
    chosen camera index. Leaves `combo` untouched if the user cancels."""
    dlg = PhoneCameraDialog(parent)
    dlg.exec()
    if dlg.selected_index is not None:
        text = str(dlg.selected_index)
        if combo.findText(text) < 0:
            combo.addItem(text)
        combo.setCurrentText(text)


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

class ViewWindow(QMdiSubWindow):
    """An MDI sub-window that hides instead of closing, so the toolbar's
    buttons can reopen it later. (A normal close would remove it for good.)"""

    def closeEvent(self, event):
        event.ignore()   # don't actually close...
        self.hide()      # ...just hide it, so show() can bring it back


class TrainWorker(QThread):
    """Trains a student's model off the UI thread, so the app never freezes
    while scikit-learn imports and fits. Emits done(result) or failed(msg)."""

    done = Signal(object)    # result dict from train_cheat_model.train()
    failed = Signal(str)     # error message

    def __init__(self, user, parent=None):
        super().__init__(parent)
        self.user = user

    def run(self):
        try:
            from train_cheat_model import train   # heavy imports happen here
            self.done.emit(train(self.user))
        except Exception as exc:
            self.failed.emit(str(exc))


class ReadingWindow(QWidget):
    """Full-screen reading page shown during calibration: the passage plus a
    live countdown and progress bar. The student reads it while the camera
    records in the background. Closes when the timer ends (or Cancel)."""

    def __init__(self, passage_text, seconds, on_cancel=None):
        super().__init__()
        self.setWindowTitle('AEye Calibration')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setStyleSheet('background: #ffffff;')
        self._on_cancel = on_cancel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 50, 80, 40)
        layout.setSpacing(18)

        heading = QLabel('Read the passage below')
        heading.setStyleSheet('font-size: 22px; font-weight: bold; color: #111;')
        layout.addWidget(heading)

        passage = QLabel(passage_text)
        passage.setWordWrap(True)
        passage.setAlignment(Qt.AlignTop)
        passage.setStyleSheet('font-size: 28px; color: #222;')
        layout.addWidget(passage, stretch=1)

        self.countdown = QLabel(f'{seconds}s left')
        self.countdown.setAlignment(Qt.AlignCenter)
        self.countdown.setStyleSheet('font-size: 18px; color: #333;')
        layout.addWidget(self.countdown)

        self.progress = QProgressBar()
        self.progress.setRange(0, seconds)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        row.addStretch(1)
        exit_btn = QPushButton('Exit Calibration')
        exit_btn.setStyleSheet(
            'color: #a00000; font-weight: bold; padding: 6px 16px; font-size: 15px;')
        exit_btn.clicked.connect(self._ask_exit)
        row.addWidget(exit_btn)
        layout.addLayout(row)

    def keyPressEvent(self, event):
        # Esc is the other way out - this window is full screen, so there is no
        # title bar to close it with.
        if event.key() == Qt.Key_Escape:
            self._ask_exit()
        else:
            super().keyPressEvent(event)

    def _ask_exit(self):
        # Leaving throws the whole recording away, so ask first - a misclick
        # here would cost the student the full two minutes.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle('Exit calibration?')
        box.setText('Nothing will be saved and no model will be trained. Exit anyway?')
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        # This window forces a white background on everything under it, so the
        # dialog needs its own colours - on a dark Windows theme the default
        # text is white and comes out invisible.
        box.setStyleSheet(
            'QMessageBox { background: #ffffff; }'
            'QLabel { color: #111111; font-size: 14px; }'
            'QPushButton { color: #111111; background: #e8e8e8;'
            ' border: 1px solid #999999; padding: 5px 20px; }'
            'QPushButton:hover { background: #d8d8d8; }')
        if box.exec() == QMessageBox.Yes:
            self.close()

    def set_progress(self, elapsed, remaining):
        self.progress.setValue(elapsed)
        self.countdown.setText(f'{max(remaining, 0)}s left')

    def closeEvent(self, event):
        cb = self._on_cancel
        self._on_cancel = None   # fire the cancel callback at most once
        if cb:
            cb()
        super().closeEvent(event)


class CalibrationView(QWidget):
    """Step 1: record a short sample of the student's NORMAL behaviour while
    they read a passage in a full-screen window, and save it to JSON
    (calibration_data/). That JSON is the seed the "Train" button turns into
    the student's personal model."""

    def __init__(self, on_proceed=None):
        super().__init__()
        self._on_proceed = on_proceed
        self.worker = None
        self._trainer = None
        self._reader = None
        self._samples = []
        self._user_id = None
        self._remaining = 0
        self._elapsed = 0

        card_l = QVBoxLayout(self)
        card_l.setContentsMargins(20, 20, 20, 20)
        card_l.setSpacing(10)

        heading = QLabel('Calibration')
        heading.setStyleSheet('font-size: 18px; font-weight: bold;')
        card_l.addWidget(heading)

        instructions = QLabel(
            'Pick your camera and duration, then press Start Calibration. A '
            'full-screen reading page opens - read it at your normal pace while '
            'the camera records. It closes on its own when the timer ends; then '
            'press Train Model, then Proceed.')
        instructions.setWordWrap(True)
        card_l.addWidget(instructions)

        # The passage the student reads - shown in the full-screen reading
        # window that opens on Start, not on this setup screen.
        self._passage_text = (
            'Effective studying is less about the number of hours spent and more '
            'about the quality of attention during those hours. When you sit down '
            'to review, remove the distractions within reach, decide on a single '
            'goal for the session, and work in focused blocks separated by short '
            'breaks. As you read, your eyes move across the lines in small jumps, '
            'pausing briefly to take in groups of words. Try to keep your attention '
            'on the text in front of you, the way you would during a real '
            'examination. If your mind wanders, gently bring it back to the '
            'sentence you were on.')

        body = QHBoxLayout()
        body.addStretch(1)
        preview_col = QVBoxLayout()
        preview_col.setSpacing(4)
        self.video = QLabel('Camera preview')
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(360, 280)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet('background: #222; color: #ddd; border: 1px solid #999;')
        caption = QLabel('Camera')
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet('color: gray;')
        preview_col.addWidget(self.video, stretch=1)
        preview_col.addWidget(caption)
        body.addLayout(preview_col, stretch=2)
        body.addStretch(1)
        card_l.addLayout(body, stretch=1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        card_l.addWidget(self.progress)

        self.status = QLabel('Press Start Calibration to begin.')
        card_l.addWidget(self.status)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('Duration (sec):'))
        self.duration_box = QComboBox()
        self.duration_box.addItems(['30', '60', '120'])
        self.duration_box.setCurrentText('120')
        controls.addWidget(self.duration_box)
        controls.addWidget(QLabel('Camera:'))
        self.cam_box = QComboBox()
        self.cam_box.addItems(['0', '1', '2', '3'])
        self.cam_box.setCurrentText('0')
        controls.addWidget(self.cam_box)
        phone_btn = QPushButton('📱 Set up phone camera')
        phone_btn.setToolTip('Use your phone as the camera via Iriun Webcam')
        phone_btn.clicked.connect(lambda: _open_phone_setup(self, self.cam_box))
        controls.addWidget(phone_btn)
        controls.addStretch(1)
        self.start_btn = QPushButton('Start Calibration')
        self.start_btn.clicked.connect(self._start)
        controls.addWidget(self.start_btn)
        self.train_btn = QPushButton('Train Model')
        self.train_btn.setEnabled(False)   # enabled once calibration is saved
        self.train_btn.clicked.connect(self._train)
        controls.addWidget(self.train_btn)
        self.proceed_btn = QPushButton('Proceed to Monitoring')
        self.proceed_btn.clicked.connect(self._proceed)
        controls.addWidget(self.proceed_btn)
        card_l.addLayout(controls)

        self.cmd_label = QLabel('')
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cmd_label.setVisible(False)
        card_l.addWidget(self.cmd_label)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    def _start(self):
        if self.worker is not None:
            return
        session = self.window().session
        self._user_id = session.user_id if session else 'test_user'
        self._samples = []
        # detect=False -> record only. We collect the raw features via
        # features_ready and save them to JSON; nothing is written to MySQL.
        cam = int(self.cam_box.currentText())
        self.worker = FrontCamWorker(camera_index=cam, session_user_id=self._user_id, detect=False)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.features_ready.connect(self._collect)
        self.worker.start()
        self._remaining = int(self.duration_box.currentText())
        self._elapsed = 0
        self.progress.setRange(0, self._remaining)
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.duration_box.setEnabled(False)
        self.cam_box.setEnabled(False)
        self.cmd_label.setVisible(False)
        self.status.setStyleSheet('')
        self.status.setText(f'Recording... {self._remaining}s left (reading window is open)')
        # Pop up the full-screen reading page (passage + countdown + progress).
        self._reader = ReadingWindow(self._passage_text, self._remaining,
                                     on_cancel=self._cancel_reading)
        self._reader.showFullScreen()
        self._timer.start()

    def _collect(self, feats):
        self._samples.append(feats)

    def _tick(self):
        self._elapsed += 1
        self.progress.setValue(self._elapsed)
        left = self._remaining - self._elapsed
        self.status.setText(f'Recording... {max(left, 0)}s left (reading window is open)')
        if self._reader is not None:
            self._reader.set_progress(self._elapsed, left)
        if self._elapsed >= self._remaining:
            self._finish()

    def _finish(self):
        self._timer.stop()
        self._stop_worker()
        self._close_reader()
        self.start_btn.setEnabled(True)
        self.start_btn.setText('Re-record')
        self.duration_box.setEnabled(True)
        self.cam_box.setEnabled(True)
        if len(self._samples) < 100:
            self.status.setStyleSheet('color: #a00000;')
            self.status.setText(
                f'Only {len(self._samples)} samples captured. Keep your face in view and record again.')
            return
        calibration_store.save(self._user_id, self._samples)
        self.status.setStyleSheet('color: #006600;')
        self.status.setText(
            f'Saved {len(self._samples)} samples for {self._user_id}. Now press Train Model.')
        self.train_btn.setEnabled(True)
        self.cmd_label.setVisible(False)

    def _train(self):
        if not self._user_id:
            session = self.window().session
            self._user_id = session.user_id if session else 'test_user'
        self.train_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status.setStyleSheet('')
        self.status.setText(f'Training model for {self._user_id}...')
        self._trainer = TrainWorker(self._user_id)
        self._trainer.done.connect(self._train_done)
        self._trainer.failed.connect(self._train_failed)
        self._trainer.start()

    def _train_done(self, result):
        self.train_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        pct = 100 * result['flagged'] / result['samples'] if result['samples'] else 0
        self.status.setStyleSheet('color: #006600;')
        self.status.setText(
            f'Model trained on {result["samples"]} samples '
            f'({pct:.0f}% flagged). You can now proceed.')

    def _train_failed(self, msg):
        self.train_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText(f'Training failed: {msg}')

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

    def _close_reader(self):
        if self._reader is not None:
            self._reader._on_cancel = None   # don't fire cancel on a programmatic close
            self._reader.close()
            self._reader = None

    def _cancel_reading(self):
        # The student exited early: stop and reset. Nothing is written - the
        # JSON is only saved in _finish(), so a cancelled run leaves no file
        # and no model behind.
        if self._timer.isActive():
            self._timer.stop()
        self._stop_worker()
        self._reader = None            # it is already closing itself
        self._samples = []             # drop the partial recording
        self.progress.setValue(0)
        self.start_btn.setEnabled(True)
        self.start_btn.setText('Start Calibration')
        self.duration_box.setEnabled(True)
        self.cam_box.setEnabled(True)
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText('Calibration cancelled - nothing was saved.')

    def stop_all(self):
        if self._timer.isActive():
            self._timer.stop()
        self._stop_worker()
        self._close_reader()
        if self._trainer is not None and self._trainer.isRunning():
            self._trainer.wait()


class DetectionView(QWidget):
    """Step 2: live tracking. The FRONT cam runs the personal cheat model
    (model + rule + 2s); the SIDE cam shows posture. Each confirmed cheating
    episode becomes one row in cheating_events (with a screenshot), which the
    proctor sees."""

    def __init__(self, on_back=None):
        super().__init__()
        self._on_back = on_back
        self.front = None
        self.side = None
        self.front_log = None
        self.logger = None
        self._count = 0

        layout = QVBoxLayout(self)
        heading = QLabel('Live Tracking')
        heading.setStyleSheet('font-weight: bold; font-size: 18px;')
        layout.addWidget(heading)

        # Camera-index pickers (which webcam each feed opens).
        cams_row = QHBoxLayout()
        cams_row.addStretch(1)
        cams_row.addWidget(QLabel('Front cam:'))
        self.front_box = QComboBox()
        self.front_box.addItems(['0', '1', '2', '3'])
        self.front_box.setCurrentText('0')
        cams_row.addWidget(self.front_box)
        cams_row.addSpacing(16)
        cams_row.addWidget(QLabel('Side cam:'))
        self.side_box = QComboBox()
        self.side_box.addItems(['0', '1', '2', '3'])
        self.side_box.setCurrentText('1')
        cams_row.addWidget(self.side_box)
        phone_btn = QPushButton('📱 Set up phone camera')
        phone_btn.setToolTip('Use your phone as the side camera via Iriun Webcam')
        phone_btn.clicked.connect(lambda: _open_phone_setup(self, self.side_box))
        cams_row.addWidget(phone_btn)
        cams_row.addStretch(1)

        # Checkbox to hide the graph (some proctors only want the feeds).
        self.graph_check = QCheckBox('Show graph')
        self.graph_check.setChecked(True)
        self.graph_check.toggled.connect(self._show_graph)
        cams_row.addWidget(self.graph_check)
        cams_row.addStretch(1)
        layout.addLayout(cams_row)

        # Splitters, so the proctor can drag the dividers: one between the two
        # camera feeds, one between the feeds and the graph under them.
        feeds = QSplitter(Qt.Horizontal)
        feeds.setChildrenCollapsible(False)   # a feed should never vanish entirely
        self.front_video = QLabel('Front camera')
        self.side_video = QLabel('Side camera')
        for lbl, cap in ((self.front_video, 'Front - Gaze + Head (detection)'),
                         (self.side_video, 'Side - Posture')):
            col_box = QWidget()
            col = QVBoxLayout(col_box)
            col.setContentsMargins(0, 0, 0, 0)
            lbl.setAlignment(Qt.AlignCenter)
            # Small minimum, otherwise the splitter cannot shrink a feed.
            lbl.setMinimumSize(160, 120)
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            lbl.setStyleSheet('background: #0f172a; color: #cbd5e1; border-radius: 8px;')
            caption = QLabel(cap)
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet('color: gray; font-size: 11px;')
            col.addWidget(lbl, stretch=1)
            col.addWidget(caption)
            feeds.addWidget(col_box)

        self.graph_box = QWidget()
        graph_col = QVBoxLayout(self.graph_box)
        graph_col.setContentsMargins(0, 0, 0, 0)
        graph_caption = QLabel('Gaze / head values (z-score, dashed band = normal)')
        graph_caption.setStyleSheet('color: gray; font-size: 11px;')
        graph_col.addWidget(graph_caption)
        self.graph = GazeGraph()
        graph_col.addWidget(self.graph, stretch=1)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(feeds)
        split.addWidget(self.graph_box)
        split.setStretchFactor(0, 3)   # cameras get most of the height by default
        split.setStretchFactor(1, 1)
        layout.addWidget(split, stretch=1)

        self.status = QLabel('Idle. Press Start to begin tracking.')
        self.status.setStyleSheet('color: gray;')
        layout.addWidget(self.status)

        btn_row = QHBoxLayout()
        self.back_btn = QPushButton('Back to Calibration')
        self.back_btn.clicked.connect(self._back)
        btn_row.addWidget(self.back_btn)
        self.button = QPushButton('Start Tracking')
        self.button.clicked.connect(self._toggle)
        btn_row.addWidget(self.button, stretch=1)
        layout.addLayout(btn_row)

    def _back(self):
        # Cameras must be released before the calibration screen opens one.
        self.stop_all()
        if self._on_back is not None:
            self._on_back()

    def _show_graph(self, on):
        self.graph_box.setVisible(on)

    def _toggle(self):
        if self.front is None:
            self._start()
        else:
            self.stop_all()

    def _start(self):
        session = self.window().session
        user_id = session.user_id if session else 'test_user'

        front_idx = int(self.front_box.currentText())
        side_idx = int(self.side_box.currentText())

        # Front cam: cheat detection (model + rule + 2s).
        self.front = FrontCamWorker(camera_index=front_idx, session_user_id=user_id, detect=True)
        self.front.frame_ready.connect(self._show_front)
        self.front.cheat_detected.connect(self._on_cheat)

        self.graph.set_user(user_id)
        self.front.features_ready.connect(self.graph.on_features)

        # Front cam rows (gaze + head pose, every frame) go to gaze_logs.
        self.front_log = FrontCamLogWriter()
        self.front_log.start()
        self.front.record_ready.connect(self.front_log.enqueue)

        # Side cam: posture feed. log_to_db=True writes joint coords to
        # posture_logs (2 rows/sec) so posture has history to train on.
        self.side = SideCameraWorker(camera_index=side_idx, session_user_id=user_id, log_to_db=True)
        self.side.frame_ready.connect(self._show_side)

        # MySQL writer - only used when a cheat actually fires.
        self.logger = CheatEventLogger()
        self.logger.start()

        self.front.start()
        self.side.start()

        self._count = 0
        self.front_box.setEnabled(False)   # can't change cameras mid-session
        self.side_box.setEnabled(False)
        self.status.setStyleSheet('color: #006600;')
        self.status.setText('Tracking... (0 flags)')
        self.button.setText('Stop Tracking')

    def _on_cheat(self, event):
        if self.logger is None:
            return   # arrived after tracking stopped; the logger closed the episode
        self.logger.enqueue(event)   # write to MySQL (only happens on a flag)
        if event['kind'] == 'start':
            self._count += 1
            ts = event['started_at'].strftime('%H:%M:%S')
            self.status.setStyleSheet('color: #a00000;')
            self.status.setText(f"Cheating flagged at {ts} - {event['reason']} (total: {self._count})")

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
        if self.front_log is not None:
            self.front_log.stop()
            self.front_log.wait()
            self.front_log = None
        if self.logger is not None:
            self.logger.stop()
            self.logger.wait()
            self.logger = None
        self.front_box.setEnabled(True)
        self.side_box.setEnabled(True)
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
        self.detect = DetectionView(on_back=self._go_calib)
        self.stack.addWidget(self.calib)     # index 0 (shown first)
        self.stack.addWidget(self.detect)    # index 1
        root = QVBoxLayout(self)
        root.addWidget(self.stack)

    def _go_detect(self):
        self.stack.setCurrentIndex(1)

    def _go_calib(self):
        self.stack.setCurrentIndex(0)

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
        exit_btn = QPushButton('X')   # exit / power button
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
    """Proctor Mode: the cheating episodes students' sessions wrote to MySQL,
    newest first, plus the screenshot of the one you click. Refreshes itself
    every 3 seconds; filter by student ID."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        heading = QLabel('Proctor - Cheating Alerts')
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

        # Alerts table on the left, screenshot of the clicked alert on the right.
        body = QHBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Student', 'Started', 'Ended', 'Reason'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellClicked.connect(self._show_screenshot)
        body.addWidget(self.table, stretch=3)
        self.preview = QLabel('Click an alert to see its screenshot.')
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setStyleSheet('background: #0f172a; color: #cbd5e1;')
        body.addWidget(self.preview, stretch=2)
        layout.addLayout(body)

        self.status = QLabel('')
        self.status.setStyleSheet('color: gray;')
        layout.addWidget(self.status)

        # Pull new alerts every 3 seconds while this screen is open, so a flag
        # on the student side shows up here without pressing Refresh.
        self._paths = []   # screenshot path of each table row
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)

    def refresh(self):
        name = self.filter_input.text().strip()
        conn = get_connection()
        if conn is None:
            self.status.setText('Could not connect to the database.')
            return
        try:
            cur = conn.cursor()
            sql = ('SELECT session_user_id, started_at, ended_at, reason, screenshot_path '
                   'FROM cheating_events ')
            if name:
                cur.execute(sql + 'WHERE session_user_id = %s ORDER BY started_at DESC', (name,))
            else:
                cur.execute(sql + 'ORDER BY started_at DESC')
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as exc:
            self.status.setText(f'Query failed: {exc}')
            return

        self.table.setRowCount(len(rows))
        self._paths = []
        for r, (user, started, ended, reason, path) in enumerate(rows):
            ended_text = ended.strftime('%H:%M:%S') if ended else 'ongoing'
            self.table.setItem(r, 0, QTableWidgetItem(str(user)))
            self.table.setItem(r, 1, QTableWidgetItem(started.strftime('%Y-%m-%d %H:%M:%S')))
            self.table.setItem(r, 2, QTableWidgetItem(ended_text))
            self.table.setItem(r, 3, QTableWidgetItem(reason or ''))
            self._paths.append(path)
        self.status.setText(f'{len(rows)} alert(s). Updates every 3 seconds.')

    def _show_screenshot(self, row, col):
        path = self._paths[row]
        full = Path(__file__).resolve().parent / path if path else None
        if full is None or not full.exists():
            self.preview.setText('No screenshot for this alert.')
            return
        self.preview.setPixmap(QPixmap(str(full)).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.timer.start(3000)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()
