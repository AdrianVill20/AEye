import sys
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QTabWidget, QFrame
REPO_ROOT = Path(__file__).resolve().parent.parent
from gaze_worker import GazeWorker
from posture_worker import SideCameraWorker
from headpose_worker import HeadPoseWorker

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
        self.feed_area.setMinimumSize(640, 480)
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

class AnalysisDashboard(QTabWidget):

    def __init__(self):
        super().__init__()
        self.eye_tab = AnalysisTab('Eye Tracking', 'Front camera — dlib gaze (Revised_Gaze)', ['Direction', 'Gaze ratio', 'Blink'], 'Demetillo')
        self.eye_video = QLabel(alignment=Qt.AlignCenter)
        self.eye_tab.set_feed_widget(self.eye_video)
        self.eye_tab.enable_controls(self.start_gaze, self.stop_gaze)
        self.gaze_worker = None
        self.head_tab = AnalysisTab('Head Pose', 'Front camera — MediaPipe face mesh', ['Direction', 'Yaw', 'Pitch', 'Roll', 'Landmarks detected (/478)'], 'Demetillo')
        self.head_video = QLabel(alignment=Qt.AlignCenter)
        self.head_tab.set_feed_widget(self.head_video)
        self.head_tab.enable_controls(self.start_headpose, self.stop_headpose)
        self.head_worker = None
        self.posture_tab = AnalysisTab('Posture', 'Side camera — MediaPipe pose', ['Left shoulder', 'Right shoulder', 'Left wrist', 'Right wrist', 'Landmarks detected (/33)'], 'Ybañez')
        self.posture_video = QLabel(alignment=Qt.AlignCenter)
        self.posture_tab.set_feed_widget(self.posture_video)
        self.posture_tab.enable_controls(self.start_posture, self.stop_posture)
        self.side_worker = None
        self.addTab(self.eye_tab, 'Eye Tracking')
        self.addTab(self.head_tab, 'Head Pose')
        self.addTab(self.posture_tab, 'Posture')

    def start_gaze(self):
        if self.gaze_worker is None:
            self.gaze_worker = GazeWorker(camera_index=0)
            self.gaze_worker.frame_ready.connect(self._show_gaze_frame)
            self.gaze_worker.stats_ready.connect(self.eye_tab.update_stats)
            self.gaze_worker.start()

    def stop_gaze(self):
        if self.gaze_worker is not None:
            self.gaze_worker.stop()
            self.gaze_worker.wait()
            self.gaze_worker = None

    def _show_gaze_frame(self, qimg):
        self.eye_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.eye_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_posture(self):
        if self.side_worker is None:
            self.side_worker = SideCameraWorker(camera_index=0)
            self.side_worker.frame_ready.connect(self._show_posture_frame)
            self.side_worker.stats_ready.connect(self.posture_tab.update_stats)
            self.side_worker.start()

    def stop_posture(self):
        if self.side_worker is not None:
            self.side_worker.stop()
            self.side_worker.wait()
            self.side_worker = None

    def _show_posture_frame(self, qimg):
        self.posture_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.posture_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_headpose(self):
        if self.head_worker is None:
            self.head_worker = HeadPoseWorker(camera_index=0)
            self.head_worker.frame_ready.connect(self._show_head_frame)
            self.head_worker.stats_ready.connect(self.head_tab.update_stats)
            self.head_worker.start()

    def stop_headpose(self):
        if self.head_worker is not None:
            self.head_worker.stop()
            self.head_worker.wait()
            self.head_worker = None

    def _show_head_frame(self, qimg):
        self.head_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.head_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def stop_all(self):
        self.stop_gaze()
        self.stop_posture()
        self.stop_headpose()

class ProctorView(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel('Proctor Mode — monitoring dashboard goes here (Module 3)')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
