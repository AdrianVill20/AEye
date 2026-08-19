import sys
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QFrame, QComboBox, QMdiArea, QMdiSubWindow, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QStackedWidget
REPO_ROOT = Path(__file__).resolve().parent.parent
from gaze_worker import GazeWorker
from posture_worker import SideCameraWorker
from headpose_worker import HeadPoseWorker
from auth import create_user, get_all_sessions
from gaze_logger import GazeLogWriter
from posture_logger import PostureLogWriter

class LoginView(QWidget):

    def __init__(self, on_login):
        super().__init__()
        self._on_login = on_login

        self.stack = QStackedWidget()
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.stack)

        # --- Page 0: Sign In ---
        self._login_page = QWidget()
        lp = QVBoxLayout(self._login_page)
        lp.setAlignment(Qt.AlignCenter)
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
        create_btn = QPushButton("Don't have an account? Create one")
        create_btn.setFlat(True)
        create_btn.setStyleSheet('color: #555;')
        create_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        lp.addWidget(title)
        lp.addWidget(self.id_input, alignment=Qt.AlignCenter)
        lp.addWidget(self.pw_input, alignment=Qt.AlignCenter)
        lp.addLayout(roles)
        lp.addWidget(self.error_label)
        lp.addWidget(sign_in_btn, alignment=Qt.AlignCenter)
        lp.addWidget(create_btn, alignment=Qt.AlignCenter)

        # --- Page 1: Create Account ---
        self._reg_page = QWidget()
        rp = QVBoxLayout(self._reg_page)
        rp.setAlignment(Qt.AlignCenter)
        reg_title = QLabel('AEye — Create Account')
        reg_title.setStyleSheet('font-size: 18px; font-weight: bold;')
        reg_title.setAlignment(Qt.AlignCenter)
        self.reg_id = QLineEdit()
        self.reg_id.setPlaceholderText('School-issued ID')
        self.reg_id.setMaximumWidth(300)
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText('Full name')
        self.reg_name.setMaximumWidth(300)
        self.reg_pw = QLineEdit()
        self.reg_pw.setPlaceholderText('Password')
        self.reg_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_pw.setMaximumWidth(300)
        self.reg_student_radio = QRadioButton('Student')
        self.reg_proctor_radio = QRadioButton('Proctor')
        self.reg_student_radio.setChecked(True)
        reg_role_group = QButtonGroup(self)
        reg_role_group.addButton(self.reg_student_radio)
        reg_role_group.addButton(self.reg_proctor_radio)
        reg_roles = QHBoxLayout()
        reg_roles.setAlignment(Qt.AlignCenter)
        reg_roles.addWidget(self.reg_student_radio)
        reg_roles.addWidget(self.reg_proctor_radio)
        self.reg_error = QLabel('')
        self.reg_error.setStyleSheet('color: red;')
        self.reg_error.setAlignment(Qt.AlignCenter)
        self.reg_success = QLabel('')
        self.reg_success.setStyleSheet('color: green;')
        self.reg_success.setAlignment(Qt.AlignCenter)
        register_btn = QPushButton('Create Account')
        register_btn.setMaximumWidth(300)
        register_btn.clicked.connect(self._handle_register)
        back_btn = QPushButton('Back to Sign In')
        back_btn.setFlat(True)
        back_btn.setStyleSheet('color: #555;')
        back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        rp.addWidget(reg_title)
        rp.addWidget(self.reg_id, alignment=Qt.AlignCenter)
        rp.addWidget(self.reg_name, alignment=Qt.AlignCenter)
        rp.addWidget(self.reg_pw, alignment=Qt.AlignCenter)
        rp.addLayout(reg_roles)
        rp.addWidget(self.reg_error)
        rp.addWidget(self.reg_success)
        rp.addWidget(register_btn, alignment=Qt.AlignCenter)
        rp.addWidget(back_btn, alignment=Qt.AlignCenter)

        self.stack.addWidget(self._login_page)
        self.stack.addWidget(self._reg_page)

    def _handle_sign_in(self):
        role = 'student' if self.student_radio.isChecked() else 'proctor'
        self._on_login(self.id_input.text(), self.pw_input.text(), role)

    def _handle_register(self):
        self.reg_error.setText('')
        self.reg_success.setText('')
        role = 'student' if self.reg_student_radio.isChecked() else 'proctor'
        err = create_user(self.reg_id.text(), self.reg_pw.text(), self.reg_name.text(), role)
        if err:
            self.reg_error.setText(err)
        else:
            self.reg_success.setText('Account created! You can now sign in.')
            self.reg_id.clear()
            self.reg_name.clear()
            self.reg_pw.clear()

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

    def __init__(self, title, worker_class, default_index=0):
        super().__init__()
        self.worker_class = worker_class   # GazeWorker or SideCameraWorker
        self.worker = None                 # the running worker, created on Start

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
        session_id = session.session_id if session else 0

        self.worker = self.worker_class(camera_index=camera_index, session_user_id=user_id)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.stats_ready.connect(self._show_status)
        self.worker.stats_ready.connect(self._log_posture)
        self.worker.start()

        self._posture_logger = PostureLogWriter()
        self._posture_logger._session_id = session_id
        self._posture_logger._session_user_id = user_id
        self._posture_logger.start()

        self.index_box.setEnabled(False)   # can't change camera while running
        self.button.setText('Stop')

    def _log_posture(self, stats):
        from datetime import datetime
        if not hasattr(self, '_posture_logger') or self._posture_logger is None:
            return
        try:
            l_sh = stats.get('Left shoulder', '')
            r_sh = stats.get('Right shoulder', '')
            l_wr = stats.get('Left wrist', '')
            r_wr = stats.get('Right wrist', '')
            if '--' in (l_sh, r_sh, l_wr, r_wr):
                return
            l_sh_x, l_sh_y = [float(v) for v in l_sh.split(',')]
            r_sh_x, r_sh_y = [float(v) for v in r_sh.split(',')]
            l_wr_x, l_wr_y = [float(v) for v in l_wr.split(',')]
            r_wr_x, r_wr_y = [float(v) for v in r_wr.split(',')]
        except (ValueError, AttributeError):
            return
        record = (
            self._posture_logger._session_id,
            self._posture_logger._session_user_id,
            datetime.now(),
            l_sh_x, l_sh_y, r_sh_x, r_sh_y,
            l_wr_x, l_wr_y, r_wr_x, r_wr_y,
        )
        self._posture_logger.enqueue(record)

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        if hasattr(self, '_posture_logger') and self._posture_logger is not None:
            self._posture_logger.stop()
            self._posture_logger.wait()
            self._posture_logger = None
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


class AnalysisDashboard(QWidget):

    def __init__(self):
        super().__init__()
        self.eye_tab = AnalysisTab('Eye Tracking', 'Front camera — MediaPipe iris + eye openness', ['Direction', 'Gaze ratio', 'Blink'], 'Demetillo')
        self.eye_video = QLabel(alignment=Qt.AlignCenter)
        self.eye_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.eye_tab.set_feed_widget(self.eye_video)
        self.eye_tab.enable_controls(self.start_gaze, self.stop_gaze)
        self.gaze_worker = None
        self.head_tab = AnalysisTab('Head Pose', 'Front camera — MediaPipe face mesh', ['Direction', 'Yaw', 'Pitch', 'Roll', 'Landmarks detected (/478)'], 'Demetillo')
        self.head_video = QLabel(alignment=Qt.AlignCenter)
        self.head_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.head_tab.set_feed_widget(self.head_video)
        self.head_tab.enable_controls(self.start_headpose, self.stop_headpose)
        self.head_worker = None
        self.posture_tab = PostureTab()
        self.gaze_posture_tab = GazePostureTab()
        self.web_tab = None
        # --- MDI area: each analysis view is its own movable sub-window ---
        self.mdi = QMdiArea()

        # Every view, in the order its "open" button appears in the toolbar.
        view_list = [
            ('Eye Tracking', self.eye_tab),
            ('Head Pose', self.head_tab),
            ('Head + Posture', self.gaze_posture_tab),
            ('Posture', self.posture_tab),
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

        self.user_label = QLabel('')
        self.user_label.setStyleSheet('color: #333; font-weight: bold; padding-right: 8px;')
        toolbar.addWidget(self.user_label)

        logout_btn = QPushButton('Logout')
        logout_btn.setStyleSheet('background-color: #c0392b; color: white;')
        logout_btn.clicked.connect(self._logout)
        toolbar.addWidget(logout_btn)

        tile_btn = QPushButton('Tile')
        tile_btn.clicked.connect(self.mdi.tileSubWindows)
        cascade_btn = QPushButton('Cascade')
        cascade_btn.clicked.connect(self.mdi.cascadeSubWindows)
        toolbar.addWidget(tile_btn)
        toolbar.addWidget(cascade_btn)

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

    def start_gaze(self):
        if self.gaze_worker is None:
            session = self.window().session
            current_user = session.user_id if session else 'test_user'
            session_id = session.session_id if session else 0

            self.gaze_worker = GazeWorker(camera_index=0, session_user_id=current_user)
            self.gaze_worker.frame_ready.connect(self._show_gaze_frame)
            self.gaze_worker.stats_ready.connect(self.eye_tab.update_stats)
            self.gaze_worker.stats_ready.connect(self._log_gaze)
            self.gaze_worker.start()

            self._gaze_logger = GazeLogWriter()
            self._gaze_logger._session_id = session_id
            self._gaze_logger._session_user_id = current_user
            self._gaze_logger.start()

    def _log_gaze(self, stats):
        from datetime import datetime
        if not hasattr(self, '_gaze_logger') or self._gaze_logger is None:
            return
        direction = stats.get('Direction', '--')
        if direction == '--':
            return
        ratio_str = stats.get('Gaze ratio', '')
        is_blink = 1 if stats.get('Blink') == 'BLINKING' else 0
        try:
            ratio = float(ratio_str.split('h=')[1].split(' ')[0]) if 'h=' in ratio_str else 0.0
        except (IndexError, ValueError):
            ratio = 0.0
        record = (
            self._gaze_logger._session_id,
            self._gaze_logger._session_user_id,
            datetime.now(),
            direction,
            ratio,
            is_blink,
        )
        self._gaze_logger.enqueue(record)

    def stop_gaze(self):
        if self.gaze_worker is not None:
            self.gaze_worker.stop()
            self.gaze_worker.wait()
            self.gaze_worker = None
        if hasattr(self, '_gaze_logger') and self._gaze_logger is not None:
            self._gaze_logger.stop()
            self._gaze_logger.wait()
            self._gaze_logger = None

    def _show_gaze_frame(self, qimg):
        self.eye_video.setPixmap(QPixmap.fromImage(qimg).scaled(self.eye_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
        self.posture_tab.stop_all()
        self.gaze_posture_tab.stop_all()
        self.stop_headpose()

    def update_user_info(self, user_id, role):
        self.user_label.setText(f'{user_id} ({role})')

    def _logout(self):
        self.stop_all()
        self.window().logout()

class ProctorView(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        header = QLabel('Proctor Dashboard')
        header.setStyleSheet('font-size: 18px; font-weight: bold;')
        layout.addWidget(header)

        self.session_table = QTableWidget()
        self.session_table.setColumnCount(4)
        self.session_table.setHorizontalHeaderLabels(['Student ID', 'Name', 'Role', 'Login Time'])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.session_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.session_table)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._refresh()

    def _refresh(self):
        sessions = get_all_sessions()
        self.session_table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            self.session_table.setItem(i, 0, QTableWidgetItem(str(s['user_id'])))
            self.session_table.setItem(i, 1, QTableWidgetItem(str(s['full_name'])))
            self.session_table.setItem(i, 2, QTableWidgetItem(str(s['role'])))
            self.session_table.setItem(i, 3, QTableWidgetItem(str(s['login_time'])))
