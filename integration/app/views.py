import sys
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QFrame, QComboBox, QMdiArea, QMdiSubWindow, QSizePolicy
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

        self.worker = self.worker_class(camera_index=camera_index, session_user_id=user_id)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.stats_ready.connect(self._show_status)
        self.worker.start()

        self.index_box.setEnabled(False)   # can't change camera while running
        self.button.setText('Stop')

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
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


class LockedPage(QWebEnginePage):
    """A web page that only allows ONE website. Any link, redirect, or URL to a
    different host is rejected, so the browser can never leave the allowed site."""

    def __init__(self, allowed_host, parent=None):
        super().__init__(parent)
        self.allowed_host = allowed_host

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        host = url.host()
        allowed = (
            host == ''                                  # internal pages (about:blank, etc.)
            or host == self.allowed_host
            or host.endswith('.' + self.allowed_host)   # its own sub-domains
        )
        if not allowed:
            return False        # reject -> the browser stays on the current page
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)

    def createWindow(self, _type):
        # Open "new tab / pop-up" links in THIS same page, so the rule above
        # still applies (off-site links stay blocked, no pop-up can escape).
        return self


class WebTab(QWidget):
    """Full-screen kiosk browser locked to ONE site. No toolbar and no window
    controls — just the page filling the whole screen. Off-site links, redirects
    and pop-ups are all blocked. Exit the app with Ctrl+Shift+Q."""

    ALLOWED_HOST = 'eclass.scs.usjr.edu.ph'
    HOME_URL = 'https://eclass.scs.usjr.edu.ph/'

    def __init__(self):
        super().__init__()
        # No toolbar, no margins — the page fills the screen edge to edge.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QWebEngineView()
        self.browser.setPage(LockedPage(self.ALLOWED_HOST, self.browser))
        self.browser.setUrl(QUrl(self.HOME_URL))
        layout.addWidget(self.browser)


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
        self.web_tab = WebTab()
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
        # The exam browser takes over the ENTIRE screen (like F11): no toolbar,
        # no title bar, no window buttons — just the locked page, always on top.
        # It can't be closed; exit the whole app with Ctrl+Shift+Q.
        self.web_tab.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.web_tab.showFullScreen()

    def start_gaze(self):
        if self.gaze_worker is None:
            current_user = self.window().session.user_id if self.window().session else "test_user"
            
            self.gaze_worker = GazeWorker(camera_index=0, session_user_id=current_user)
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

class ProctorView(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel('Proctor Mode — monitoring dashboard goes here (Module 3)')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
