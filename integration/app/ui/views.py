import random
import time
from collections import deque
from statistics import mean, pstdev
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QTransform, QColor, QFont, QPen
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QRadioButton, QButtonGroup, QFrame, QComboBox, QMdiArea, QMdiSubWindow, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QInputDialog, QCheckBox
from paths import APP_DIR
from workers.posture_worker import SideCameraWorker
from workers.front_cam_worker import FrontCamWorker
from loggers.front_cam_logger import FrontCamLogWriter
from cheat import calibration_store
from loggers.cheat_logger import CheatEventLogger
from ui.gaze_graph import GazeGraph
from ui.hull_view import HullView
from core.db_config import get_connection
from ui.phone_camera import SideCameraDialog, populate_camera_combo, pick_cameras, CameraPreview


def _combo_index(combo):
    """Camera index behind the selected item in a name-populated combo."""
    data = combo.currentData()
    return data if isinstance(data, int) else 0


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
    # Train in the background so the app doesn't freeze.

    done = Signal(object)    # result dict from train_cheat_model.train()
    failed = Signal(str)     # error message

    def __init__(self, user, parent=None):
        super().__init__(parent)
        self.user = user

    def run(self):
        try:
            from cheat.train_cheat_model import train   # heavy imports happen here
            self.done.emit(train(self.user))
        except Exception as exc:
            self.failed.emit(str(exc))


# dot spots near the screen edges (0 to 1)
POINTS_9 = [(.5, .5), (.05, .05), (.95, .05), (.05, .95), (.95, .95), (.5, .05), (.05, .5), (.95, .5), (.5, .95)]
POINTS_5 = [(.5, .5), (.05, .05), (.95, .05), (.05, .95), (.95, .95)]
# test dots, not used to make the area
VAL_POINTS = [(.3, .3), (.7, .3), (.3, .7), (.7, .7)]
READY_SEC = 2       # wait before the first dot
DOT_SEC = 2         # seconds per dot
AWAY_DEG = 25       # head turned more than this from the start = looking away
CLOSED = 0.12       # eye openness below this = eyes closed
DIM = 200            # 0 = see-through camera, 255 = black
HEAD = (.5, .42, .22)  # head circle in the camera: center x, center y, radius (0 to 1 of the height)

# --- WebGazer-style click calibration --------------------------------------
# One dot HOPS around the screen; the student follows it and clicks it. People
# look where they click, so each click is proof of attention, and because the dot
# keeps jumping they must actually follow it. The dot only visits the fixed
# targets (so training still gets enough frames per target), each VISITS_PER_DOT
# times in a shuffled order (no two the same in a row) so it can't be clicked
# from memory. We record only the frames around each click, then verify the eye
# features moved with the dots before saving. Tune these on real students.
VISITS_PER_DOT = 3     # times each target appears in the hop path (>=10 frames/target)
CLICK_WINDOW = 0.4     # seconds of frames kept around each click and recorded
MAX_SPREAD = 0.05      # max std of h_ratio / v_openness across one target's clicks
MIN_GAP = 0.015        # min mean gap between left/right (and top/bottom) dots


def hop_path(targets):
    # Each target VISITS_PER_DOT times, shuffled, never the same target twice in a
    # row - so every hop is a real move the student has to follow.
    path = list(targets) * VISITS_PER_DOT
    random.shuffle(path)
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            for j in range(i + 1, len(path)):
                if path[j] != path[i - 1]:
                    path[i], path[j] = path[j], path[i]
                    break
    return path


def person_path(w, h):
    # Head + shoulders shape, in camera pixels, where the student should sit.
    cx, cy, r = HEAD[0] * w, HEAD[1] * h, HEAD[2] * h
    head = QPainterPath()
    head.addEllipse(QPointF(cx, cy), r, r)
    body = QPainterPath()
    top = cy + r * 0.9
    body.addRoundedRect(cx - r * 1.6, top, r * 3.2, h - top + r, r, r)
    return head.united(body)


class DotWindow(QWidget):
    """Full screen dots for calibration.

    WebGazer-style: one dot HOPS around the screen and the student follows it and
    clicks it (people look where they click, so a click is proof of attention).
    Each click makes the dot jump to the next spot in the hop path. After the hop
    path, the VAL_POINTS are shown timed (no clicking) as the accuracy step. The
    mouse cursor stays visible; CalibrationView checks and records each click via
    on_click, and records the timed val frames in _collect.
    """

    def __init__(self, mode, on_done, on_cancel, on_click):
        super().__init__()
        self.setWindowTitle('AEye Calibration')
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self._on_done = on_done
        self._on_cancel = on_cancel
        self._on_click = on_click            # CalibrationView: checks + records a click
        targets = POINTS_5 if mode == '5 point' else POINTS_9
        self.path = hop_path(targets)         # shuffled hops among the targets
        self.val_points = VAL_POINTS
        self.step = 0                         # current hop
        self.phase = 'click'                  # click -> val -> done
        self.val_i = 0                        # current val dot
        self.val_elapsed = 0.0                # seconds on the current val dot (while facing)
        self.msg = ''                         # red reason text after a rejected click
        self.pause = None                     # head turned / eyes closed reason, None = ok
        self.frame = None                     # newest camera picture, drawn faintly behind
        self.in_place = False                 # face inside the person shape
        self.last_seen = 0.0                  # last time any face was seen
        self.last_face = 0.0                  # last usable frame
        self._last_tick = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)                 # 60 fps

    # --- val phase (timed staring, no clicking) ---------------------------
    def val_point(self):
        return self.val_points[self.val_i]

    def val_recording(self):
        # record only the settled second of each val dot
        return self.phase == 'val' and self.val_elapsed > 1.0

    def _start_val(self):
        self.phase = 'val'
        self.val_i = 0
        self.val_elapsed = 0.0
        self._last_tick = time.time()

    def _finish_all(self):
        self._timer.stop()
        self.phase = 'done'
        self._on_cancel = None
        QTimer.singleShot(0, self._on_done)   # run after this frame's slot returns

    def _tick(self):
        now = time.time()
        if self.phase == 'val':
            if now - self.last_face < 0.5:            # only count time while facing
                self.val_elapsed += now - self._last_tick
            if self.val_elapsed >= DOT_SEC:
                self.val_i += 1
                self.val_elapsed = 0.0
                if self.val_i >= len(self.val_points):
                    self._finish_all()
                    return
        self._last_tick = now
        self.update()

    def mousePressEvent(self, event):
        if self.phase != 'click':
            return
        tx, ty = self.path[self.step]
        cx, cy = tx * self.width(), ty * self.height()
        if ((event.position().x() - cx) ** 2 + (event.position().y() - cy) ** 2) ** 0.5 > 26:
            return                                    # only clicks on the dot count
        reason = self._on_click(self.path[self.step])  # checks + records the window
        if reason:
            self.msg = reason                          # bad click: show why, no hop
        else:
            self.msg = ''
            self.step += 1                             # good click -> hop to the next spot
            if self.step >= len(self.path):
                self._start_val()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        W, H = self.width(), self.height()
        in_place = self.in_place and time.time() - self.last_seen < 0.5
        if self.frame is not None:
            # camera behind a dark layer, so the student can see themselves a little
            img = self.frame.scaled(self.size(), Qt.KeepAspectRatio)
            fx, fy = (W - img.width()) // 2, (H - img.height()) // 2
            p.drawImage(fx, fy, img)
            # person shape from camera pixels to screen pixels
            s = img.width() / self.frame.width()
            person = QTransform().translate(fx, fy).scale(s, s).map(
                person_path(self.frame.width(), self.frame.height()))
            outside = QPainterPath()
            outside.addRect(QRectF(self.rect()))
            p.fillPath(outside.subtracted(person), QColor(0, 0, 0, DIM))
            ok = in_place and not self.pause
            p.setPen(QPen(QColor('#22c55e') if ok else QColor('#ef4444'), 3))
            p.setBrush(Qt.NoBrush)
            p.drawPath(person)
        p.setFont(QFont('Arial', 16))
        if not in_place:
            p.setPen(Qt.red)
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop,
                       '\nPaused - sit inside the person shape')
        elif self.pause:
            p.setPen(Qt.red)
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, '\nPaused - ' + self.pause)
        elif self.msg:
            p.setPen(Qt.red)
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, '\n' + self.msg)
        elif self.phase == 'click':
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop,
                       '\nFollow the dot and click it')
        else:
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop,
                       '\nNow just look at each dot (no clicking)')
        # the dot
        if self.phase == 'click':
            tx, ty = self.path[self.step]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawEllipse(QPointF(int(tx * W), int(ty * H)), 18, 18)
        elif self.phase == 'val':
            tx, ty = self.val_point()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor('#22c55e') if self.val_recording() else QColor('#ffffff'))
            p.drawEllipse(QPointF(int(tx * W), int(ty * H)), 18, 18)
        # dot counter (bottom-right)
        total = len(self.path) + len(self.val_points)
        done = self.step if self.phase == 'click' else len(self.path) + self.val_i
        p.setPen(QColor('#cbd5e1'))
        p.setFont(QFont('Arial', 12))
        p.drawText(self.rect().adjusted(0, 0, -12, -8), Qt.AlignRight | Qt.AlignBottom,
                   f'dot {min(done + 1, total)}/{total}')
        p.end()

    def keyPressEvent(self, event):
        # esc to exit
        if event.key() == Qt.Key_Escape:
            answer = QMessageBox.question(self, 'Exit calibration?',
                                          'Nothing will be saved. Exit anyway?')
            if answer == QMessageBox.Yes:
                self.close()

    def closeEvent(self, event):
        self._timer.stop()
        cb = self._on_cancel
        self._on_cancel = None   # fire the cancel callback at most once
        if cb:
            cb()
        super().closeEvent(event)


class ScreenBorder(QWidget):
    # Red border around the screen when eyes are off screen.

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.off = False

    def set_gaze(self, feats):
        # Turn the border on or off.
        if feats.get('off_screen', False) != self.off:
            self.off = feats.get('off_screen', False)
            self.update()

    def paintEvent(self, event):
        # Draw the red border.
        if self.off:
            p = QPainter(self)
            p.setPen(QPen(QColor('#ef4444'), 12))
            p.drawRect(self.rect().adjusted(6, 6, -6, -6))
            p.end()


class CalibrationView(QWidget):
    # Step 1: dot calibration screen.

    def __init__(self, on_proceed=None):
        super().__init__()
        self._on_proceed = on_proceed
        self.worker = None
        self._trainer = None
        self._reader = None       # the dot window
        self._samples = []
        self._buf = deque(maxlen=40)   # recent (time, feats) for click-window sampling
        self._latest = None       # newest feats, for the click-time checks
        self._user_id = None
        self._preview = None      # live face-cam preview shown before recording
        self._saved_runs = 0      # calibration runs saved (for the "done" message)

        card_l = QVBoxLayout(self)
        card_l.setContentsMargins(20, 20, 20, 20)
        card_l.setSpacing(10)

        heading = QLabel('Calibration')
        heading.setStyleSheet('font-size: 18px; font-weight: bold;')
        card_l.addWidget(heading)

        instructions = QLabel(
            'Pick your camera and calibration mode, then press Start Calibration. '
            'A full-screen window shows one dot that HOPS around - look straight at '
            'it and CLICK it each time it moves, following it around the screen. '
            'After that, just look at a few dots without clicking. Esc exits. If '
            'your gaze did not follow the dots the run is rejected; otherwise the '
            'model trains automatically - then press Proceed.')
        instructions.setWordWrap(True)
        card_l.addWidget(instructions)

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

        self.status = QLabel('Press Start Calibration to begin.')
        card_l.addWidget(self.status)

        controls = QHBoxLayout()
        controls.addWidget(QLabel('Mode:'))
        self.mode_box = QComboBox()
        self.mode_box.addItems(['9 point', '5 point'])
        controls.addWidget(self.mode_box)
        controls.addWidget(QLabel('Camera:'))
        self.cam_box = QComboBox()
        populate_camera_combo(self.cam_box)   # picks the face camera by itself
        self.cam_box.setMinimumWidth(170)
        self.cam_box.currentIndexChanged.connect(self._restart_preview)
        controls.addWidget(self.cam_box)
        controls.addStretch(1)
        self.start_btn = QPushButton('Start Calibration')
        self.start_btn.clicked.connect(self._start)
        controls.addWidget(self.start_btn)
        self.proceed_btn = QPushButton('Proceed to Monitoring')
        self.proceed_btn.clicked.connect(self._proceed)
        controls.addWidget(self.proceed_btn)
        card_l.addLayout(controls)

        self.cmd_label = QLabel('')
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cmd_label.setVisible(False)
        card_l.addWidget(self.cmd_label)

    def _start(self):
        if self.worker is not None:
            return
        session = self.window().session
        self._user_id = session.user_id if session else 'test_user'
        self._samples = []
        self._buf = deque(maxlen=40)
        self._latest = None
        self._stop_preview()   # hand the camera to the recorder
        # detect=False -> record only. We collect the raw features via
        # features_ready and save them to JSON; nothing is written to MySQL.
        cam = _combo_index(self.cam_box)
        self.worker = FrontCamWorker(camera_index=cam, session_user_id=self._user_id, detect=False)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.features_ready.connect(self._collect)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.mode_box.setEnabled(False)
        self.cam_box.setEnabled(False)
        self.cmd_label.setVisible(False)
        self.status.setStyleSheet('')
        self.status.setText('Calibrating... (dot window is open)')
        self._reader = DotWindow(self.mode_box.currentText(),
                                 on_done=self._finish, on_cancel=self._cancel_reading,
                                 on_click=self._click)
        self._reader.showFullScreen()

    def _collect(self, feats):
        # Buffer recent frames; a valid click copies a short window of them. In
        # the val phase, record the timed staring frames (val = True).
        r = self._reader
        if r is None or r.frame is None:
            return
        # whole face must be inside the person shape
        w, h = r.frame.width(), r.frame.height()
        shape = person_path(w, h)
        r.in_place = all(shape.contains(QPointF(x * w, y * h)) for x, y in feats['face_pts'])
        r.last_seen = time.time()
        if not r.in_place:
            return                                      # outside the shape: pause, save nothing
        self._latest = feats
        # head turned or eyes closed: pause, save nothing, val timer stops
        r.pause = self._gates(feats)
        if r.pause:
            return
        r.last_face = time.time()
        self._buf.append((time.time(), feats))
        # val phase: record the timed staring frames as test dots
        if r.val_recording():
            self._samples.append(self._sample(feats, r.val_point(), True))

    def _gates(self, feats):
        # Shared face/head/eye checks. None = ok, else a short red reason.
        # yaw/pitch 0 = facing the camera
        if abs(feats['yaw']) > AWAY_DEG or abs(feats['pitch']) > AWAY_DEG:
            return 'keep your head facing the screen'
        if feats['openness'] < CLOSED:
            return 'keep your eyes open'
        return None

    def _sample(self, feats, dot, val):
        # Same JSON shape as before (extra feats keys are ignored by training).
        s = dict(feats)
        s['target'] = [dot[0], dot[1]]
        s['val'] = val
        return s

    def _click(self, dot):
        # Called by DotWindow on a click that landed on the dot. Returns None if
        # accepted (and records the last CLICK_WINDOW seconds of frames), else a
        # short reason to show in red.
        r = self._reader
        if r is None or not r.in_place or time.time() - r.last_seen > 0.5:
            return 'sit inside the person shape'
        feats = self._latest
        if feats is None:
            return 'no face detected'
        reason = self._gates(feats)
        if reason:
            return reason
        now = time.time()
        kept = [f for t, f in self._buf if now - t <= CLICK_WINDOW]
        for f in kept or [feats]:                       # at least the click frame
            self._samples.append(self._sample(f, dot, False))
        return None

    def _finish(self):
        screen = [self._reader.width(), self._reader.height()]
        self._stop_worker()
        self._close_reader()
        self.start_btn.setEnabled(True)
        self.start_btn.setText('Calibrate Again')
        self.mode_box.setEnabled(True)
        self.cam_box.setEnabled(True)
        self._restart_preview()   # bring the live camera view back
        clicks = [s for s in self._samples if not s['val']]   # click dots only
        # 1. enough samples
        if len(self._samples) < 100:
            self._reject(f'Only {len(self._samples)} samples captured. '
                         f'Click each dot while looking at it, and calibrate again.')
            return
        # 2. direction: the eyes must move with the dots (prints the means)
        ok, info = self._direction_ok(clicks)
        print('[calib]', info)
        if not ok:
            self._reject(f'Your gaze did not follow the dots ({info}). '
                         f'Look at each dot as you click it, and calibrate again.')
            return
        # 3. spread: each dot's clicks must be steady
        if not self._spread_ok(clicks):
            self._reject('Your gaze was not steady on each dot. '
                         'Look straight at each dot as you click, and calibrate again.')
            return
        self._saved_runs = calibration_store.add_session(self._user_id, self._samples, screen)
        self.cmd_label.setVisible(False)
        # Training now runs automatically as soon as a good calibration is saved.
        self._train()

    def _reject(self, msg):
        # Save nothing and tell the student why, so they can calibrate again.
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText(msg)

    def _direction_ok(self, samples):
        # Looking right raises h_ratio; looking up raises v_openness. A student
        # who clicks without looking gives about the same means left/right and
        # top/bottom. The printed means show the real sign on our hardware - if a
        # good run comes out negative, flip the two compares below.
        left = [s['h_ratio'] for s in samples if s['target'][0] < 0.5]
        right = [s['h_ratio'] for s in samples if s['target'][0] > 0.5]
        top = [s['v_openness'] for s in samples if s['target'][1] < 0.5]
        bottom = [s['v_openness'] for s in samples if s['target'][1] > 0.5]
        if not (left and right and top and bottom):
            return False, 'not enough dots'
        hl, hr, vt, vb = mean(left), mean(right), mean(top), mean(bottom)
        info = (f'h_ratio left={hl:.3f} right={hr:.3f} gap={hr - hl:+.3f}  '
                f'v_openness top={vt:.3f} bottom={vb:.3f} gap={vt - vb:+.3f}')
        ok = (hr - hl) >= MIN_GAP and (vt - vb) >= MIN_GAP
        return ok, info

    def _spread_ok(self, samples):
        # Each dot's clicks should land on similar values; staring elsewhere while
        # clicking makes them jump around (high std).
        by_dot = {}
        for s in samples:
            by_dot.setdefault(tuple(s['target']), []).append(s)
        for ss in by_dot.values():
            if len(ss) < 2:
                continue
            if (pstdev(s['h_ratio'] for s in ss) > MAX_SPREAD
                    or pstdev(s['v_openness'] for s in ss) > MAX_SPREAD):
                return False
        return True

    def _train(self):
        if not self._user_id:
            session = self.window().session
            self._user_id = session.user_id if session else 'test_user'
        self.start_btn.setEnabled(False)
        self.status.setStyleSheet('color: #7c5c00;')
        self.status.setText(f'Calibration saved. Training model for {self._user_id}... please wait.')
        self._trainer = TrainWorker(self._user_id)
        self._trainer.done.connect(self._train_done)
        self._trainer.failed.connect(self._train_failed)
        self._trainer.start()

    def _train_done(self, result):
        self.start_btn.setEnabled(True)
        total = result['val_total']
        pct = 100 * result['val_inside'] / total if total else 0
        runs = self._saved_runs or result['runs']
        self.status.setStyleSheet('color: #006600; font-weight: bold;')
        self.status.setText(
            f'✓ Training done for {self._user_id}. Screen area: {result["dots"]} dots, '
            f'{result["corners"]} corners, {runs} run(s); validation inside {pct:.0f}%. '
            f'You can now press Proceed to Monitoring.')

    def _train_failed(self, msg):
        self.start_btn.setEnabled(True)
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText(f'Training failed: {msg}')

    def _proceed(self):
        self._stop_preview()
        self._stop_worker()
        if self._on_proceed is not None:
            self._on_proceed()

    def _show_frame(self, qimg):
        if self._reader is not None:
            self._reader.frame = qimg     # faint background in the dot window
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def _start_preview(self):
        # Live view of the selected face camera before recording. Skipped while
        # a recording is running (the real worker owns the camera then).
        if self.worker is not None:
            return
        self._stop_preview()
        self._preview = CameraPreview(_combo_index(self.cam_box))
        self._preview.frame.connect(self._show_frame)
        self._preview.start()

    def _stop_preview(self):
        if self._preview is not None:
            self._preview.stop()
            self._preview.wait()
            self._preview = None

    def _restart_preview(self, *args):
        # Camera changed, or a recording ended: resume the live preview if the
        # screen is visible and we are not recording.
        if self.isVisible() and self.worker is None:
            self._start_preview()

    def showEvent(self, event):
        super().showEvent(event)
        self._start_preview()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._stop_preview()

    def _close_reader(self):
        if self._reader is not None:
            self._reader._on_cancel = None   # don't fire cancel on a programmatic close
            self._reader.close()
            self._reader = None

    def _cancel_reading(self):
        # The student exited early: stop and reset. Nothing is written - the
        # JSON is only saved in _finish(), so a cancelled run leaves no file
        # and no model behind.
        self._stop_worker()
        self._reader = None            # it is already closing itself
        self._samples = []             # drop the partial recording
        self.start_btn.setEnabled(True)
        self.start_btn.setText('Start Calibration')
        self.mode_box.setEnabled(True)
        self.cam_box.setEnabled(True)
        self._restart_preview()
        self.status.setStyleSheet('color: #a00000;')
        self.status.setText('Calibration cancelled - nothing was saved.')

    def stop_all(self):
        self._stop_preview()
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

        # Cameras are picked by NAME. The front (face) cam is a named dropdown;
        # the side cam is chosen in a popup with a live preview.
        cams_row = QHBoxLayout()
        cams_row.addStretch(1)
        cams_row.addWidget(QLabel('Front cam:'))
        self.front_box = QComboBox()
        names = populate_camera_combo(self.front_box)   # picks the face camera by itself
        # pick the side camera by itself (phone first)
        self._side_idx = pick_cameras(names)[1]
        self._side_label = names[self._side_idx] if self._side_idx < len(names) else ''
        self.front_box.setMinimumWidth(150)
        cams_row.addWidget(self.front_box)
        cams_row.addSpacing(16)
        self.side_summary = QLabel()
        self.side_summary.setStyleSheet('color: #334155;')
        cams_row.addWidget(self.side_summary)
        self.choose_side_btn = QPushButton('Choose side camera')
        self.choose_side_btn.clicked.connect(self._choose_side)
        cams_row.addWidget(self.choose_side_btn)
        cams_row.addStretch(1)

        # Checkbox to hide the graph (some proctors only want the feeds).
        self.graph_check = QCheckBox('Show graph')
        self.graph_check.setChecked(True)
        self.graph_check.toggled.connect(self._show_graph)
        cams_row.addWidget(self.graph_check)

        # red border checkbox
        self.border_check = QCheckBox('Show off-screen border')
        self.border_check.setChecked(True)
        self.border_check.toggled.connect(self._show_border)
        cams_row.addWidget(self.border_check)
        self.border = ScreenBorder()
        self.hull_view = HullView()     # graham scan picture
        cams_row.addStretch(1)
        layout.addLayout(cams_row)
        self._update_side_summary()

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

    def _show_border(self, on):
        self.border.setVisible(on and self.front is not None)   # only when tracking

    def _update_side_summary(self):
        name = self._side_label or f'Camera {self._side_idx}'
        self.side_summary.setText(f'Side camera:  {name}')

    def _choose_side(self):
        # Open the name + preview popup; the front (face) cam can't be reused.
        if self.front is not None:
            return
        dlg = SideCameraDialog(self, front_index=_combo_index(self.front_box))
        dlg.exec()
        if dlg.selected_index is not None:
            self._side_idx = dlg.selected_index
            self._side_label = dlg.selected_label
            self._update_side_summary()

    def _toggle(self):
        if self.front is None:
            self._start()
        else:
            self.stop_all()

    def _start(self):
        session = self.window().session
        user_id = session.user_id if session else 'test_user'

        front_idx = _combo_index(self.front_box)
        side_idx = self._side_idx

        # Front cam: cheat detection (model + rule + 2s).
        self.front = FrontCamWorker(camera_index=front_idx, session_user_id=user_id, detect=True)
        self.front.frame_ready.connect(self._show_front)
        self.front.cheat_detected.connect(self._on_cheat)

        self.graph.set_user(user_id)
        self.front.features_ready.connect(self.graph.on_features)

        self.border.off = False
        self.front.features_ready.connect(self.border.set_gaze)
        self.hull_view.set_user(user_id)
        self.front.features_ready.connect(self.hull_view.on_features)
        self._show_border(self.border_check.isChecked())

        # Front cam rows (gaze + head pose, every frame) go to gaze_logs.
        self.front_log = FrontCamLogWriter()
        self.front_log.start()
        self.front.record_ready.connect(self.front_log.enqueue)

        # Side cam: posture feed. log_to_db=True writes joint coords to
        # posture_logs (2 rows/sec) so posture has history to train on.
        self.side = SideCameraWorker(camera_index=side_idx, session_user_id=user_id, log_to_db=True)
        self.side.frame_ready.connect(self._show_side)
        self.side.cheat_detected.connect(self._on_cheat)   # phone flags

        # MySQL writer - only used when a cheat actually fires.
        self.logger = CheatEventLogger()
        self.logger.start()

        self.front.start()
        self.side.start()

        self._count = 0
        self.front_box.setEnabled(False)   # can't change cameras mid-session
        self.choose_side_btn.setEnabled(False)
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
        self.border.hide()
        self.front_box.setEnabled(True)
        self.choose_side_btn.setEnabled(True)
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
            ('Graham Scan', self.front_side_tab.detect.hull_view),   # graham scan tab
            ('Proctor', ProctorView()),
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
            from ui.web_tab import WebTab
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
        full = APP_DIR / path if path else None
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
