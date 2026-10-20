# Video test window: plays a recorded video through the SAME front-cam detector
# used in live tracking, with a student's already-trained models.
# Nothing is written to MySQL - this is only for testing. Press P to pause.
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QFileDialog, QSizePolicy)

from paths import MODELS_DIR
from workers.front_cam_worker import FrontCamWorker


class VideoTestWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Test')
        self.resize(900, 650)

        self.worker = None        # the FrontCamWorker while a video plays, None when stopped
        self.video_path = None    # the chosen video file
        self.flags = []           # video times of every cheating flag, e.g. ['00:12.4', '01:03.0']
        self.last_line = ''       # last "time | usual/unusual" text, kept for when we pause

        layout = QVBoxLayout(self)

        # --- Row 1: choose the video + whose trained model to use ---
        row = QHBoxLayout()
        self.pick_btn = QPushButton('Choose Video')
        self.pick_btn.clicked.connect(self._choose_video)
        row.addWidget(self.pick_btn)

        self.file_label = QLabel('No video chosen')
        row.addWidget(self.file_label, stretch=1)

        # every student that has a trained forest: models/isolation_<user>.joblib
        row.addWidget(QLabel('Model of:'))
        self.user_box = QComboBox()
        for model_file in sorted(MODELS_DIR.glob('isolation_*.joblib')):
            self.user_box.addItem(model_file.stem.removeprefix('isolation_'))
        row.addWidget(self.user_box)
        layout.addLayout(row)

        # --- The video picture (same dark box style as Live Tracking) ---
        self.video = QLabel('Video')
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(320, 240)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # lets the window shrink
        self.video.setStyleSheet('background: #0f172a; color: #cbd5e1;')
        layout.addWidget(self.video, stretch=1)

        # --- Status: video time + the model's answer, then the list of flags ---
        self.status = QLabel('Choose a video, then press Play.')
        self.status.setStyleSheet('font-weight: bold; font-size: 14px;')
        layout.addWidget(self.status)

        self.flag_label = QLabel('Flags: none')
        self.flag_label.setWordWrap(True)      # long flag lists go onto the next line
        layout.addWidget(self.flag_label)

        # --- Row 2: Play/Stop + Pause ---
        row2 = QHBoxLayout()
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self._play_or_stop)
        row2.addWidget(self.play_btn, stretch=1)

        self.pause_btn = QPushButton('Pause (P)')
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)       # nothing to pause until a video plays
        row2.addWidget(self.pause_btn)
        layout.addLayout(row2)

        # P on the keyboard does the same as the Pause button
        self._pause_shortcut = QShortcut(QKeySequence('P'), self)
        self._pause_shortcut.setContext(Qt.WidgetWithChildrenShortcut)   # P only works inside this view
        self._pause_shortcut.activated.connect(self._toggle_pause)

    def _choose_video(self):
    # Open the Windows file picker; only video files are listed.
        path, _ = QFileDialog.getOpenFileName(self, 'Choose a test video', '',
                                            'Videos (*.mp4 *.avi *.mov *.mkv *.webm)')
        if path:                                     # empty text = the user pressed Cancel
            self.video_path = path
            self.file_label.setText(Path(path).name)   # just the name, e.g. cheat1.mp4

    def _play_or_stop(self):
        # One button: Play when stopped, Stop while playing.
        if self.worker is not None:
            self._stop()
            return
        if not self.video_path:
            self.status.setText('Choose a video first.')
            return
        if self.user_box.count() == 0:
            self.status.setText('No trained model yet. Calibrate + Train a student first.')
            return

        # the SAME worker as Live Tracking, only fed a video file instead of the camera
        self.worker = FrontCamWorker(session_user_id=self.user_box.currentText(),
                                     detect=True, video_path=self.video_path)
        self.worker.frame_ready.connect(self._show_frame)      # each picture
        self.worker.stats_ready.connect(self._show_stats)      # time + usual/unusual
        self.worker.cheat_detected.connect(self._on_cheat)     # 2-second flags
        self.worker.finished.connect(self._on_finished)        # the thread ended (video over)

        self.flags = []
        self.flag_label.setText('Flags: none')
        self.worker.start()                                    # runs run() in its own thread

        self.play_btn.setText('Stop')
        self.pause_btn.setEnabled(True)
        self.pick_btn.setEnabled(False)       # no changing the video or model mid-play
        self.user_box.setEnabled(False)

    def _show_frame(self, qimg):
        # Fit the picture inside the box without stretching it.
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_stats(self, stats):
        # e.g. "Time 00:12.4  |  OFF SCREEN - unusual"
        self.last_line = f"Time {stats.get('Video Time', '--')}  |  {stats.get('Model', '--')}"
        self.status.setText(self.last_line)

    def _on_cheat(self, event):
        # 'start' = one cheating episode began (eyes off screen for 2 s); skip 'end'.
        if event['kind'] != 'start':
            return
        t = event['video_time']                              # seconds, e.g. 75.4
        self.flags.append(f'{int(t // 60):02d}:{t % 60:04.1f}')   # -> '01:15.4'
        self.flag_label.setText('Flags at: ' + ', '.join(self.flags))

    def _toggle_pause(self):
        if self.worker is None:
            return                                  # nothing playing, nothing to pause
        paused = self.worker.toggle_pause()         # the worker's own pause (same as Live Tracking)
        if paused:
            self.status.setText('PAUSED   ' + self.last_line)   # the frozen frame's answer stays
            self.pause_btn.setText('Resume (P)')
        else:
            self.pause_btn.setText('Pause (P)')

    def _stop(self):
        if self.worker is None:
            return
        self.worker.stop()       # tells the while-loop in run() to end
        self.worker.wait()       # wait until the thread is really finished
        self.worker = None
        self.play_btn.setText('Play')
        self.pause_btn.setText('Pause (P)')
        self.pause_btn.setEnabled(False)
        self.pick_btn.setEnabled(True)
        self.user_box.setEnabled(True)

    def _on_finished(self):
        # Runs when the video reaches its end (the worker's break).
        self._stop()
        self.status.setText(f'Video finished. {len(self.flags)} flags.')

    def closeEvent(self, event):
        self._stop()             # closing the window must also stop the thread
        super().closeEvent(event)


# test: open only this window, no login needed
if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = VideoTestWindow()
    window.show()
    sys.exit(app.exec())