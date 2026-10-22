# Train an Isolation Forest on a NORMAL video, then test it on a second video.
# No calibration, no login, no MySQL - only for testing.
# Run from integration/app:  ../.venv/Scripts/python.exe video_forest.py
# A file picker asks for the normal video first, then the video to test.
# The forest is saved to models/video_<normal>.joblib, so the next run with the same
# normal video skips training. Flags are saved to reports/flags_<test>.csv.
import csv
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QSizePolicy, QVBoxLayout, QWidget
from sklearn.ensemble import IsolationForest

from paths import APP_DIR, MODELS_DIR
from workers.front_cam_worker import FrontCamWorker

FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']   # the same 5 numbers as calibration
ALERT_SECONDS = 2.0                                             # unusual this long in a row = flag


class VideoForest(QWidget):

    def __init__(self, normal_video, test_video):
        super().__init__()
        self.setWindowTitle('Video Forest')
        self.resize(900, 650)

        self.normal_video = normal_video
        self.test_video = test_video
        self.forest = None      # None while training, the trained forest while testing
        self.rows = []          # training: the 5 numbers of every normal frame
        self.frames = 0         # frames shown so far in the current video
        self.fps = 30
        self.since = None       # video time when the current unusual stretch began
        self.flagged = False    # True while inside one flagged stretch
        self.flags = []         # video times of every flag, e.g. ['00:12.4']
        self.ranges = []        # every flagged stretch as [start_s, end_s], for the CSV
        self.old_workers = []   # finished workers, kept so Qt doesn't delete them too early
        self.model_path = MODELS_DIR / f'video_{Path(normal_video).stem}.joblib'

        layout = QVBoxLayout(self)
        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # lets the window shrink
        self.video.setStyleSheet('background: #0f172a;')
        layout.addWidget(self.video, stretch=1)

        self.status = QLabel()
        self.status.setStyleSheet('font-weight: bold; font-size: 14px;')
        layout.addWidget(self.status)

        self.flag_label = QLabel('Flags: none')
        self.flag_label.setWordWrap(True)
        layout.addWidget(self.flag_label)

        if self.model_path.exists():
            # this normal video was trained before: load the forest, go straight to testing
            self.forest = joblib.load(self.model_path)
            print('Loaded the saved forest:', self.model_path)
            self.play(test_video)
        else:
            self.play(normal_video)    # step 1: training

    def play(self, path):
        # The SAME worker as live tracking, fed a video file. detect=False = just give us the numbers.
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        self.frames = 0
        self.worker = FrontCamWorker(detect=False, video_path=path)
        self.worker.frame_ready.connect(self.show_frame)    # each picture
        self.worker.features_ready.connect(self.on_frame)   # each frame's numbers (only when a face is found)
        self.worker.finished.connect(self.on_video_end)     # the video is over
        self.worker.start()

    def show_frame(self, qimg):
        self.frames += 1
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_frame(self, feats):
        row = [feats[name] for name in FEATURES]
        sec = self.frames / self.fps                 # seconds into the video
        clock = f'{int(sec // 60):02d}:{sec % 60:04.1f}'

        # step 1: training - only collect the normal frames
        if self.forest is None:
            self.rows.append(row)
            self.status.setText(f'TRAINING on {Path(self.normal_video).name}   {clock}   '
                                f'{len(self.rows)} normal frames')
            return

        # step 3: testing - ask the forest about this one frame (1 = normal, -1 = unusual)
        unusual = self.forest.predict([row])[0] == -1
        if unusual:
            if self.since is None:
                self.since = sec                     # an unusual stretch begins
            if sec - self.since >= ALERT_SECONDS and not self.flagged:
                self.flagged = True                  # one flag per stretch, not one per frame
                self.ranges.append([self.since, sec])
                self.flags.append(f'{int(self.since // 60):02d}:{self.since % 60:04.1f}')
                self.flag_label.setText('Flags at: ' + ', '.join(self.flags))
                print('FLAG at', self.flags[-1])
            if self.flagged:
                self.ranges[-1][1] = sec             # the flagged stretch keeps growing
        else:
            self.since = None                        # back to normal: the stretch ends
            self.flagged = False

        answer = 'UNUSUAL' if unusual else 'usual'
        self.status.setText(f'TESTING {Path(self.test_video).name}   {clock}   {answer}'
                            + ('   - FLAGGED' if self.flagged else ''))

    def on_video_end(self):
        self.old_workers.append(self.worker)

        # end of the test video: show the result
        if self.forest is not None:
            # save the flags: one row per flagged stretch, in whole seconds
            name = Path(self.test_video).stem
            out = APP_DIR / 'reports' / f'flags_{name}.csv'
            out.parent.mkdir(exist_ok=True)
            with open(out, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['video', 'start_s', 'end_s', 'label'])
                for start, end in self.ranges:
                    writer.writerow([name, round(start), round(end), 'unusual'])
            self.status.setText(f'DONE. {len(self.flags)} flags. Saved to {out.name}')
            print(f'DONE. {len(self.flags)} flags. Saved to {out}')
            return

        # end of the normal video: step 2, train the forest, then start the test video
        if len(self.rows) < 100:
            self.status.setText(f'Only {len(self.rows)} frames with a face in the normal video - use a longer one.')
            return
        table = np.array(self.rows, dtype=float)
        self.forest = IsolationForest(
            n_estimators=200,      # build 200 trees
            contamination=0.03,    # the weirdest 3% of the normal frames count as unusual
            random_state=42).fit(table)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.forest, self.model_path)   # so the next run can skip training
        print(f'Trained the forest on {len(table)} normal frames. Saved to {self.model_path}')
        self.play(self.test_video)

    def closeEvent(self, event):
        self.worker.stop()       # closing the window must also stop the thread
        self.worker.wait()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    videos = 'Videos (*.mp4 *.avi *.mov *.mkv *.webm)'
    normal, _ = QFileDialog.getOpenFileName(None, '1 of 2: choose the NORMAL video (training)', '', videos)
    test, _ = QFileDialog.getOpenFileName(None, '2 of 2: choose the video to TEST', '', videos)
    if not normal or not test:
        sys.exit('You need to choose two videos.')
    window = VideoForest(normal, test)
    window.show()
    sys.exit(app.exec())
