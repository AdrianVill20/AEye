"""AEye — Analysis Dashboard (locked demo for the panel).

Temporary demo. A fullscreen, LOCKED-DOWN window (no Alt+Tab, no
Windows key, no Alt+F4) whose content is placed in MANUALLY as tabs —
instead of wrapping the e-class website. Each tab is one analysis view
with a placeholder feed area; teammates embed the real camera + ML
output later via AnalysisTab.set_feed_widget().

Run with:  python analysis_gui.py
Exit during the demo with:  Ctrl + Shift + Q
Needs:  keyboard_lock.py  (same folder)
"""

import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame,
)

from keyboard_lock import KeyboardLock


class AnalysisTab(QWidget):
    """One analysis view: swappable feed area (left) + stats (right)."""

    def __init__(self, title, camera_desc, stats_fields, owner):
        super().__init__()

        root = QHBoxLayout(self)

        # --- Left: the feed area (the integration point) ---
        self.feed_area = QFrame()
        self.feed_area.setFrameShape(QFrame.Shape.Box)
        self.feed_area.setMinimumSize(640, 480)
        self.feed_area.setStyleSheet("background-color: #111; color: #ddd;")
        feed_layout = QVBoxLayout(self.feed_area)

        self.placeholder = QLabel(
            f"📷  {camera_desc}\n\n"
            "Live camera feed + ML overlay will be embedded here.\n"
            f"Provided by: {owner}"
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        feed_layout.addWidget(self.placeholder)

        # --- Right: stats panel (placeholder values for now) ---
        panel = QVBoxLayout()
        heading = QLabel(title)
        heading.setStyleSheet("font-weight: bold; font-size: 16px;")
        panel.addWidget(heading)

        for field in stats_fields:
            panel.addWidget(QLabel(f"{field}: --"))

        panel.addStretch()
        owner_label = QLabel(f"Owner: {owner}")
        owner_label.setStyleSheet("color: gray;")
        panel.addWidget(owner_label)

        root.addWidget(self.feed_area, stretch=3)
        root.addLayout(panel, stretch=1)

    def set_feed_widget(self, widget):
        """Teammates call this to replace the placeholder with their real
        camera/ML widget."""
        self.placeholder.hide()
        self.feed_area.layout().addWidget(widget)


class DashboardWindow(QMainWindow):
    """Fullscreen, locked window. Content = the analysis tabs, placed
    manually (not a web page)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEye")
        # Stay above everything so nothing can cover the demo.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        tabs = QTabWidget()
        tabs.addTab(
            AnalysisTab(
                title="Eye Tracking",
                camera_desc="Front camera — eye feature detection",
                stats_fields=["Pupil position", "Left eye", "Right eye",
                              "Gaze direction"],
                owner="Demetillo",
            ),
            "Eye Tracking",
        )
        tabs.addTab(
            AnalysisTab(
                title="Head Pose",
                camera_desc="Front camera — MediaPipe face mesh",
                stats_fields=["Yaw", "Pitch", "Roll",
                              "Landmarks detected (/478)"],
                owner="Demetillo",
            ),
            "Head Pose",
        )
        tabs.addTab(
            AnalysisTab(
                title="Posture",
                camera_desc="Side camera — MediaPipe pose",
                stats_fields=["Shoulder points", "Hand-to-desk distance",
                              "Landmarks detected (/33)"],
                owner="Ybañez",
            ),
            "Posture",
        )
        self.setCentralWidget(tabs)

        # DEV/DEMO exit. 'Q' is not a blocked key, so this still reaches
        # Qt even with the keyboard hook active.
        quit_sc = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        quit_sc.activated.connect(QApplication.quit)


def main():
    app = QApplication(sys.argv)

    # Lock the environment: block Alt+Tab, Windows key, Alt+F4, etc.
    lock = KeyboardLock()
    lock.install()
    app.aboutToQuit.connect(lock.uninstall)   # always remove hook on exit

    window = DashboardWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()