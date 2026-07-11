import sys
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView

from keyboard_lock import KeyboardLock

# The e-class exam page AEye displays inside its controlled window.
# Placeholder for now — later point this at your researcher-controlled
# test instance (e.g. "http://localhost/moodle" or your test exam URL).
EXAM_URL = "https://example.com"


class LockdownWindow(QMainWindow):
    """AEye's controlled exam window (student mode): fullscreen,
    always-on-top, app-switching blocked, showing the e-class page."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEye")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # Embedded Chromium browser = the "controlled window".
        # We only DISPLAY the e-class page; we don't share its login
        # session (thesis: no session-level e-class integration).
        self.web = QWebEngineView()
        self.web.load(QUrl(EXAM_URL))
        self.setCentralWidget(self.web)

        # DEV-ONLY exit (Q isn't a blocked key, so it still reaches Qt).
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        quit_shortcut.activated.connect(QApplication.quit)


def main():
    # Recommended before creating the app when using QtWebEngine, so the
    # embedded browser shares one OpenGL context (avoids a GL warning).
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    # System-wide keyboard hook; removed automatically on quit.
    lock = KeyboardLock()
    lock.install()
    app.aboutToQuit.connect(lock.uninstall)

    window = LockdownWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()