import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from keyboard_lock import KeyboardLock
from views import LoginView, AnalysisDashboard, ProctorView
from auth import authenticate
from session import Session


class MainWindow(QMainWindow):
    """AEye shell: login (start) -> student mode (locked dashboard) or
    proctor mode (unlocked placeholder)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEye")

        self.lock = KeyboardLock()
        self.session = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_view = LoginView(on_login=self.handle_login)
        self.student_view = AnalysisDashboard()   # locked, manual tabs
        self.proctor_view = ProctorView()

        self.stack.addWidget(self.login_view)      # index 0 (start here)
        self.stack.addWidget(self.student_view)    # index 1
        self.stack.addWidget(self.proctor_view)    # index 2

        quit_sc = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        quit_sc.setContext(Qt.ApplicationShortcut)   # works even over the full-screen web view
        quit_sc.activated.connect(QApplication.quit)

    def handle_login(self, user_id, password, role):
        if not authenticate(user_id, password, role):
            self.login_view.show_error("Please enter your ID and password.")
            return
        self.session = Session(user_id=user_id.strip(), role=role)
        if role == "student":
            self.enter_student_mode()
        else:
            self.enter_proctor_mode()

    def enter_student_mode(self):
        # Show the dashboard, then lock down: on-top + keyboard hook +
        # fullscreen (no Alt+Tab / Windows key / Alt+F4).
        self.stack.setCurrentWidget(self.student_view)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.lock.install()
        self.showFullScreen()

    def enter_proctor_mode(self):
        self.stack.setCurrentWidget(self.proctor_view)
        self.showMaximized()


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)   # required by Qt WebEngine
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.lock.uninstall)         # remove hook on exit
    app.aboutToQuit.connect(window.student_view.stop_all)  # kill launched procs
    window.show()   # start on the login screen (windowed)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()