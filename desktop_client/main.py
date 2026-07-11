import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from keyboard_lock import KeyboardLock
from views import LoginView, StudentExamView, ProctorView
from auth import authenticate
from session import Session


class MainWindow(QMainWindow):
    """AEye's single-app shell: login -> route to student or proctor
    mode. Student mode locks down (fullscreen + keyboard hook); login
    and proctor mode run as a normal window.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEye")

        self.lock = KeyboardLock()
        self.session = None   # set once someone signs in

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_view = LoginView(on_login=self.handle_login)
        self.student_view = StudentExamView()
        self.proctor_view = ProctorView()

        self.stack.addWidget(self.login_view)     # index 0
        self.stack.addWidget(self.student_view)   # index 1
        self.stack.addWidget(self.proctor_view)   # index 2

        quit_sc = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        quit_sc.activated.connect(QApplication.quit)

    def handle_login(self, user_id, password, role):
        # Verify (placeholder for now), then start a session and route.
        if not authenticate(user_id, password, role):
            self.login_view.show_error("Please enter your ID and password.")
            return

        self.session = Session(user_id=user_id.strip(), role=role)

        if role == "student":
            self.enter_student_mode()
        else:
            self.enter_proctor_mode()

    def enter_student_mode(self):
        self.stack.setCurrentWidget(self.student_view)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.lock.install()
        self.showFullScreen()

    def enter_proctor_mode(self):
        self.stack.setCurrentWidget(self.proctor_view)
        self.showMaximized()


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)

    window = MainWindow()
    app.aboutToQuit.connect(window.lock.uninstall)

    window.show()   # start on the login screen (windowed)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()