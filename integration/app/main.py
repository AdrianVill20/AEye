import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from keyboard_lock import KeyboardLock
from views import LoginView, AnalysisDashboard, ProctorView
from auth import authenticate
from session import Session
from db_config import init_schema


class MainWindow(QMainWindow):
    """AEye shell: login (start) -> student mode (locked dashboard) or
    proctor mode (unlocked monitoring)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEye")

        self.lock = KeyboardLock()
        self.session = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_view = LoginView(on_login=self.handle_login)
        self.student_view = AnalysisDashboard()
        self.proctor_view = ProctorView()

        self.stack.addWidget(self.login_view)      # index 0 (start here)
        self.stack.addWidget(self.student_view)    # index 1
        self.stack.addWidget(self.proctor_view)    # index 2

        quit_sc = QShortcut(QKeySequence("Ctrl+Shift+Q"), self)
        quit_sc.setContext(Qt.ApplicationShortcut)
        quit_sc.activated.connect(QApplication.quit)

    def handle_login(self, user_id, password, role):
        if not authenticate(user_id, password, role):
            self.login_view.show_error("Invalid ID, password, or role.")
            return
        self.session = Session(user_id=user_id.strip(), role=role)
        if role == "student":
            self.enter_student_mode()
        else:
            self.enter_proctor_mode()

    def enter_student_mode(self):
        self.student_view.update_user_info(self.session.user_id, self.session.role)
        self.stack.setCurrentWidget(self.student_view)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.lock.install()
        self.showFullScreen()

    def enter_proctor_mode(self):
        self.proctor_view._refresh()
        self.stack.setCurrentWidget(self.proctor_view)
        self.showMaximized()

    def logout(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.lock.uninstall()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.stack.setCurrentWidget(self.login_view)
        self.login_view.show_error('')
        self.login_view.id_input.clear()
        self.login_view.pw_input.clear()
        self.showNormal()
        self.resize(600, 400)


def main():
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)

    init_schema()

    window = MainWindow()
    app.aboutToQuit.connect(window.lock.uninstall)
    app.aboutToQuit.connect(window.student_view.stop_all)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
