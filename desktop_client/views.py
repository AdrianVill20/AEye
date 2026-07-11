from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QRadioButton, QButtonGroup,
)

# The e-class page AEye displays. Placeholder for now; later point this
# at your researcher-controlled test instance.
EXAM_URL = "https://example.com"


class ModeSelectView(QWidget):
    """Landing screen: pick Student or Proctor mode.

    It doesn't know anything about the main window. Instead it's handed
    two callbacks (on_student, on_proctor) and just calls them when a
    button is clicked. Later, Module 2's login will choose the mode
    automatically based on who signed in.
    """

    def __init__(self, on_student, on_proctor):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("AEye")
        title.setAlignment(Qt.AlignCenter)

        student_btn = QPushButton("Student Mode")
        proctor_btn = QPushButton("Proctor Mode")
        student_btn.clicked.connect(on_student)
        proctor_btn.clicked.connect(on_proctor)

        layout.addWidget(title)
        layout.addWidget(student_btn)
        layout.addWidget(proctor_btn)


class StudentExamView(QWidget):
    """Student mode: the embedded e-class exam page."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)   # webview fills the screen
        self.web = QWebEngineView()
        self.web.load(QUrl(EXAM_URL))
        layout.addWidget(self.web)


class ProctorView(QWidget):
    """Proctor mode placeholder.

    Module 3 (Ybañez) builds the real dashboard here: live incident feed,
    confidence/severity, time-stamped screenshots tied to student IDs.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel("Proctor Mode — monitoring dashboard goes here (Module 3)")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

# add these to the existing imports at the top of views.py:
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QRadioButton, QButtonGroup,
)


class LoginView(QWidget):
    """AEye's own sign-in screen (identity from login, not biometrics).

    Collects a school-issued ID, a password, and the role, then hands
    them to on_login() so the main window can verify and route.
    """

    def __init__(self, on_login):
        super().__init__()
        self._on_login = on_login

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("AEye — Sign In")
        title.setAlignment(Qt.AlignCenter)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("School-issued ID")

        self.pw_input = QLineEdit()
        self.pw_input.setPlaceholderText("Password")
        self.pw_input.setEchoMode(QLineEdit.EchoMode.Password)  # hide chars

        # Role picker — default to Student.
        self.student_radio = QRadioButton("Student")
        self.proctor_radio = QRadioButton("Proctor")
        self.student_radio.setChecked(True)
        role_group = QButtonGroup(self)               # makes them exclusive
        role_group.addButton(self.student_radio)
        role_group.addButton(self.proctor_radio)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)

        sign_in_btn = QPushButton("Sign In")
        sign_in_btn.clicked.connect(self._handle_sign_in)

        for w in (title, self.id_input, self.pw_input,
                  self.student_radio, self.proctor_radio,
                  self.error_label, sign_in_btn):
            layout.addWidget(w)

    def _handle_sign_in(self):
        role = "student" if self.student_radio.isChecked() else "proctor"
        self._on_login(self.id_input.text(), self.pw_input.text(), role)

    def show_error(self, message):
        self.error_label.setText(message)