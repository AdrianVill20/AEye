from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QRadioButton, QButtonGroup, QTabWidget, QFrame,
)


class LoginView(QWidget):
    """AEye's own sign-in — the STARTING POINT of the GUI. Collects a
    school-issued ID, password, and role, then hands them to on_login()
    so the main window can verify and route to the chosen mode."""

    def __init__(self, on_login):
        super().__init__()
        self._on_login = on_login

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("AEye — Sign In")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("School-issued ID")
        self.id_input.setMaximumWidth(300)

        self.pw_input = QLineEdit()
        self.pw_input.setPlaceholderText("Password")
        self.pw_input.setEchoMode(QLineEdit.EchoMode.Password)  # hide chars
        self.pw_input.setMaximumWidth(300)

        # Role picker (default: Student). QButtonGroup makes them exclusive.
        self.student_radio = QRadioButton("Student")
        self.proctor_radio = QRadioButton("Proctor")
        self.student_radio.setChecked(True)
        role_group = QButtonGroup(self)
        role_group.addButton(self.student_radio)
        role_group.addButton(self.proctor_radio)

        roles = QHBoxLayout()
        roles.setAlignment(Qt.AlignCenter)
        roles.addWidget(self.student_radio)
        roles.addWidget(self.proctor_radio)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setAlignment(Qt.AlignCenter)

        sign_in_btn = QPushButton("Sign In")
        sign_in_btn.setMaximumWidth(300)
        sign_in_btn.clicked.connect(self._handle_sign_in)

        layout.addWidget(title)
        layout.addWidget(self.id_input, alignment=Qt.AlignCenter)
        layout.addWidget(self.pw_input, alignment=Qt.AlignCenter)
        layout.addLayout(roles)
        layout.addWidget(self.error_label)
        layout.addWidget(sign_in_btn, alignment=Qt.AlignCenter)

    def _handle_sign_in(self):
        role = "student" if self.student_radio.isChecked() else "proctor"
        self._on_login(self.id_input.text(), self.pw_input.text(), role)

    def show_error(self, message):
        self.error_label.setText(message)


class AnalysisTab(QWidget):
    """One analysis view: a swappable feed area (left) + stats (right).
    The camera/ML is a teammate's job; here it's a labeled placeholder."""

    def __init__(self, title, camera_desc, stats_fields, owner):
        super().__init__()
        root = QHBoxLayout(self)

        # Left: feed area = the integration point for teammates.
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

        # Right: stats panel (placeholder values for now).
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


class AnalysisDashboard(QTabWidget):
    """Student-mode content: the tabbed 'navigator' of analysis views,
    placed manually (no web page for now)."""

    def __init__(self):
        super().__init__()
        self.addTab(
            AnalysisTab("Eye Tracking",
                        "Front camera — eye feature detection",
                        ["Pupil position", "Left eye", "Right eye",
                         "Gaze direction"], "Demetillo"),
            "Eye Tracking")
        self.addTab(
            AnalysisTab("Head Pose",
                        "Front camera — MediaPipe face mesh",
                        ["Yaw", "Pitch", "Roll",
                         "Landmarks detected (/478)"], "Demetillo"),
            "Head Pose")
        self.addTab(
            AnalysisTab("Posture",
                        "Side camera — MediaPipe pose",
                        ["Shoulder points", "Hand-to-desk distance",
                         "Landmarks detected (/33)"], "Ybañez"),
            "Posture")


class ProctorView(QWidget):
    """Proctor mode placeholder (unlocked). Module 3 (Ybañez) builds the
    real monitoring dashboard here."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel("Proctor Mode — monitoring dashboard goes here (Module 3)")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)