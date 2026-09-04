from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QInputDialog, QMessageBox


class LockedPage(QWebEnginePage):
    """A web page that only allows ONE website. Any link, redirect, or URL to a
    different host is rejected, so the browser can never leave the allowed site."""

    def __init__(self, allowed_host, parent=None):
        super().__init__(parent)
        self.allowed_host = allowed_host

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        host = url.host()
        allowed = (
            host == ''                                  # internal pages (about:blank, etc.)
            or host == self.allowed_host
            or host.endswith('.' + self.allowed_host)   # its own sub-domains
        )
        if not allowed:
            return False        # reject -> the browser stays on the current page
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)

    def createWindow(self, _type):
        return self


class WebTab(QWidget):
    """Full-screen kiosk browser locked to ONE site."""

    ALLOWED_HOST = 'eclass.scs.usjr.edu.ph'
    HOME_URL = 'https://eclass.scs.usjr.edu.ph/'

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QWebEngineView()
        self.browser.setPage(LockedPage(self.ALLOWED_HOST, self.browser))
        self.browser.setUrl(QUrl(self.HOME_URL))
        layout.addWidget(self.browser)

        # Power/exit button overlaid on the top-right corner. It is a child of
        # this widget (not in the layout) so it floats above the browser; its
        # position is kept in the corner by resizeEvent, and raise_() keeps it
        # stacked above the web view.
        self.exit_btn = QPushButton('⏻', self)   # ⏻ power symbol
        self.exit_btn.setToolTip('Exit AEye')
        self.exit_btn.setFixedSize(40, 32)
        self.exit_btn.setStyleSheet(
            'QPushButton { color: white; background-color: #c0392b;'
            ' font-size: 16px; font-weight: bold; border-radius: 4px; }'
            ' QPushButton:hover { background-color: #e74c3c; }'
        )
        self.exit_btn.clicked.connect(self._exit_with_password)
        self.exit_btn.raise_()

    def resizeEvent(self, event):
        # Keep the exit button pinned to the top-right corner and above the
        # browser whenever the window size changes (e.g. going full-screen).
        super().resizeEvent(event)
        margin = 10
        self.exit_btn.move(self.width() - self.exit_btn.width() - margin, margin)
        self.exit_btn.raise_()

    def _exit_with_password(self):
        # Ask for the exit password; only "quit" closes the app.
        # QApplication.quit() runs the aboutToQuit cleanup wired up in main.py.
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
