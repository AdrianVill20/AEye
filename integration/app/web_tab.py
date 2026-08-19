from PySide6.QtCore import Qt, QUrl, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QWidget


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
