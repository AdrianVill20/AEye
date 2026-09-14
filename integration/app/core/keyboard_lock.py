"""Low-level Windows keyboard hook that blocks app-switching keys.

Uses the Win32 API (WH_KEYBOARD_LL) via ctypes to see every keystroke
before Windows acts on it, and 'swallow' combos like Alt+Tab and the
Windows key so the student can't leave the exam window. This is what
makes AEye's student mode behave like a lockdown browser.

Windows-only by design. It CANNOT block Ctrl+Alt+Del (the OS reserves
that), matching the thesis note that restrictions are not a guarantee
against every workaround.
"""

import ctypes
from ctypes import wintypes

# --- Win32 constants ---------------------------------------------------
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104        # key pressed while Alt is held

VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_CONTROL = 0x11
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F4 = 0x73

LLKHF_ALTDOWN = 0x20          # flag: Alt was down for this key event

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


# The data Windows hands our callback for each key event.
class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]

# Signature of the callback Windows will call: LRESULT (int, WPARAM, LPARAM)
HOOKPROC = ctypes.CFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)

# Declare arg/return types so pointers aren't truncated on 64-bit Windows.
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD
]
user32.CallNextHookEx.restype = wintypes.LPARAM
user32.CallNextHookEx.argtypes = [
    wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
]
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.GetAsyncKeyState.restype = wintypes.SHORT
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


def _ctrl_down():
    # High bit set (0x8000) means the key is currently held down.
    return bool(user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)


class KeyboardLock:
    def __init__(self):
        self._hook = None
        # Keep a strong reference to the callback so Python's garbage
        # collector doesn't free it while Windows still holds a pointer.
        self._proc = HOOKPROC(self._callback)

    def _callback(self, nCode, wParam, lParam):
        # nCode == 0 (HC_ACTION) means there's a real key event to inspect.
        if nCode == 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = kb.vkCode
            alt = bool(kb.flags & LLKHF_ALTDOWN)

            blocked = (
                vk in (VK_LWIN, VK_RWIN)               # Windows key
                or (vk == VK_TAB and alt)              # Alt+Tab
                or (vk == VK_ESCAPE and alt)           # Alt+Esc
                or (vk == VK_F4 and alt)               # Alt+F4
                or (vk == VK_ESCAPE and _ctrl_down())  # Ctrl+Esc (Start menu)
            )
            if blocked:
                return 1  # non-zero = swallow it; Windows never sees the key

        # Everything else: hand off to the next hook / the OS as normal.
        return user32.CallNextHookEx(None, nCode, wParam, lParam)

    def install(self):
        if self._hook:
            return
        self._hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc,
            kernel32.GetModuleHandleW(None), 0
        )

    def uninstall(self):
        if self._hook:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None