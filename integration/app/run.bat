@echo off
REM Launch the AEye integration app using the Python 3.12 venv that has all
REM dependencies (PySide6, dlib, mediapipe 0.10.x, opencv, mysql-connector).
REM Run this instead of "python main.py" so modules are always found.
REM Works from anywhere and by double-click (paths are relative to this file).
"%~dp0..\..\app_video\.venv\Scripts\python.exe" "%~dp0main.py" %*
