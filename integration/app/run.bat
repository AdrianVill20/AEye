@echo off
REM Launch the AEye integration app using the Python 3.12 venv that has all
REM dependencies (PySide6, mediapipe, opencv, mysql-connector, scikit-learn).
REM Run this instead of "python main.py" so modules are always found.
REM Works from anywhere and by double-click (paths are relative to this file).
"%~dp0..\..\venv\Scripts\python.exe" "%~dp0main.py" %*
