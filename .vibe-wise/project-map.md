# Project Map

## Purpose
AEye watches a student during an online exam with two cameras and flags cheating episodes for a proctor.

## Requirements
- Thesis group project. Teammates: Christian Demetillo, lildelrep@gmail.com. Learner: allainjamesybanez23.
- Practice setup: C:\AEye_Clone, branch answer = answer key from C:\AEye feature/refactor 78674d8 (never edit), branch practice = his rewrite, branch practice-old = work on the older f101892 key. Never push.
- C:\AEye is on branch feature/refactor (78674d8, his refactor commits from 2026-10-18).

## Components
- integration/app/main.py: MainWindow, login -> student dashboard or proctor view (verified)
- integration/app/ui/: screens. views.py (CalibrationView, DotWindow, DetectionView, ProctorView, ScreenBorder, person_path), phone_camera.py (camera pick), gaze_graph.py, hull_view.py, web_tab.py
- integration/app/workers/: QThread camera workers. front_cam_worker.py (iris gaze + head pose), posture_worker.py (side cam pose, YOLO phone + people), head_pose.py
- integration/app/cheat/: calibration_store.py, train_cheat_model.py, cheat_detector.py (Graham scan screen area + isolation forest)
- integration/app/loggers/: posture_logger.py, front_cam_logger.py, cheat_logger.py (batch writes to MySQL)
- integration/app/core/: db_config.py, auth.py, session.py, keyboard_lock.py
- integration/app/database/schema.sql: reference schema (posture_logs, gaze_logs, cheating_events)

## Main Flow
INPUT: front cam + side cam -> pick_cameras -> calibration (DotWindow dots, person outline gate)
  -> calibration_store -> train_cheat_model -> screen_area_<user>.json + cheat_model_<user>.joblib
PROCESS: FrontCamWorker features -> CheatDetector.outside / is_anomaly -> 2 s hold -> cheat event
         SideCameraWorker -> pose + YOLO phone/people -> cheat event
OUTPUT: loggers -> MySQL tables, save_screenshot -> evidence/, ProctorView reads cheating_events, ScreenBorder/GazeGraph/HullView live UI

## Data and Trust Boundaries
- MySQL at 127.0.0.1:3306, database aeye_db (core/db_config.py). Credentials in code, not repeated here.
- Evidence images in integration/app/evidence/

## Build and Deployment
- Python 3.12 venv: C:\AEye\venv\Scripts\python.exe
- Deps: integration/requirements.txt (PySide6, opencv-python, mediapipe, numpy<2, mysql-connector-python, scikit-learn, joblib, ultralytics, pygrabber)
- Launch: integration/app/run.bat (points at ..\.venv, not venv)

## Unknowns
- run.bat uses ..\.venv while the working venv is C:\AEye\venv
- Meaning of the "# a" comment lines added in the refactor commits
