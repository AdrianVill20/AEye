# AEye — Integration

This folder is the whole app. Open it, run it, done.

## How to run

**Double-click `run.bat`.**

Or, from a terminal at the repo root:

```powershell
.\integration\.venv\Scripts\python.exe integration\app\main.py
```

Either way it uses **`integration\.venv`** (Python 3.12) — the environment with
all the dependencies, kept next to the project. Running it with any other Python
will fail on a missing package.

First time on a new machine, build that environment:

```powershell
.\integration\setup_env.ps1
```

MySQL must be running on `127.0.0.1:3306` with user `root` / password `root`
(see `app/db_config.py`). The app creates the `aeye_db` database and its tables
on startup if they are missing.

## Using the app

1. A sign-in screen appears. Type **any** ID + password (auth is a placeholder),
   pick **Student**, Sign In. The window locks to fullscreen.
2. Click **Front + Side Cam** in the toolbar. That opens the exam flow:
   - **Step 1 — Calibration.** Read the passage on screen while the front camera
     records what normal looks like for you. Saves to
     `app/calibration_data/calibration_<id>.json`.
   - **Train.** The calibration screen prints the exact command when it finishes.
     Run it from `integration/app`:
     ```powershell
     ..\.venv\Scripts\python.exe train_cheat_model.py --user <id>
     ```
     That writes `app/models/cheat_model_<id>.joblib`.
   - **Step 2 — Live Tracking.** Press *Proceed to Monitoring*, then
     *Start Tracking*. The front camera runs detection; the side camera shows
     posture. Confirmed episodes are written to MySQL.
3. **Web** opens the e-class exam page fullscreen.
4. Exit with the red **X** button (password: `quit`) or **Ctrl + Shift + Q**.

Sign in as **Proctor** instead to see the alerts table.

### ⚠️ Camera notes
- With **one webcam**, both the front and side workers open camera **0** and will
  fight over it. Set the side camera back to `camera_index=1` in
  `app/views.py` (`DetectionView._start`) once the real side camera is plugged in.
- If a camera shows an **OBS logo**, that's the OBS Virtual Camera on another
  index — not a real webcam.

## What's in here

```
integration/
   run.bat                 <- double-click to launch
   setup_env.ps1           <- builds integration/.venv
   requirements.txt
   README.md               <- this file
   app/
      main.py              <- entry point: login -> student / proctor
      views.py             <- all screens (login, calibration, tracking, proctor)
      front_cam_worker.py  <- front camera thread: eye gaze + head pose + detection
      posture_worker.py    <- side camera thread: upper-body posture
      cheat_detector.py    <- loads the student's Isolation Forest model
      train_cheat_model.py <- trains it from the calibration JSON (run by hand)
      calibration_store.py <- reads / writes the calibration JSON
      front_cam_logger.py  <- gaze_logs writer (batched)
      posture_logger.py    <- posture_logs writer (batched)
      cheat_logger.py      <- cheating_events writer (one row per episode)
      db_config.py         <- MySQL connection + table bootstrap
      web_tab.py           <- the locked e-class browser window
      auth.py  session.py  keyboard_lock.py
      calibration_data/    <- per-student calibration JSON
      models/              <- per-student trained models
      database/schema.sql  <- reference copy of the schema
   head_pose/
      face_landmarker.task       <- MediaPipe face model (478 landmarks)
      pose_landmarker_heavy.task <- MediaPipe pose model (33 landmarks)
```

## How it fits together (the short version)

- `main.py` starts the GUI and the login → student/proctor routing.
- `views.py` builds every screen. `ExamView` is calibration then live tracking.
- `front_cam_worker.py` reads the front camera, turns each frame into five
  numbers (`h_ratio`, `v_openness`, `yaw`, `pitch`, `roll`), and paints the frame
  into the window.
- During calibration those five numbers are collected into a JSON file.
  `train_cheat_model.py` fits an Isolation Forest on them — the student's
  personal "this is normal" model.
- During tracking, `cheat_detector.py` scores each frame against that model. A
  flag needs all three: the model says unusual, the eyes are off screen, and it
  holds for 2 seconds.
- `posture_worker.py` runs the side camera and logs shoulder/wrist positions.
  It does **not** feed the detection decision yet.
- `ProctorView` reads the `cheating_events` table.

## Databases

| Table | Written by | When |
|---|---|---|
| `gaze_logs` | `FrontCamLogWriter` | every frame while tracking |
| `posture_logs` | `PostureLogWriter` | ~2 rows/sec while tracking |
| `cheating_events` | `CheatEventLogger` | once per confirmed episode |
