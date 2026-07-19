# AEye — Integration

This folder is the **whole integrated app in one place**: your front-end (the
locked kiosk GUI) plus copies of the teammate code it runs. Open this folder,
run it, done — you don't need to touch `desktop_client/`, `Revised_Gaze/`, or
`app_video/` anymore.

> These are **copies**. The originals stay where your teammates keep them; if a
> teammate updates their code, re-copy their files into `eye_gaze/` or
> `posture/` here.

## How to run

**Double-click `run.bat`.**

Or, from a terminal:

```powershell
D:\School_Doc\AEye\app_video\.venv\Scripts\python.exe D:\School_Doc\AEye\integration\app\main.py
```

Either way it uses **`app_video\.venv`** — the only environment with all the
dependencies (PySide6 + dlib + opencv + mediapipe). Running it with any other
Python will fail on a missing package.

## Using the app

1. A sign-in screen appears. Type **any** ID + password (auth is a placeholder),
   pick **Student**, Sign In. The window locks to fullscreen.
2. Open a tab and click **Start**:
   - **Eye Tracking** — Christian's dlib gaze. Shows the camera with a
     red/blue/green banner ("looking right / center / left") + Direction /
     Gaze ratio / Blink.
   - **Posture** — Allain's posture. Shows the skeleton + shoulder/wrist coords.
   - **Head Pose** — MediaPipe FaceLandmarker. Shows the face + live
     Yaw / Pitch / Roll and the landmark count.
3. Click **Stop** to turn a tab's camera off.
4. Exit the locked app with **Ctrl + Shift + Q**.

### ⚠️ Camera notes
- You currently have **one webcam**, so both Eye Tracking and Posture use
  camera **0**. **Run only one tab at a time** — stop one before starting the
  other, or they'll fight over the camera.
- When you add the real **side camera** for posture, change `camera_index=0`
  back to `1` in `app/views.py` (method `start_posture`).
- If a camera shows an **OBS logo**, that's the OBS Virtual Camera on another
  index — not a real webcam.

## What's in here

```
integration/
   run.bat              <- double-click to launch
   README.md            <- this file
   app/                 <- YOUR front-end (the GUI)
      main.py           <- entry point (run this)
      views.py          <- tabs, Start/Stop buttons, wiring
      gaze_worker.py    <- runs Christian's gaze on a thread, into the tab
      posture_worker.py <- runs Allain's posture on a thread, into the tab
      headpose_worker.py<- head pose (yaw/pitch/roll) via FaceLandmarker
      auth.py  session.py  keyboard_lock.py
   eye_gaze/            <- Christian's code (copied from Revised_Gaze/)
      gaze_core.py                          <- his gaze functions
      shape_predictor_68_face_landmarks.dat <- his dlib model
      main.py  blinking.py  threshold.py    <- his originals (reference)
   posture/             <- Allain's code (copied)
      pose_common.py       <- his logic, ported to the MediaPipe Tasks API
      pose_landmarker.task <- pose model
      posture.py           <- his original (reference)
   head_pose/           <- head pose model
      face_landmarker.task <- MediaPipe face model (from app_video/gaze/)
```

## How it fits together (the short version)

- `app/main.py` starts the GUI and the login → student/proctor routing.
- `app/views.py` builds the tabs. Each detection tab has a **Start/Stop** button
  that starts a **worker thread**.
- `app/gaze_worker.py` imports Christian's functions from `eye_gaze/gaze_core.py`,
  runs his per-frame logic, and paints each frame **into the tab**.
- `app/posture_worker.py` does the same with Allain's `posture/pose_common.py`.
- Nothing opens a separate window — everything renders inside the locked kiosk.
