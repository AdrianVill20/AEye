# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AEye is a Windows desktop app for proctoring computer-based exams: it reads a student's
eye gaze, head pose, and posture from two cameras and surfaces behavior signals to a
human proctor. Full product description is in the root `README.md`. This is a school
thesis project, not a production codebase — expect placeholder auth, hardcoded DB
credentials, and unfinished modules (proctor dashboard, real scoring pipeline).

## Repo layout — only `integration/` is live

The repo has several top-level folders because each teammate originally built their
piece standalone. **`integration/` is the actual app** — everything else is either the
original per-teammate source (kept as reference) or superseded:

- `integration/` — the real app. Work here.
  - `integration/app/` — the PySide6 GUI (front-end, owned by the integration/desktop dev)
  - `integration/eye_gaze/` — copy of the dlib-based gaze code (originally `Revised_Gaze/`)
  - `integration/head_pose/` — just holds the MediaPipe `face_landmarker.task` model file
- `desktop_client/`, `app_video/`, `Revised_Gaze/` — earlier/standalone versions of the
  same pieces. Not run directly anymore; don't edit expecting it to affect the app.
  `integration/eye_gaze/` and the model files under `integration/` are **manual copies**
  of code from these folders — if a teammate updates their original, someone re-copies
  it into `integration/`, there's no build step or symlink doing this automatically.
- `venv/` at repo root — an unrelated/leftover Python 3.12 venv, **not** what the app
  uses (see Running below).

## Running the app

```
integration\run.bat
```

(or `integration\app\run.bat`, which is the one with correct dependency comments)

This runs `integration/app/main.py` using **`app_video\.venv`** — the only environment
with the full dependency set: PySide6, dlib, opencv-python, mediapipe, mysql-connector-python.
There is no tracked `requirements.txt`; that venv must already have these installed.
Running with any other Python/venv will fail on a missing package. `app_video/.venv`
itself is gitignored and not present until someone creates it locally.

Login accepts **any non-empty ID + password** — `integration/app/auth.py::authenticate()`
is a placeholder that doesn't check a real account yet.

No automated test suite exists in this repo. `app_video/live_gaze_test.py` and
`webcam_pupil_test.py` are manual/interactive scripts, not pytest tests.

## Database

MySQL, database `aeye_db`, schema in `integration/app/database/schema.sql` (two tables:
`gaze_logs`, `posture_logs`, both keyed loosely by `session_user_id` + `captured_at`).
Connection config is hardcoded in `integration/app/db_config.py` (`127.0.0.1:3306`,
`root`/`root`) — no env vars, no secrets management. `GazeLogWriter` / `PostureLogWriter`
(`gaze_logger.py`, `posture_logger.py`) are `QThread`s that drain a queue and INSERT in
batches so DB writes never block the camera/UI threads.

Note: DB-logging wiring between the camera workers and these log writers is actively
in flux (see `posture_worker.py`'s uncommitted changes and the `session_user_id` param
some worker constructors accept inconsistently) — check current worker `__init__`
signatures against how `views.py` instantiates them before assuming logging works
end-to-end.

## Architecture — `integration/app/`

`FRONTEND_NOTES.md` and `HEADPOSE_EXPLAINED.md` in this folder are a teammate's personal
study notes. They're useful for the worker-thread pattern and headpose math, but the
tab layout they describe (plain `QTabWidget`, 3 fixed tabs) is **stale** — the current
`views.py` uses an MDI-based dashboard instead. Trust the code over those notes for
anything about `views.py` structure.

**Screen flow** (`main.py`): `QStackedWidget` with `LoginView` → `AnalysisDashboard`
(student, locked fullscreen) or `ProctorView` (proctor, unbuilt placeholder — Module 3).
`enter_student_mode()` is where the exam lockdown kicks in: always-on-top + a Win32
low-level keyboard hook (`keyboard_lock.py`, blocks Alt+Tab/Win/Alt+F4/Ctrl+Esc, cannot
block Ctrl+Alt+Del) + fullscreen. `Ctrl+Shift+Q` is the only way out. The hook **must**
be uninstalled on quit (`app.aboutToQuit`) or it keeps swallowing those keys system-wide.

**`AnalysisDashboard`** (`views.py`) is a `QMdiArea` — each analysis view is its own
movable/closable-but-not-really sub-window (`ViewWindow` overrides `closeEvent` to hide
instead of close, so toolbar buttons can reopen it). Views: Eye Tracking, Head Pose,
Posture, Head + Gaze (combined), Web (an embedded Chromium browser via `QWebEngineView`,
for viewing the school's e-class exam page).

**The worker-thread pattern — the one thing to understand well.** The GUI never touches
a camera directly. Each detection view starts a `QThread` subclass that opens a camera,
loops (read frame → run the ML model → draw an overlay → emit), and exposes exactly two
signals:

```python
frame_ready = Signal(QImage)   # processed video frame, painted onto a QLabel
stats_ready = Signal(dict)     # {field: value} for the side/status panel
```

`stop()` flips a `self._running` flag; the caller then calls `.wait()` after `stop()` to
join the thread before releasing the camera. The three workers:

- `GazeWorker` (`gaze_worker.py`) — dlib 68-landmark detector, imports
  `blink`/`gaze_ratio` from `../eye_gaze/gaze_core.py`. Camera opened with `CAP_DSHOW`.
- `HeadPoseWorker` (`headpose_worker.py`) — MediaPipe Tasks `FaceLandmarker` loaded from
  `../head_pose/face_landmarker.task`; derives yaw/pitch/roll from the rotation matrix,
  applies a 10° threshold before labeling a direction.
- `SideCameraWorker` (`posture_worker.py`) — legacy `mp.solutions.pose` API (bundles its
  own model, no external `.task` file); throttles stat updates to every 0.5s so numbers
  don't flicker, though frames still stream every loop.

**`CameraSection`** (`views.py`) is the reusable widget behind the Posture and
Head+Gaze tabs: a camera-index dropdown + Start/Stop + video label + status line, wired
to whichever worker class is passed in. Eye Tracking and Head Pose tabs instead use the
older `AnalysisTab` widget with fixed camera index 0 wired directly in
`AnalysisDashboard.start_gaze`/`start_headpose`.

**Camera-sharing gotcha**: most views default to camera index 0 because dev machines
often have one webcam. Running two camera-consuming views at once makes them fight over
the same device — `CameraSection`'s dropdown (indices 0-3) exists so testers can point
one view at a second physical camera when available. When wiring up the real elevated
side camera for posture, the intended index is 1, not 0.

## Workflow: Grill-Me

Whenever I ask you to build, create, make, code, or design anything non-trivial, you MUST follow this process:

- **Interview First:** Do not write code yet. Ask me 10-15 targeted questions in batches of 4-6 questions at a time. Number the questions so I can easily answer them.
- **Synthesize:** After the interview, generate a short spec covering: Goal, Users, Must-haves, Out of scope, Constraints, and Assumptions.
- **Confirm & Build:** Get my explicit approval on the spec before writing any code.
