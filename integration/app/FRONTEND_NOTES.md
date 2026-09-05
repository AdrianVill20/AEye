# AEye Front-End — Study Notes

> ## ⚠️ OUTDATED — kept as study notes, not as a description of the code
>
> These notes describe the OLD three-tab architecture (Eye Tracking / Posture /
> Head Pose, each with its own `AnalysisTab`). That layout is gone. So are
> `gaze_worker.py`, `headpose_worker.py`, `AnalysisTab` and `CameraSection`.
>
> The app now has one flow: `ExamView` -> calibration -> live tracking, and the
> front camera work all happens in `front_cam_worker.py`. See `../README.md`
> for the current picture. The Qt concepts below (QThread workers, signals,
> QStackedWidget) are still accurate and still worth reading.

> My personal notes for the front-end (`integration/app/`). This is the GUI: login,
> the locked student dashboard, the tabs, and the camera worker threads that paint
> live video + stats into each tab.
>
> Everything here lives in **`integration/app/`**. The detection *logic* the app calls
> lives in sibling folders (`../eye_gaze/`, `../head_pose/`). Line numbers below point
> at the current code — treat them as "roughly here", they drift when I edit.

---

## 1. The 30-second mental model

```
run.bat  ─►  app/main.py  ─►  QApplication + MainWindow
                                   │
                                   │  MainWindow owns a QStackedWidget (a deck of screens)
                                   ▼
        ┌───────────────┬──────────────────────────┬────────────────┐
        │ index 0        │ index 1                   │ index 2         │
        │ LoginView      │ AnalysisDashboard         │ ProctorView     │
        │ (start here)   │ (student mode, LOCKED)    │ (proctor stub)  │
        └───────────────┴──────────────────────────┴────────────────┘
                              │
                              │  3 tabs, each with a Start/Stop button
                              ▼
             ┌──────────────┬──────────────┬──────────────┐
             │ Eye Tracking │ Head Pose    │ Posture      │
             │ GazeWorker   │ HeadPoseWorker│ SideCamera-  │
             │              │              │ Worker        │
             └──────────────┴──────────────┴──────────────┘
                    each worker = a background thread reading a camera
                    and emitting (video frame + stats) back to its tab
```

**The one pattern to really understand:** the GUI never touches a camera directly.
Each tab starts a **worker thread** (a `QThread`). The worker grabs frames, runs ML,
and **emits Qt signals** back to the GUI thread, which paints them. GUI stays responsive;
camera work happens off to the side. Section 6 breaks this down.

---

## 2. File map (what each file is for)

| File | Role | Key classes / functions |
|------|------|------|
| [`main.py`](main.py) | **Entry point.** Builds the window, wires the 3 screens, handles login routing, installs the lockdown, quit shortcut. | `MainWindow`, `main()` |
| [`views.py`](views.py) | **All the screens & widgets.** Login form, the reusable tab, the 3-tab dashboard, worker wiring, proctor stub. | `LoginView`, `AnalysisTab`, `AnalysisDashboard`, `ProctorView` |
| [`auth.py`](auth.py) | **Login check (placeholder).** Right now just "are both fields filled". | `authenticate()` |
| [`session.py`](session.py) | **Who is signed in.** A tiny dataclass tagging the session to a student ID. | `Session` |
| [`keyboard_lock.py`](keyboard_lock.py) | **The lockdown.** Win32 keyboard hook that swallows Alt+Tab / Win key / Alt+F4 etc. | `KeyboardLock` |
| [`gaze_worker.py`](gaze_worker.py) | **Eye Tracking thread.** dlib gaze. | `GazeWorker` |
| [`headpose_worker.py`](headpose_worker.py) | **Head Pose thread.** MediaPipe yaw/pitch/roll. | `HeadPoseWorker` |
| [`posture_worker.py`](posture_worker.py) | **Posture thread.** MediaPipe pose skeleton. | `SideCameraWorker` |

---

## 3. Startup & screen flow — `main.py`

Read [`main.py`](main.py) top to bottom; it's short. Flow:

1. **`main()`** ([main.py:60](main.py)) — makes the `QApplication`, makes `MainWindow`,
   calls `window.show()` so the app **starts on the login screen, windowed** (not locked yet).
2. **`MainWindow.__init__`** ([main.py:16](main.py)):
   - Creates `self.lock = KeyboardLock()` (not installed yet) and `self.session = None`.
   - Builds `QStackedWidget` and adds the 3 screens in order:
     - `LoginView` → index 0 ([main.py:30](main.py))
     - `AnalysisDashboard` (student) → index 1
     - `ProctorView` → index 2
   - `LoginView` is handed `on_login=self.handle_login` — **this is the callback**; when the
     user hits Sign In, the login screen calls back into `MainWindow`.
   - Registers the **quit shortcut `Ctrl+Shift+Q`** ([main.py:34](main.py)) → `QApplication.quit`.
3. **`handle_login(user_id, password, role)`** ([main.py:37](main.py)):
   - Calls `authenticate(...)` ([auth.py](auth.py)). If it fails → `login_view.show_error(...)` and stop.
   - On success → build a `Session` ([session.py](session.py)) and route by role:
     - `"student"` → `enter_student_mode()`
     - anything else → `enter_proctor_mode()`
4. **`enter_student_mode()`** ([main.py:47](main.py)) — this is where the **lockdown** happens:
   - Switch the stack to the dashboard.
   - `setWindowFlag(Qt.WindowStaysOnTopHint, True)` — window stays above others.
   - `self.lock.install()` — turn on the keyboard hook.
   - `self.showFullScreen()`.
5. **`enter_proctor_mode()`** ([main.py:55](main.py)) — just `showMaximized()`, **no lock** (proctors aren't restricted).
6. **On exit** ([main.py:63](main.py)) — `app.aboutToQuit` runs two cleanups:
   - `lock.uninstall()` — remove the keyboard hook (important, or keys stay swallowed system-wide).
   - `student_view.stop_all()` — stop any running camera workers.

---

## 4. The screens — `views.py`

### `LoginView` ([views.py:12](views.py))
- Builds the centered form: ID field, password field (masked, [views.py:27](views.py)),
  **Student / Proctor** radio buttons (Student checked by default, [views.py:31](views.py)),
  a red error label, and the Sign In button.
- `_handle_sign_in` ([views.py:52](views.py)) reads which radio is checked → `'student'` or
  `'proctor'`, then calls `self._on_login(id, pw, role)` — i.e. back into `MainWindow.handle_login`.
- `show_error(msg)` ([views.py:56](views.py)) — sets the red label text.

### `AnalysisTab` ([views.py:59](views.py)) — the reusable tab (understand this one well)
One instance = one detection panel. Built generically so all 3 tabs share it.
- **Left = feed area** ([views.py:64](views.py)): a dark `QFrame`, min 640×480, with a placeholder
  label until a real video widget is dropped in.
- **Right = stats panel** ([views.py:73](views.py)): a title, one `QLabel` per stat field (stored in
  `self.stat_labels` dict, [views.py:77](views.py)), and an owner label.
- Methods I'll actually call:
  - `set_feed_widget(widget)` ([views.py:92](views.py)) — hide the placeholder, drop a live `QLabel`
    (the video surface) into the feed area.
  - `enable_controls(on_start, on_stop)` ([views.py:96](views.py)) — adds the **Start/Stop** button
    that toggles between the two callbacks. **This is what the 3 real tabs use.**
  - `update_stats(dict)` ([views.py:111](views.py)) — for each `field: value` in the dict, updates
    the matching label to `"field: value"`. This is the slot the worker's `stats_ready` connects to.
- ⚠️ `set_launcher` / `_toggle_launch` / `stop_launch` ([views.py:116](views.py)–139) are an
  **alternate** mode that launches an external script via `subprocess` and minimizes the window.
  **None of the 3 current tabs use it** — they all use `enable_controls`. Leftover/optional path.

### `AnalysisDashboard` ([views.py:141](views.py)) — the student screen (a `QTabWidget`)
Builds 3 tabs. Each tab follows the **identical 4-line recipe**:

```python
self.eye_tab   = AnalysisTab('Eye Tracking', '...desc...', ['Direction','Gaze ratio','Blink'], 'Demetillo')
self.eye_video = QLabel(alignment=Qt.AlignCenter)   # the surface frames get painted onto
self.eye_tab.set_feed_widget(self.eye_video)        # put that surface in the tab
self.eye_tab.enable_controls(self.start_gaze, self.stop_gaze)  # wire Start/Stop
self.gaze_worker = None                             # worker created lazily on Start
```

The three tabs & their stat fields:

| Tab | AnalysisTab stat fields | Worker | Owner label |
|-----|------|--------|-------|
| Eye Tracking ([views.py:145](views.py)) | Direction, Gaze ratio, Blink | `GazeWorker` | Demetillo |
| Head Pose ([views.py:150](views.py)) | Direction, Yaw, Pitch, Roll, Landmarks (/478) | `HeadPoseWorker` | Demetillo |
| Posture ([views.py:155](views.py)) | L/R shoulder, L/R wrist, Landmarks (/33) | `SideCameraWorker` | Ybañez |

The **start / stop / show-frame trio** repeats per tab (gaze shown; head & posture identical):
- `start_gaze` ([views.py:164](views.py)): if no worker yet → make `GazeWorker(camera_index=0)`,
  connect its 2 signals, `.start()` the thread.
- `stop_gaze` ([views.py:171](views.py)): `worker.stop()` → `worker.wait()` → set to `None`.
- `_show_gaze_frame(qimg)` ([views.py:177](views.py)): paint the frame onto `self.eye_video`,
  scaled to fit (`QPixmap.fromImage(...).scaled(...)`).
- `stop_all()` ([views.py:212](views.py)): stops all 3 — called on app quit (see main.py:64).

### `ProctorView` ([views.py:217](views.py))
Just a placeholder label — the proctor monitoring dashboard is **Module 3**, not built yet.

### Minor note
`REPO_ROOT` is defined at [views.py:7](views.py) but doesn't appear to be used in this file
(each worker computes its own paths). Harmless, but that's why it's there.

---

## 5. Login check & session (tiny files)

- [`auth.py`](auth.py) — `authenticate(user_id, password, role)` currently returns `True` if
  **both fields are non-empty**. It's a **placeholder**; the real check against the central
  server is Module 3. Not secure yet — noted in its own docstring.
- [`session.py`](session.py) — `Session` is a `@dataclass` with `user_id` and `role`. It exists so
  later modules can tag evidence (screenshots/incidents) with the student's ID. Created in
  `handle_login` ([main.py:41](main.py)).

---

## 6. The worker pattern — the heart of the front end

All 3 workers are the **same shape**. Learn one, you know all three.

Each worker is a **`QThread` subclass** with **two signals**:
```python
class XWorker(QThread):
    frame_ready = Signal(QImage)   # one processed video frame
    stats_ready = Signal(dict)     # the numbers for the side panel
```

Lifecycle:
1. Dashboard does `worker = XWorker(camera_index=0)`, connects signals, calls `.start()`.
2. `.start()` runs the worker's **`run()`** on a background thread.
3. `run()` opens the camera and loops: **read frame → run ML → draw overlay → `emit`** both signals.
4. The GUI thread receives `frame_ready` → paints the picture; receives `stats_ready` → updates labels.
5. `stop()` flips `self._running = False`; the loop ends, camera is released.
   Dashboard then calls `.wait()` to let the thread finish cleanly.

**Why threads?** If the GUI thread read the camera itself, the whole window would freeze
between frames. Signals are Qt's safe way to hand data from a worker thread to the GUI thread.

### Signal wiring (who connects to what)
Set up in each `start_*` method in `views.py`:

| Worker signal | Connected to (slot) | Effect |
|---|---|---|
| `frame_ready(QImage)` | `AnalysisDashboard._show_*_frame` | paints video onto the tab's `QLabel` |
| `stats_ready(dict)` | `AnalysisTab.update_stats` | updates the side-panel labels |

---

## 7. The 3 workers in detail (+ external files they reference)

### `GazeWorker` — [`gaze_worker.py`](gaze_worker.py)  (Eye Tracking tab)
- **Imports teammate logic from a sibling folder:**
  - [gaze_worker.py:7-9](gaze_worker.py): `REVISED_DIR = ../eye_gaze`, added to `sys.path`, then
    `from gaze_core import blink, gaze_ratio` → **[`../eye_gaze/gaze_core.py`](../eye_gaze/gaze_core.py)** (Christian's).
  - [gaze_worker.py:10](gaze_worker.py): `PREDICTOR_PATH = ../eye_gaze/shape_predictor_68_face_landmarks.dat`
    → the **dlib 68-landmark model file** (loaded at [gaze_worker.py:24](gaze_worker.py)).
- Uses `dlib.get_frontal_face_detector()` + the predictor ([gaze_worker.py:23-24](gaze_worker.py)).
- Loop ([gaze_worker.py:33](gaze_worker.py)): grayscale → detect face → per eye compute `blink()` and
  `gaze_ratio()` → average gaze → decide **looking right / center / left** (colored banner drawn on
  the frame, [gaze_worker.py:54-64](gaze_worker.py)) → emit stats `Direction / Gaze ratio / Blink`.
- Camera: **index 0**, opened with `CAP_DSHOW` ([gaze_worker.py:35](gaze_worker.py)). If it can't open,
  emits a "Camera unavailable" stat and returns.

### `HeadPoseWorker` — [`headpose_worker.py`](headpose_worker.py)  (Head Pose tab)
- **References a model file:** [headpose_worker.py:10](headpose_worker.py):
  `MODEL = ../head_pose/face_landmarker.task` → **[`../head_pose/face_landmarker.task`](../head_pose/face_landmarker.task)**
  (MediaPipe FaceLandmarker, Tasks API). This is *why the `head_pose/` folder exists* — just to hold this model.
- Uses the **MediaPipe Tasks API** ([headpose_worker.py:6-7, 26-27](headpose_worker.py)), `RunningMode.VIDEO`,
  and asks for facial transformation matrices.
- Loop ([headpose_worker.py:33](headpose_worker.py)): flip frame (mirror) → detect → take the 3×3 rotation
  from the transform matrix → compute **yaw / pitch / roll** in degrees ([headpose_worker.py:45-50](headpose_worker.py))
  → derive a "looking up/down/left/right/center" `Direction` using a 10° threshold ([headpose_worker.py:51-56](headpose_worker.py))
  → emit stats incl. `Landmarks detected (/478)`.
- Camera: **index 0**, `CAP_DSHOW`.

### `SideCameraWorker` — [`posture_worker.py`](posture_worker.py)  (Posture tab)
- **No external model file.** Uses the **legacy MediaPipe Solutions API** `mp.solutions.pose`
  ([posture_worker.py:7-8](posture_worker.py)), which bundles its own pose model internally.
  (This is why the `../posture/` folder is empty and unused — see the repo notes.)
- `extract_posture(result)` ([posture_worker.py:16](posture_worker.py)) pulls L/R shoulder + L/R wrist
  normalized coords into the stats dict.
- Loop ([posture_worker.py:47](posture_worker.py)): detect pose → draw the skeleton overlay
  ([posture_worker.py:70-76](posture_worker.py)) → **throttle** the coordinate update to every 0.5 s
  ([posture_worker.py:57, 79-85](posture_worker.py)) so numbers don't flicker → emit stats incl.
  `Landmarks detected (/33)`.
- Class default `camera_index=1` ([posture_worker.py:35](posture_worker.py)) **but** the dashboard
  currently constructs it with `camera_index=0` ([views.py:182](views.py)) because there's only one webcam.

---

## 8. Camera gotcha (important while testing)
All three workers currently open **camera index 0** (single webcam). If two tabs run at once they
**fight over the camera**. → **Start one tab, Stop it, then start another.** When the real elevated
side camera is added, change Posture back to `camera_index=1` in `start_posture` ([views.py:182](views.py)).

---

## 9. The lockdown — `keyboard_lock.py`
- `KeyboardLock` installs a **Win32 low-level keyboard hook** (`WH_KEYBOARD_LL`) via `ctypes`.
- The callback ([keyboard_lock.py:78](keyboard_lock.py)) inspects each keydown and **swallows** (returns `1`,
  so Windows never sees it) these combos ([keyboard_lock.py:85-91](keyboard_lock.py)):
  Windows key, Alt+Tab, Alt+Esc, Alt+F4, Ctrl+Esc.
- `install()` ([keyboard_lock.py:98](keyboard_lock.py)) turns it on (called from `enter_student_mode`);
  `uninstall()` ([keyboard_lock.py:106](keyboard_lock.py)) removes it (called on quit).
- **Cannot** block Ctrl+Alt+Del — the OS reserves that. Windows-only by design.
- ⚠️ Always uninstall on exit, or the hook keeps swallowing those keys system-wide.

---

## 10. How to run
- **Double-click [`../run.bat`](../run.bat)**, or from a terminal:
  ```
  ..\app_video\.venv\Scripts\python.exe app\main.py
  ```
- It must use **`app_video\.venv`** — the only environment with PySide6 + dlib + opencv + mediapipe.
- Log in with **any** non-empty ID + password (auth is a placeholder), pick a role, Sign In.
- Exit the locked app with **Ctrl + Shift + Q**.

---

## 11. Dependency map (quick reference)

```
main.py
 ├─ imports keyboard_lock.KeyboardLock
 ├─ imports views.{LoginView, AnalysisDashboard, ProctorView}
 ├─ imports auth.authenticate
 └─ imports session.Session

views.py
 ├─ imports gaze_worker.GazeWorker
 ├─ imports posture_worker.SideCameraWorker
 └─ imports headpose_worker.HeadPoseWorker

gaze_worker.py      ─► ../eye_gaze/gaze_core.py   (blink, gaze_ratio)
                    ─► ../eye_gaze/shape_predictor_68_face_landmarks.dat
headpose_worker.py  ─► ../head_pose/face_landmarker.task
posture_worker.py   ─► (none — mediapipe bundles the pose model)
```

**Front-end owns:** everything in `app/`.
**Front-end depends on (don't need to edit, just don't break the paths):** `../eye_gaze/`, `../head_pose/`.

---

## 12. Where I'd extend things (front-end TODO hooks)
- **Real login** → replace [`auth.py`](auth.py) `authenticate()` (talks to Module 3 server).
- **Proctor dashboard** → fill in [`ProctorView`](views.py) (Module 3).
- **Second camera for posture** → flip `camera_index` back to `1` at [views.py:182](views.py).
- **New detection tab** → copy the 4-line recipe in `AnalysisDashboard.__init__` + add a
  `start_/stop_/_show_*_frame` trio, backed by a new `QThread` worker with the same
  `frame_ready` / `stats_ready` signals.
