# AEye — Detailed Study Guide (file by file)

A deeper companion to `STUDY_GUIDE.md`. This one goes through **every file**, in
plain language, and ends with a **defense Q&A** so you can answer panel questions
with confidence.

> How to read this: skim **Part 1** (the mental model) first, then read each file's
> section *with the actual `.py` file open beside it*. The "If they ask…" boxes are
> the questions a panelist is most likely to throw at you.

---

## Part 1 — The mental model (read this first)

AEye is a **lockdown exam app** for Windows. There are two kinds of user:

- **Student** — logs in, the screen locks, two cameras watch them take the exam.
- **Proctor** — logs in, sees a table of flagged cheating incidents.

The heart of the system is a **personal, per-student model**. Instead of one fixed
rule for everyone, each student first **calibrates** (reads a passage on camera).
That sample teaches the app what *their* normal looks like. During the exam, the
front camera runs that personal model live and only saves a **confirmed** incident
to the database.

### The flow, end to end

```
LOGIN (main.py + views.LoginView + auth.py)
  │  identity stored in session.py
  │
  ├── student ──► LOCKED DASHBOARD (views.AnalysisDashboard)
  │                 screen locked by keyboard_lock.py + fullscreen
  │                 │
  │                 ├─ Step 1 CALIBRATE (views.CalibrationView)
  │                 │     front_cam_worker records "normal" ──► calibration_store (JSON)
  │                 │
  │                 ├─ Step 2 TRAIN (views.TrainWorker ► train_cheat_model.py)
  │                 │     Isolation Forest learns your normal ──► models/*.joblib
  │                 │
  │                 ├─ Step 3 DETECT (views.DetectionView)
  │                 │     front_cam_worker runs cheat_detector live
  │                 │     confirmed cheat ──► cheat_logger ──► MySQL cheating_events
  │                 │     (side camera logs posture ──► posture_logs)
  │                 │
  │                 └─ WEB (web_tab.py) the actual exam website, locked to one host
  │
  └── proctor ───► ALERTS TABLE (views.ProctorView) reads cheating_events from MySQL
```

### Two ideas that explain the whole design

1. **Threads + signals.** Anything slow (cameras, AI, database) runs on its own
   background **thread** so the window never freezes. Threads talk back to the UI
   with Qt **signals** (`frame_ready`, `cheat_detected`, …). A signal is just
   "something happened, here's the data" — other parts *connect* to it and react.

2. **The model is personal.** It learns *your* normal, so a naturally fidgety
   student and a perfectly still student get **different baselines**. This is the
   fairness argument of the whole thesis.

### The 3 gates (when does it actually flag a cheat?)

A frame is only flagged when **all three** are true (see `front_cam_worker.py`):

1. **Model gate** — the Isolation Forest says this frame is *unusual for you*.
2. **Rule gate** — your eyes are really off-screen (looking down at the desk, or
   gaze to the side). A pure head turn with eyes on screen is ignored.
3. **Time gate** — it stays that way for **2 seconds** (a quick glance doesn't count).

When all three hold, it fires **once per incident** (not once per frame) → one row
in the database.

---

## Part 2 — File by file

Files live in `integration/app/`. The two MediaPipe model files
(`face_landmarker.task`, `pose_landmarker_heavy.task`) live one folder up, in
`integration/head_pose/`.

---

### 🟢 Startup & shell

#### `main.py` — the entry point

**Job:** build the three top-level screens, decide student vs proctor, and wire up
clean-up on exit.

**How it works**
- `main()` first calls `ensure_database()` (makes the MySQL database + tables if
  they don't exist), then starts the Qt application.
- `MainWindow` holds a `QStackedWidget` — a stack of full-page screens where only
  one shows at a time. Three pages are added: **login (0)**, **student dashboard (1)**,
  **proctor view (2)**. It starts on login.
- `handle_login(user_id, password, role)` runs when the login form is submitted. It
  calls `authenticate(...)`; if OK it stores a `Session` and switches to student or
  proctor mode.
- `enter_student_mode()` is where lockdown happens: show the dashboard, set the
  window **always-on-top**, call `self.lock.install()` (the keyboard hook), and go
  **fullscreen**.
- A global shortcut **Ctrl+Shift+Q** quits the app even over the fullscreen exam
  browser (an escape hatch for you during the demo).
- On quit, `aboutToQuit` removes the keyboard hook and stops the camera workers, so
  nothing is left hanging.

> **If they ask "where does the lockdown start?"** → `enter_student_mode()` in
> `main.py`: always-on-top + `KeyboardLock.install()` + `showFullScreen()`.

---

#### `session.py` — who is signed in

**Job:** a tiny data holder (`@dataclass Session`) storing `user_id` and `role`.

**Why it matters:** every piece of evidence (a cheating row, a gaze log, a
calibration file) is tagged with this `user_id`, so incidents can be traced back to
a specific student. It's small but it's the "identity backbone."

---

### 🟢 Identity

#### `auth.py` — login check

**Job:** verify the login. **Right now it's a placeholder** — it only checks that
both fields are non-empty. Real ID/password verification against the central server
is **Module 3** (not yours).

> **If they ask "is your login secure?"** → Be honest and confident: *"Authentication
> is a placeholder in Module 1; real credential checking against the central server
> is scoped to Module 3. Module 1 assumes a valid logged-in identity and focuses on
> the detection pipeline."* This matches Limitation #1 in your paper.

---

### 🟢 The screens — `views.py` (the biggest file)

This one file holds **every window**. Read it as a set of independent classes.
Here's each class and what it does.

| Class | Role |
|---|---|
| `LoginView` | The sign-in form (ID, password, Student/Proctor radio). |
| `AnalysisDashboard` | The locked student dashboard (toolbar + MDI area + Web button + password-protected exit). |
| `ExamView` | The two-step flow container: Calibration → Detection. |
| `CalibrationView` | **Step 1** — record normal behaviour, save JSON, train the model. |
| `ReadingWindow` | The fullscreen passage + countdown shown *during* calibration. |
| `TrainWorker` | Background thread that trains the model without freezing the UI. |
| `DetectionView` | **Step 2** — live tracking; runs detection, shows both cameras + the graph. |
| `ProctorView` | The proctor's alerts table (reads `cheating_events`). |
| `ViewWindow` | An MDI sub-window that **hides** instead of closing (so buttons can reopen it). |

**`LoginView`** — collects ID/password/role and calls back into `main.handle_login`.
Nothing clever; just the form.

**`CalibrationView` (Step 1)** — the important one:
- You pick a **duration** (30/60/120s, default **120**) and a **camera index**, then
  press **Start Calibration**.
- It launches a `FrontCamWorker` with `detect=False` (record-only — nothing goes to
  MySQL) and opens a fullscreen `ReadingWindow` with the passage + countdown.
- Every frame's 5 features arrive via `features_ready` and are collected into a list.
- When the timer ends, `_finish()` checks you captured **≥100 samples**; if so it
  saves them with `calibration_store.save(...)` → `calibration_data/calibration_<user>.json`.
- **Train Model** starts a `TrainWorker` thread → `train_cheat_model.train(user)` →
  saves `models/cheat_model_<user>.joblib`. The status line reports how many samples
  and what % were flagged as unusual.
- **Proceed to Monitoring** stops the calibration camera and switches to `DetectionView`.

**`DetectionView` (Step 2)** — live tracking:
- Pick **front cam** and **side cam** indices, press **Start Tracking**.
- **Front cam** (`FrontCamWorker(detect=True)`): runs the personal model + rule + 2s
  gate. Its per-frame rows go to `gaze_logs`; a confirmed cheat fires `cheat_detected`
  → `CheatEventLogger` → `cheating_events`. It also feeds the live `GazeGraph`.
- **Side cam** (`SideCameraWorker(log_to_db=True)`): shows posture and logs joint
  coordinates to `posture_logs` (~2 rows/sec).
- A running **flag counter** shows in the status line.

> **Key honesty point:** the side camera is **logged but not fused into the decision**.
> The cheat model uses only the front-camera's 5 features. (See "Paper-vs-build gaps".)

**`AnalysisDashboard`** — the locked shell around `ExamView`:
- A toolbar with an "open" button per view (currently just **Front + Side Cam**), a
  **Web** button (opens the exam site fullscreen), **Tile/Cascade** window arrangers,
  and a red **X** exit that requires the password `quit`.
- Uses an **MDI area** (`QMdiArea`) so each view is a movable sub-window. `ViewWindow`
  overrides `closeEvent` to **hide** instead of close, so a closed view can be reopened.

**`ProctorView`** — reads incidents:
- On **Refresh**, it runs `SELECT session_user_id, detected_at FROM cheating_events
  ORDER BY detected_at DESC` (optionally filtered by a student ID) and fills a table,
  newest first.
- Note this is **pull/refresh**, not an automatic push — the proctor clicks Refresh.

> **If they ask "how does the proctor get alerts in real time?"** → Currently the
> proctor **refreshes** to pull the latest rows from MySQL. True server-push is future
> work; the pipeline (student writes → shared DB → proctor reads) is what's built.

---

### 🟢 Lockdown

#### `keyboard_lock.py` — blocks app-switching keys

**Job:** a low-level **Windows keyboard hook** (Win32 `WH_KEYBOARD_LL` via `ctypes`)
that sees every keypress before Windows does and **swallows** escape combos.

**What it blocks:** Windows key, Alt+Tab, Alt+Esc, Alt+F4, Ctrl+Esc.
**What it can't block:** **Ctrl+Alt+Del** — Windows reserves that at the OS level.
This is exactly why your paper says restrictions are "not a guarantee against every
workaround."

**One detail worth knowing:** it keeps a strong reference to the callback
(`self._proc`) so Python's garbage collector can't free it while Windows still holds
a pointer to it — a classic ctypes gotcha.

> **If they ask "so a student could still cheat the lockdown?"** → Yes, and you say so
> up front: no lockdown is absolute (Ctrl+Alt+Del, a second device, etc.). AEye raises
> the effort and, more importantly, **watches behaviour** — the lockdown and the
> detection are complementary, not a single wall.

#### `web_tab.py` — the kiosk exam browser

**Job:** a fullscreen browser locked to **one website** (`eclass.scs.usjr.edu.ph`).

**How it works**
- `LockedPage.acceptNavigationRequest` rejects any navigation to a different host —
  so links/redirects can't leave the exam site. Only the allowed host and its
  sub-domains pass.
- A floating red power button lets you exit — but only after typing the password `quit`.

> **If they ask "what stops them opening Google?"** → The browser physically refuses
> to navigate off the allowed host (`acceptNavigationRequest` returns `False`).

---

### 🟢 Front camera + detection (the core)

#### `front_cam_worker.py` — ⭐ the most important file

**Job:** read **eye gaze + head pose** from the front camera every frame, and (when
`detect=True`) run the 3-gate cheat detection.

**The 5 features it produces** (this is what the model sees):

| Feature | Plain meaning |
|---|---|
| `h_ratio` | **h = horizontal.** Where the eyes point left↔right (iris position between the eye corners). |
| `v_openness` | **v = vertical.** How open the eyes are; compared to a baseline to tell up vs down. |
| `yaw` | Head turned left/right. |
| `pitch` | Head tilted up/down (nodding). |
| `roll` | Head tilted sideways. |

**How it works (per frame)**
1. Grab a frame, mirror it, hand it to MediaPipe **FaceLandmarker** (478 face points).
2. **Eye gaze:** find the iris centre (landmarks 468/473), measure its horizontal
   position between the eye corners → `h_ratio`; measure eye openness → `v_openness`.
   The first 40 frames (`CALIB_FRAMES`) set an openness **baseline** so "up/down" is
   relative to *this* person right now.
3. **Head pose:** from MediaPipe's transformation matrix, compute `yaw`, `pitch`, `roll`.
4. Emit signals: `frame_ready` (video), `features_ready` (the 5 features, for the graph
   and for calibration), `record_ready` (a full row for `gaze_logs`).
5. **If detecting:** feed the 5 features to `cheat_detector.is_anomaly()`. Then apply
   the gates:
   - `looking_down = pitch > 8° or v_dir == 'Down'`
   - `gaze_to_side = eyes not centered`
   - `suspicious = unusual AND (looking_down OR gaze_to_side)`
   - hold `suspicious` for **2 seconds** (`ALERT_SECONDS`) → fire `cheat_detected`
     **once** per episode (rising edge), so you get one DB row per incident.

**Key thresholds** (top of the file): `H_LEFT_THRESH 0.42`, `H_RIGHT_THRESH 0.58`,
`HEAD_TH 5°`, `PITCH_DOWN_TH 8°`, `ALERT_SECONDS 2.0`. These are the knobs you'd tune
to reduce false alarms.

> **If they ask "why iris landmarks and not a gaze-tracking device?"** → MediaPipe runs
> on an ordinary webcam with no special hardware — which is the whole point for
> low-resource Philippine classrooms (your RRL cites this).

#### `cheat_detector.py` — ⭐ scores one frame

**Job:** load the student's personal `.joblib` model and answer one yes/no question:
*is this frame unusual for this student?*

**How it works**
- `load(user_id)` reads `models/cheat_model_<user>.joblib`. If it's missing (or
  scikit-learn isn't installed), it returns a **not-ready** detector whose
  `is_anomaly()` always returns `False` — so detection quietly turns **off** and the
  camera keeps working. This "fail-safe" means the exam is never blocked by a missing
  model.
- `is_anomaly(values)` scales the 5 features and calls `model.predict()`. Isolation
  Forest returns **-1 for an outlier** (unusual) and **1 for normal**.

> **Important for the rename question earlier:** the model reads features by
> **position**, not by name — that's why renaming `h_ratio` in code was safe as long
> as the *order* stayed the same.

#### `train_cheat_model.py` — builds the personal model

**Job:** turn a student's calibration JSON into their Isolation Forest model.

**How it works**
- Loads the calibration samples (needs **≥100**, else it refuses — a too-small sample
  makes a bad baseline).
- `StandardScaler` normalises the features, then `IsolationForest(n_estimators=200,
  contamination=0.03)` learns the shape of "normal." `contamination=0.03` means "assume
  ~3% of the calibration frames are slightly off" — a small allowance for noise.
- Saves `{scaler, model, features}` to `models/cheat_model_<user>.joblib`.

> **If they ask "what is Isolation Forest doing?"** → It's an **unsupervised anomaly
> detector**: it builds random trees and measures how easily each point gets "isolated."
> Normal points (like your calibration) sit in dense regions and are hard to isolate;
> unusual points get isolated quickly and score as outliers. You train it only on
> *normal* behaviour, so at exam time anything far from that normal is flagged.

#### `calibration_store.py` — reads/writes the calibration JSON

**Job:** save and load `calibration_data/calibration_<user>.json`.

**How it works**
- `save()` writes `{user, saved_at, features, count, samples}` **atomically**
  (temp file then replace) so a crash mid-write can't corrupt the file.
- `FEATURES` is defined here *and* in `cheat_detector.py` — they **must match** and be
  in the **same order** (the training script imports this list).

---

### 🟢 Side camera

#### `posture_worker.py` — reads posture from the side camera

**Job:** run MediaPipe **PoseLandmarker** on the side camera, draw the skeleton, and
(when `log_to_db=True`) log **shoulder and wrist** coordinates to `posture_logs`.

**How it works**
- Tracks landmarks 11/12 (shoulders) and 15/16 (wrists). Hidden joints are logged as
  `NULL` rather than skipped, so gaps stay visible in the data.
- A joint counts as visible only above `MIN_VISIBILITY 0.5`; the "Signal" indicator
  reads OK/LOST based on whether shoulders are seen.

> **Honesty note:** posture is **recorded for future work / analysis**, but it does
> **not** feed the cheat decision yet. Say this plainly if asked about Objective 3
> (multimodal fusion) — the *data* is being collected; the *fusion* is the next step.

---

### 🟢 Database writers

All three follow the same **thread + queue + batch** pattern: the camera hands rows to
a queue; a background thread commits them so the camera never waits on MySQL. If the
DB is down, they fail safe (drop rows / return) rather than crashing the exam.

#### `db_config.py` — connection + schema

**Job:** connect to MySQL (`aeye_db`) and, on first run, **create the three tables**
if they don't exist:
- `gaze_logs` — one row per front-cam frame (~30/sec): gaze + head pose values.
- `posture_logs` — shoulder/wrist coordinates from the side cam (~2/sec).
- `cheating_events` — **one row per confirmed incident** (`session_user_id`, `detected_at`).

> **If they ask about the DB config:** it's a local MySQL at `127.0.0.1:3306`, user
> `root`. For a real deployment that'd move to the central server (Module 3) with
> proper credentials.

#### `front_cam_logger.py` — writes `gaze_logs`

Batched inserts (30 rows or every 2s) because the front cam produces ~30 rows/sec.
Has a bounded queue (`MAX_QUEUED 3000`) so a dead database can't slowly eat memory —
it drops samples instead.

#### `posture_logger.py` — writes `posture_logs`

Same pattern, smaller batches (20 rows / 3s), since posture is only ~2 rows/sec.

#### `cheat_logger.py` — writes `cheating_events`

The **only** writer that matters to the proctor. It fires **once per confirmed
episode**, so MySQL stays idle unless a student is actually flagged. Each row is
"who + when."

---

### 🟢 The live graph

#### `gaze_graph.py` — the live values plot

**Job:** a read-only, real-time chart of `h_ratio`, `yaw`, and `pitch`, shown as
**z-scores** (how many standard deviations from *your* baseline). The dashed band is
the "normal" range; spikes outside it are where behaviour is drifting.

**How it works**
- On `set_user()` it loads your baseline — preferably the mean/scale stored inside your
  trained model's scaler, falling back to the raw calibration samples.
- `on_features()` converts each incoming value to a z-score and appends it; a timer
  repaints ~15×/sec (repainting on a timer, not per frame, keeps it smooth).
- Uses Qt Charts if available, else a hand-drawn `QPainter` fallback.

> This is a **visualisation aid** for you and the panel — it's not part of the
> detection decision. It makes the "unusual vs normal" idea visible on screen.

---

## Part 3 — Where data lives (storage map)

| What | Where | Format | Written by |
|---|---|---|---|
| Your "normal" sample | `calibration_data/calibration_<user>.json` | JSON | `calibration_store.save` |
| Your personal model | `models/cheat_model_<user>.joblib` | binary (scaler+model) | `train_cheat_model` |
| Every gaze/head frame | `gaze_logs` table | MySQL | `front_cam_logger` |
| Posture coordinates | `posture_logs` table | MySQL | `posture_logger` |
| Confirmed incidents | `cheating_events` table | MySQL | `cheat_logger` |

**Why two kinds of storage?** Calibration is a reusable per-student file (JSON, easy to
open and retrain from). Live, high-volume data and incidents go to MySQL (queryable,
shared with the proctor). Different jobs, different storage.

---

## Part 4 — Anticipated defense Q&A

**Q: Walk us through what happens from login to a flagged cheat.**
Login (`main.py`) → identity in `session.py` → student dashboard locks
(`keyboard_lock.py` + fullscreen) → **calibrate** (front cam records normal →
`calibration_store` JSON) → **train** (`train_cheat_model` → Isolation Forest
`.joblib`) → **detect** (`front_cam_worker` runs `cheat_detector` + rule + 2s gate) →
confirmed incident → `cheat_logger` → `cheating_events` → proctor refreshes and sees it.

**Q: Why Isolation Forest and not a supervised classifier?**
Because the whole fairness idea is a **personal baseline** — each student judged
against *their own* normal. We have no labelled cheating data per student (you'd have
to make each student cheat to collect it), so we model "normal" unsupervised and flag
deviations. The proctor makes the final call, so we need low false alarms, not
classifier-grade certainty.

**Q: How do you avoid false positives?**
Three gates: the model must find it unusual **and** the eyes must actually be
off-screen **and** it must last 2 seconds — plus the baseline is per-student, so a
naturally fidgety person isn't punished for their normal.

**Q: What are the 5 features and why these?**
`h_ratio` (eyes left/right), `v_openness` (eyes up/down via openness), `yaw`, `pitch`,
`roll` (head turn/nod/tilt). They capture "looking away from the screen" — the visible
behaviour cheating produces — from an ordinary webcam.

**Q: What's your accuracy?**
Be honest: Module 1 delivers the detection **pipeline**; a formal accuracy /
false-positive study needs a small **labelled test set** (scripted honest vs cheating
sessions), which is the evaluation step (Objective 7). Detection is unsupervised; the
labels exist only to *measure* it.

**Q: Does it handle students with glasses / tremors / disabilities?**
That's exactly what the **per-student baseline** is for — it learns *their* normal.
It has not been clinically tested with medical groups (Limitation #3); the fairness
benefit is shown through accuracy and false-positive measurements.

**Q: Can students bypass the lockdown?**
No lockdown is absolute — Ctrl+Alt+Del and second devices exist (Limitation, and
`keyboard_lock.py` can't block Ctrl+Alt+Del). AEye raises the effort *and* watches
behaviour; the two are complementary.

---

## Part 5 — Honest paper-vs-build gaps (so you're not caught off guard)

Know these cold — if a panelist spots one and you already named it, you look prepared.

| Paper says | As built (Module 1) | Your line |
|---|---|---|
| Fuse **both cameras** into one judgment (Obj 3) | Posture is **logged**, not fused; model is front-cam only | "Fusion is the next module; the posture data is already being collected for it." |
| **Facial expression** as a 3rd signal | Only gaze + head pose (5 features) | "Expression is a planned front-camera feature." |
| **Confidence score + severity level** | Fires a binary incident (who + when) | "Severity scoring is a small, planned addition — the model already produces a raw score we can surface." |
| **Screenshot evidence** (Obj 6) | Stores user + timestamp only | "Evidence capture is a planned add-on at the fire point." |
| **30-second** calibration | UI defaults to **120s** (30/60/120 options) | Just know the number so it doesn't surprise you. |
| **Real-time** proctor push | Proctor **refreshes** to pull from MySQL | "The write→shared-DB→read pipeline is built; live push is future work." |
| Learns across **several exams** | Single calibration per session | "Per-session baseline now; longitudinal profile is future work." |

---

## Part 6 — One-line cheat sheet (memorize this)

> **Login → lock → calibrate (record normal) → train (Isolation Forest on your normal)
> → detect (model + eyes-off-screen + 2 seconds) → one row to MySQL → proctor sees it.
> The model is personal, so you're judged against your own normal.**
