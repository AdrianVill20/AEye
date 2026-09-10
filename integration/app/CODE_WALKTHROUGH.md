# AEye — Code Walkthrough (what the actual code does, file by file)

This explains the **real code** in every file, block by block, in the order the
program runs. Read each section with the matching `.py` file open beside it.

**Plain-name decoder** (used everywhere):
`h_ratio` = eyes left/right · `v_openness` = how open the eyes are (used for up/down) ·
`yaw` = head turn · `pitch` = head nod · `roll` = head tilt.

**Run order:** `main.py` → `auth.py` → `session.py` → `views.py` → `keyboard_lock.py`
→ `web_tab.py` → `front_cam_worker.py` → `cheat_detector.py` → `train_cheat_model.py`
→ `calibration_store.py` → `posture_worker.py` → `front_cam_logger.py` /
`posture_logger.py` / `cheat_logger.py` → `db_config.py` → `gaze_graph.py`.

---

## 1. `main.py` — the launcher

The file you run. Read it bottom-up: the real start is `main()`.

```python
def main():
    ensure_database()                                     # make MySQL db + tables if missing
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)  # needed by the web browser widget
    app = QApplication(sys.argv)                          # the Qt application
    window = MainWindow()                                 # build every screen
    app.aboutToQuit.connect(window.lock.uninstall)        # on exit: remove keyboard hook
    app.aboutToQuit.connect(window.student_view.stop_all) # on exit: stop cameras/threads
    window.show()                                         # show the login screen
    sys.exit(app.exec())                                  # start the event loop (runs until quit)
```
`app.exec()` is the **event loop** — the line that keeps the app alive, dispatching
clicks and camera frames until you quit.

**`MainWindow.__init__`** builds a slide-deck of screens:
```python
self.lock = KeyboardLock()          # the Windows key-blocker (built, not yet active)
self.stack = QStackedWidget()       # deck of full-page screens; one visible at a time
self.stack.addWidget(self.login_view)    # index 0  ← starts here
self.stack.addWidget(self.student_view)  # index 1
self.stack.addWidget(self.proctor_view)  # index 2
```
`self.login_view = LoginView(on_login=self.handle_login)` hands the form a **callback**,
so submitting the form calls back into this file.

**`handle_login`** is the fork in the road:
```python
if not authenticate(user_id, password, role):   # auth.py
    self.login_view.show_error(...); return
self.session = Session(user_id=user_id.strip(), role=role)   # remember who
if role == "student": self.enter_student_mode()
else:                 self.enter_proctor_mode()
```

**`enter_student_mode`** = the lockdown moment (three things together):
```python
self.stack.setCurrentWidget(self.student_view)    # switch to the dashboard
self.setWindowFlag(Qt.WindowStaysOnTopHint, True) # float above other windows
self.lock.install()                               # start swallowing Alt+Tab etc.
self.showFullScreen()                             # take the whole screen
```
Proctor mode skips all that — it only `showMaximized()`, no lock.

**Remember:** `main.py` is the conductor. It builds screens, decides student vs
proctor, toggles lockdown, and guarantees clean-up on quit. It does no camera work.

---

## 2. `auth.py` — the login check

The whole file is one function:
```python
def authenticate(user_id, password, role):
    return bool(user_id.strip()) and bool(password.strip())
```
It only checks that **both fields are filled**. It's a **placeholder** — real ID
verification against the central server is Module 3. This lines up with Limitation #1
in your paper: Module 1 trusts the logged-in identity and focuses on detection.

---

## 3. `session.py` — who is signed in

A tiny data holder:
```python
@dataclass
class Session:
    user_id: str
    role: str   # "student" or "proctor"
```
Small but important: every log row, calibration file, and cheating incident is tagged
with this `user_id`, so evidence can always be traced back to a specific student.

---

## 4. `views.py` — all the screens (the big file)

One file, many classes. Here is each, in the order they matter.

### `LoginView` — the sign-in form
Builds the ID box, password box, Student/Proctor radios, and a Sign In button. On
submit:
```python
def _handle_sign_in(self):
    role = 'student' if self.student_radio.isChecked() else 'proctor'
    self._on_login(self.id_input.text(), self.pw_input.text(), role)  # calls main.handle_login
```
`show_error()` just sets the red label. That's the whole screen.

### `ViewWindow` — a sub-window that hides instead of closing
```python
def closeEvent(self, event):
    event.ignore()   # don't really close...
    self.hide()      # ...just hide, so a toolbar button can reopen it later
```
Used so closing a panel doesn't destroy it for good.

### `TrainWorker` — trains the model off the UI thread
```python
class TrainWorker(QThread):
    done = Signal(object); failed = Signal(str)
    def run(self):
        try:
            from train_cheat_model import train   # heavy import happens here, off the UI
            self.done.emit(train(self.user))
        except Exception as exc:
            self.failed.emit(str(exc))
```
Training imports scikit-learn and fits a model — slow. Doing it on a **background
thread** keeps the window from freezing. When done, it emits `done` (with the result)
or `failed` (with a message), which the UI listens for.

### `ReadingWindow` — the fullscreen passage shown during calibration
A white fullscreen page with the passage, a `QProgressBar`, and a "Ns left" countdown.
`set_progress(elapsed, remaining)` updates the bar each second. Its `closeEvent` fires
an `on_cancel` callback **once**, so closing early cleanly cancels calibration.

### `CalibrationView` — Step 1: record "normal", then train
This is the important screen. Walking the flow:

**Start ([`_start`]):**
```python
self.worker = FrontCamWorker(camera_index=cam, session_user_id=self._user_id, detect=False)
self.worker.frame_ready.connect(self._show_frame)     # show the camera
self.worker.features_ready.connect(self._collect)     # gather the 5 features per frame
self.worker.start()
self._reader = ReadingWindow(self._passage_text, self._remaining, on_cancel=self._cancel_reading)
self._reader.showFullScreen()
self._timer.start()   # 1-second ticks
```
`detect=False` means **record only** — nothing is written to MySQL; we just collect
features. `_collect` appends each frame's features to a list.

**Each second ([`_tick`]):** advance the progress bar and countdown; when time's up call
`_finish`.

**Finish ([`_finish`]):**
```python
if len(self._samples) < 100:                 # too few → warn, save nothing
    self.status.setText('Only N samples...'); return
calibration_store.save(self._user_id, self._samples)   # → calibration_<user>.json
self.train_btn.setEnabled(True)              # now you can train
```
The **≥100 samples** rule stops a bad, too-small baseline.

**Train ([`_train`]):** starts a `TrainWorker`; on success `_train_done` reports
"trained on N samples (X% flagged)". **Proceed** stops the camera and switches to the
detection screen.

**Defaults to know:** duration options are `30/60/120` with **120 selected** (your
paper says 30s — know this mismatch). Camera index default `0`.

### `DetectionView` — Step 2: live tracking + detection
**Start ([`_start`]):** this is where everything gets wired together.
```python
# FRONT camera = detection
self.front = FrontCamWorker(camera_index=front_idx, session_user_id=user_id, detect=True)
self.front.frame_ready.connect(self._show_front)       # video
self.front.cheat_detected.connect(self._on_cheat)      # a confirmed incident
self.graph.set_user(user_id)
self.front.features_ready.connect(self.graph.on_features)  # feed the live graph

# every front-cam frame → gaze_logs
self.front_log = FrontCamLogWriter(); self.front_log.start()
self.front.record_ready.connect(self.front_log.enqueue)

# SIDE camera = posture, logged to posture_logs
self.side = SideCameraWorker(camera_index=side_idx, session_user_id=user_id, log_to_db=True)

# MySQL writer, only used when a cheat fires
self.logger = CheatEventLogger(); self.logger.start()
self.front.start(); self.side.start()
```
**On a confirmed cheat ([`_on_cheat`]):**
```python
self.logger.enqueue(event)        # write ONE row to cheating_events
self._count += 1
self.status.setText(f'Cheating flagged at {ts} (total: {self._count})')
```
`stop_all()` stops both cameras and all three writer threads and re-enables the camera
pickers. **Key point:** the side camera is *logged but not used in the decision* — the
cheat model reads only the front camera's 5 features.

### `ExamView` — glues Step 1 → Step 2
A `QStackedWidget` holding `CalibrationView` (index 0) then `DetectionView` (index 1).
When calibration calls `on_proceed`, `_go_detect` flips to index 1.

### `AnalysisDashboard` — the locked student shell
Builds a toolbar and an **MDI area** (`QMdiArea`, movable sub-windows). Each view gets
a sub-window (`ViewWindow`, hides-not-closes) and an "open" button. Extra buttons:
- **Web** → `_open_web` creates `WebTab` and shows it fullscreen (the actual exam site).
- **Tile / Cascade** → arrange the sub-windows.
- **X (exit)** → `_exit_with_password`: only the password `quit` calls
  `QApplication.quit()` (which triggers the clean-up wired in `main.py`).

### `ProctorView` — the alerts table
**Refresh ([`refresh`]):**
```python
if name:  cur.execute('SELECT session_user_id, detected_at FROM cheating_events '
                      'WHERE session_user_id = %s ORDER BY detected_at DESC', (name,))
else:     cur.execute('SELECT session_user_id, detected_at FROM cheating_events '
                      'ORDER BY detected_at DESC')
rows = cur.fetchall()
```
Then it fills a 2-column table (Student, Detected at), newest first. Note it's
**pull/refresh** — the proctor clicks Refresh; there is no automatic push.

---

## 5. `keyboard_lock.py` — block app-switching keys

A low-level **Windows keyboard hook** (`WH_KEYBOARD_LL`) via `ctypes`. It sees every
keypress before Windows and swallows escape combos.

```python
def _callback(self, nCode, wParam, lParam):
    if nCode == 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
        kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk = kb.vkCode; alt = bool(kb.flags & LLKHF_ALTDOWN)
        blocked = (vk in (VK_LWIN, VK_RWIN)           # Windows key
                   or (vk == VK_TAB and alt)          # Alt+Tab
                   or (vk == VK_ESCAPE and alt)       # Alt+Esc
                   or (vk == VK_F4 and alt)           # Alt+F4
                   or (vk == VK_ESCAPE and _ctrl_down()))  # Ctrl+Esc (Start menu)
        if blocked:
            return 1     # non-zero = swallow it; Windows never sees the key
    return user32.CallNextHookEx(None, nCode, wParam, lParam)   # else pass through
```
- `install()` registers the hook; `uninstall()` removes it (called on quit).
- `self._proc = HOOKPROC(self._callback)` is kept as a **strong reference** so Python's
  garbage collector can't free the callback while Windows still points at it (a classic
  ctypes bug if you forget).
- **Can't block Ctrl+Alt+Del** — the OS reserves it. This is why your paper says the
  lockdown isn't a guarantee against every workaround.

---

## 6. `web_tab.py` — the kiosk exam browser

A fullscreen browser locked to **one site**.
```python
class LockedPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        host = url.host()
        allowed = (host == '' or host == self.allowed_host
                   or host.endswith('.' + self.allowed_host))
        if not allowed:
            return False      # reject → browser stays put, can't leave the site
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)
```
`ALLOWED_HOST = 'eclass.scs.usjr.edu.ph'`. Any link/redirect to another host is refused,
so a student can't browse to Google. A floating red power button exits — but only after
typing the password `quit`.

---

## 7. `front_cam_worker.py` — ⭐ read eyes + head, run detection

The most important file. It runs on its own thread; each camera frame does this:

**(a) Detect the face:**
```python
result = landmarker.detect_for_video(mp_img, timestamp_ms)  # MediaPipe, 478 face points
```

**(b) Eye gaze** — iris position and openness:
```python
h_ratio = _get_h_ratio(lm, (r_cx, r_cy), RIGHT_EYE_CORNERS, w)   # eyes left↔right
self._prev_h = _ema(self._prev_h, h_ratio)   # smooth it (exponential moving average)
avg_open = (r_open + l_open) / 2.0            # how open the eyes are
```
The first 40 frames set an **openness baseline** so "Up/Down" is judged relative to
*this* person right now:
```python
if self._baseline is None:
    self._calib_openness.append(avg_open)
    if len(self._calib_openness) >= CALIB_FRAMES:      # 40 frames
        self._baseline = np.mean(self._calib_openness)
else:
    diff = avg_open - self._baseline
    v_dir = "Up" if diff > 0.01 else "Down" if diff < -0.01 else "Center"
```

**(c) Head pose** — from MediaPipe's transformation matrix:
```python
R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
yaw   = math.degrees(math.atan2(-R[2, 0], sy))
pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
```
(These are standard rotation-matrix → angle conversions; you don't need the trig, just
"it turns the 3×3 rotation into turn/nod/tilt degrees.")

**(d) Send data out** via signals: `frame_ready` (video), `features_ready` (the 5
features → graph + calibration), `record_ready` (a full row → `gaze_logs`).

**(e) Detection — the 3 gates** (only when `detect=True`):
```python
unusual = self._detector.is_anomaly((h_ratio, v_openness, yaw, pitch, roll))  # MODEL gate
looking_down = (pitch > PITCH_DOWN_TH) or (v_dir == 'Down')   # eyes down at desk
gaze_to_side = (h_dir != 'Center')                            # eyes off to a side
suspicious = unusual and (looking_down or gaze_to_side)       # RULE gate

if suspicious:                              # TIME gate: must persist 2s
    if self._anom_since is None: self._anom_since = now
    held = now - self._anom_since
else:
    self._anom_since = None; held = 0.0

if held >= ALERT_SECONDS:                   # 2.0 seconds
    if not self._alert_active:              # fire ONCE per episode (rising edge)
        self._alert_active = True
        self.cheat_detected.emit({'user': ..., 'timestamp': datetime.now()})
```
`self._alert_active` is the trick that gives **one DB row per incident** instead of one
per frame. The tunable knobs are the thresholds at the top: `H_LEFT_THRESH 0.42`,
`H_RIGHT_THRESH 0.58`, `PITCH_DOWN_TH 8`, `ALERT_SECONDS 2.0`.

---

## 8. `cheat_detector.py` — score one frame

Loads the student's personal model and answers "is this frame unusual for them?"
```python
@classmethod
def load(cls, user_id=None):
    try:
        bundle = joblib.load(user_model_path(user_id))     # models/cheat_model_<user>.joblib
        return cls(scaler=bundle['scaler'], model=bundle['model'])
    except Exception:
        return cls()          # not-ready detector → is_anomaly() always False (fail-safe)

def is_anomaly(self, values):
    if not self.ready: return False
    x = self.scaler.transform(np.asarray(values, float).reshape(1, -1))
    return self.model.predict(x)[0] == -1   # Isolation Forest: -1 = outlier
```
Two things to say in a defense:
- **Fail-safe:** no model → detection silently off, exam never blocked.
- **Positional features:** it reads the 5 values by **order**, not name — that's why
  renaming the feature variables was safe as long as the order stayed the same.

---

## 9. `train_cheat_model.py` — build the personal model

Turns the calibration JSON into an Isolation Forest.
```python
data = load_calib(user)
samples = data.get('samples', [])
if len(samples) < 100:
    raise ValueError('too few samples...')                     # need a real baseline
X = np.array([[s[f] for f in FEATURES] for s in samples])      # rows of 5 features
scaler = StandardScaler().fit(X)                               # normalise
model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
model.fit(scaler.transform(X))                                 # learn "normal"
joblib.dump({'scaler': scaler, 'model': model, 'features': FEATURES}, out)
```
- **Isolation Forest, in one sentence:** it builds random trees; points that get
  "isolated" quickly are outliers. You train it only on *normal* behaviour, so anything
  far from that at exam time scores as unusual.
- `contamination=0.03` = "assume ~3% of calibration frames are slightly off" (noise
  allowance). `random_state=42` makes training repeatable.

---

## 10. `calibration_store.py` — read/write the calibration JSON

```python
def save(user_id, samples):
    payload = {'user': ..., 'saved_at': ..., 'features': FEATURES,
               'count': len(samples), 'samples': samples}
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f: json.dump(payload, f)
    tmp.replace(path)          # atomic: write temp, then rename — a crash can't corrupt it
```
`FEATURES` is defined here **and** in `cheat_detector.py`; they must match and stay in
the **same order** (training imports this list).

---

## 11. `posture_worker.py` — read posture from the side camera

Runs MediaPipe **PoseLandmarker** on the side camera, draws the skeleton, and logs
shoulder/wrist coordinates.
```python
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST = 11, 12, 15, 16
def db_values(landmark):
    if not is_visible(landmark):                 # below 0.5 visibility
        return (None, None, landmark.visibility) # log NULLs so gaps stay visible
    return (landmark.x, landmark.y, landmark.visibility)
```
When `log_to_db=True`, every ~0.5s it builds a row and enqueues it to `posture_logs`.
**It does not feed the cheat decision** — this is data collected for future fusion /
analysis (say this if asked about Objective 3).

---

## 12–14. The three database writers (same pattern)

All three are background threads with a **queue + batching**: the camera drops rows in a
queue; the thread commits them so the camera never waits on MySQL. A dead DB fails safe.

- **`front_cam_logger.py`** → `gaze_logs`. Big volume (~30/sec), so it batches 30 rows
  or every 2s, with a bounded queue (`MAX_QUEUED 3000`) that **drops** samples rather
  than eat memory if the DB stalls.
- **`posture_logger.py`** → `posture_logs`. Same idea, smaller (20 rows / 3s).
- **`cheat_logger.py`** → `cheating_events`. The one the proctor sees. Fires **once per
  confirmed episode**, so MySQL stays idle unless someone is actually flagged:
```python
INSERT_SQL = "INSERT INTO cheating_events (session_user_id, detected_at) VALUES (%s, %s)"
```

The batch loop (same in the two high-volume writers):
```python
record = self._queue.get(timeout=0.5)
batch.append(record)
if len(batch) >= BATCH_SIZE or time.time() - last_flush > FLUSH_SECONDS:
    cursor.executemany(INSERT_SQL, batch); conn.commit(); batch.clear()
```

---

## 15. `db_config.py` — connection + schema

Connects to a local MySQL (`aeye_db` at `127.0.0.1:3306`, user `root`) and, on first
run, **creates the tables if missing**:
```python
cursor.execute("SHOW TABLES")
existing = {row[0] for row in cursor.fetchall()}
if 'posture_logs'    not in existing: cursor.execute("""CREATE TABLE posture_logs (...)""")
if 'gaze_logs'       not in existing: cursor.execute("""CREATE TABLE gaze_logs (...)""")
if 'cheating_events' not in existing: cursor.execute("""CREATE TABLE cheating_events (...)""")
```
`get_connection()` returns a live connection or `None` on failure — which is why the
writers can quietly stop instead of crashing when MySQL is down.

**Table roles:** `gaze_logs` = every front-cam frame (gaze + head); `posture_logs` =
side-cam joint coordinates; `cheating_events` = one row per confirmed incident.

---

## 16. `gaze_graph.py` — the live values chart

A read-only chart of `h_ratio`, `yaw`, `pitch` shown as **z-scores** (how many standard
deviations from *your* baseline). The dashed band is "normal"; spikes past it are drift.
```python
def on_features(self, feats):
    for i, k in enumerate(KEYS):
        v = float(feats[k])
        if self._means is not None:
            v = (v - self._means[i]) / std       # convert to z-score vs baseline
        self.values[k].append(max(-Z_LIMIT, min(Z_LIMIT, v)))
```
`set_user()` loads the baseline (preferably from the trained model's scaler, else from
the raw calibration). A timer repaints ~15×/sec — repainting on a timer, not per frame,
keeps it smooth. It's a **visual aid**, not part of the detection decision.

---

## The whole thing in one breath

> Run `main.py` → log in (`auth`/`session`) → student screen locks
> (`keyboard_lock` + fullscreen) → **calibrate**: front cam records your normal
> (`front_cam_worker` → `calibration_store` JSON) → **train**: Isolation Forest learns
> your normal (`train_cheat_model` → `.joblib`) → **detect**: front cam runs
> `cheat_detector` + eyes-off-screen rule + 2-second hold → one row to `cheating_events`
> (`cheat_logger`) → proctor refreshes `ProctorView` and sees it. The side camera and
> the graph run alongside but don't decide anything. The model is **personal** — you're
> judged against your own normal.
