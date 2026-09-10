# AEye — Study Guide

A simple walk-through of how the app works and what each file does.

---

## What is AEye?

AEye is a **lockdown exam app**. A student logs in, the app takes over the
screen (fullscreen + blocks Alt+Tab), and two cameras watch them:

- **Front camera** — where the eyes and head are pointing
- **Side camera** — posture (shoulders, wrists)

Cheating is **not** judged by one fixed rule for everyone. Each student first
**calibrates** (reads a passage on camera) so the app learns what *their*
normal looks like. That sample trains a **personal model**. During the exam
the front camera runs that model live, and only a confirmed incident is saved
to the database, where the **proctor** can see it.

Anything slow (cameras, AI, training, database) runs on its **own background
thread**, so the window never freezes.

---

## The big picture

```
LOGIN
  |
  |-- student --> LOCKED DASHBOARD --> Calibrate --> Train --> Detect (live)
  |                                                              |
  |                                                     writes cheating to DB
  |
  '-- proctor --> ALERTS TABLE (reads cheating from DB)
```

---

## The 3 main steps (the heart of the app)

**1. Calibrate** → front camera records your *normal* behaviour
  → saved as `calibration_<user>.json`

**2. Train** → an Isolation Forest learns your normal from that JSON
  → saved as `cheat_model_<user>.joblib`

**3. Detect** → front camera runs your model on every frame
  → a confirmed cheat is saved to MySQL

The two files in the middle (the `.json` and the `.joblib`) are real files
on disk you can open and look at.

---

## When does it actually flag a cheat?

A frame is only flagged when **all three** are true:

1. **Model gate** — the model says this is *unusual for you*
2. **Rule gate** — your eyes are really off-screen (looking down at a desk,
   or gaze to a side). A simple head turn is ignored.
3. **Time gate** — it stays that way for **2 seconds** (a quick glance
   doesn't count)

When all three pass, it fires **once** per incident (not once per frame), and
that one incident becomes one row in the database.

---

## The files, grouped by job

### Startup
| File | What it does |
|------|--------------|
| `main.py` | The entry point. Builds the 3 screens, decides student vs proctor, cleans up on exit. |
| `db_config.py` | Connects to MySQL and creates the database + tables if they don't exist. |

### Who is logged in
| File | What it does |
|------|--------------|
| `auth.py` | Login check. Right now it only checks both fields are filled (placeholder). |
| `session.py` | Remembers the current `user_id` + `role`. |

### The screens
| File | What it does |
|------|--------------|
| `views.py` | **All the windows in one file:** the login form, the student dashboard, the calibration screen, the reading page, the live tracking screen, and the proctor's alert table. |

### The cameras (background threads)
| File | What it does |
|------|--------------|
| `front_cam_worker.py` | ⭐ The most important file. Reads gaze + head pose, and runs the 3-gate cheat detection. |
| `posture_worker.py` | Reads posture from the side camera and logs it. |

### The personal model
| File | What it does |
|------|--------------|
| `calibration_store.py` | Saves/loads the calibration JSON (your "normal" sample). |
| `train_cheat_model.py` | Trains the Isolation Forest and saves the `.joblib` model. |
| `cheat_detector.py` | ⭐ Loads your model and scores a frame: *unusual or not?* If there's no model, it just never flags. |

### Saving to the database
| File | What it does |
|------|--------------|
| `front_cam_logger.py` | Writes gaze data to `gaze_logs` (fast, ~30/sec, in batches). |
| `posture_logger.py` | Writes posture data to `posture_logs` (~2/sec). |
| `cheat_logger.py` | Writes **one row per confirmed cheat** to `cheating_events`. This is what the proctor sees. |

### Lockdown & the graph
| File | What it does |
|------|--------------|
| `keyboard_lock.py` | Blocks Alt+Tab, Windows key, Alt+F4, etc. (Can't block Ctrl+Alt+Del — Windows won't allow it.) |
| `web_tab.py` | A fullscreen browser locked to one exam website only. |
| `gaze_graph.py` | The live graph showing gaze/head values vs your baseline. |

---

## How the pieces talk (during the exam)

The front camera thread **sends out signals**, and other parts listen:

```
FrontCamWorker  --frame_ready-----> Front video on screen
                --features_ready--> Live graph
                --record_ready----> gaze_logs   (database history)
                --cheat_detected--> cheating_events  (what the proctor sees)

SideCameraWorker -----------------> posture_logs (database history)
```

Only `cheat_detected` reaches the proctor. The log writers are just history
for later analysis.

---

## 5 ideas that explain the design

1. **Threads + signals** — slow work runs on background threads and reports
   back with signals, so the app never freezes.
2. **The model is personal** — it learns *your* normal, so a naturally fidgety
   student and a still student get different baselines.
3. **Three gates** — model + rule + 2 seconds, so there are fewer false alarms
   and one row per real incident.
4. **JSON vs MySQL** — calibration is a reusable JSON file; live data and
   cheats go to MySQL. Different jobs, different storage.
5. **Fail-safe** — if there's no model yet, detection just turns off and the
   cameras keep working. The exam is never blocked.

---

## The 5 features the model uses

Always in this exact order (defined once, reused everywhere):

`h_ratio` · `v_openness` · `yaw` · `pitch` · `roll`

- `h_ratio` — eyes left/right
- `v_openness` — eyes up/down (how open they are)
- `yaw` / `pitch` / `roll` — head turn / nod / tilt

---

## Quick "what happens when…"

**A student sits down:**
login → lock the screen → open exam view → calibrate → train → live tracking →
open the exam website.

**A frame becomes an alert:**
read landmarks → model says unusual → rule says eyes off-screen → held for 2s →
fire one event → save to `cheating_events` → proctor refreshes and sees it.
