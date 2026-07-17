# AEye — Eye Gaze Module

Feature-based eye-gaze estimation built on MediaPipe's **FaceLandmarker (Tasks API)**.
It reads iris + head-pose features from the webcam, learns a per-user mapping to
screen position via a short calibration, and shows live gaze.

> Requires **Python 3.12**. Built and tested with `mediapipe 0.10.35` (the old
> `mp.solutions` API is gone in this version — everything here uses the Tasks API).

## Setup

```powershell
# from the repo root
cd app_video
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r gaze\requirements.txt
```

The model file `face_landmarker.task` is included in this folder. If it's missing,
download it once:

```powershell
curl.exe -L -o gaze\face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

## Run (in order)

```powershell
cd gaze
python calibrate.py     # follow the moving dot with your eyes (~40s)
python train.py         # trains the gaze model, prints honest held-out error
python live_gaze.py     # red dot follows where you look;  q = quit
```

Each tester **calibrates on their own machine** — calibration data and the trained
model are per-user and are gitignored, so run `calibrate.py` first.

## Files

| File | Role |
|------|------|
| `gaze_common.py` | Shared: landmarker setup + feature extraction (imported by the rest) |
| `calibrate.py` | Smooth-pursuit calibration → `calibration_data.json` |
| `train.py` | Trains features → screen mapping → `gaze_model.pkl` |
| `live_gaze.py` | Live gaze dot + debug readout |
| `calibrate_zones.py` / `train_zones.py` / `live_zones.py` | Experimental: classify SCREEN / DESK / AWAY instead of exact coordinates |
| `step1_*.py … step3_*.py` | Learning/reference scripts (camera → mesh → features) |

## Known limitations

- Per-session calibration; accuracy drifts if you sit very differently than when you
  calibrated. Re-calibrate if you move a lot.
- Vertical gaze is weaker than horizontal (a known webcam-gaze limitation).
- Degrades on very dark rooms / very low-resolution cameras.
