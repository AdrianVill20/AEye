# AEye

**Multimodal Analysis of Visual Behavior for Cheating Detection in Computer-Based Exams**

AEye is a standalone Windows desktop application for proctoring computer-based examinations. It works alongside a school's e-class platform, reads a student's behavior from two cameras, judges that behavior against the student's own personal baseline rather than a generic standard, and delivers real-time confidence scores to a human proctor who makes the final decision.

The project is built to be **more accurate and fairer** than typical single-signal, single-camera proctoring tools — especially in the uneven lighting and varied equipment conditions common in Philippine schools.

---

## Why AEye

Most camera-based proctoring tools watch a single signal (usually eye gaze) through a single front-facing camera, and treat almost any look away from the screen as suspicious. This causes two recurring problems:

- **False positives on normal behavior.** Looking away to think, glancing down to write on scratch paper, or reading a long question all get flagged as cheating.
- **Unfairness to students who move differently.** Students with tremors, fidgeting, or involuntary movements, and those with eyeglasses, cross-eyed, or squint-eyed conditions, are repeatedly red-flagged for features and behaviors they cannot change.

A front-only view also can't see the desk or the student's hands, so it can't tell whether a downward glance is legitimate writing or reaching for a hidden device.

AEye addresses these gaps directly.

---

## Key Features

- **Two-mode desktop application** — a *student mode* for taking exams and a *proctor mode* for monitoring, backed by a central server.
- **Controlled exam window** — displays the school's e-class exam page inside a locked environment, running fullscreen and limiting application switching (functioning like a lockdown browser).
- **Own login** — students sign in with their school-issued student ID; identity comes from the login rather than biometric face recognition.
- **Multi-signal detection** — combines eye gaze, head pose, and facial expression instead of relying on eye gaze alone.
- **Dual-camera setup** — a front-facing line-of-sight camera captures the face, while an elevated side camera captures upper-body posture and hand movement toward or under the desk.
- **30-second calibration per exam** — records two areas of interest (Zone 1: the screen, Zone 2: the desk) and checks the environment (camera quality, lighting) and the student's facial state (eyeglasses, cross-eyed/squint-eyed conditions).
- **Per-student behavioral baseline** — learns each student's normal behavior across exams so involuntary movements are treated as normal for that student, not as cheating.
- **Temporal pattern detection** — analyzes behavior over time (e.g., the look-away → read → return cycle) rather than judging a single frozen frame.
- **Real-time proctor alerts** — flags suspicious moments live, with a confidence score, severity level, and time-stamped screenshots tied to the student's ID. The human proctor always makes the final decision.

---

## How It Works

AEye follows an **Input → Process → Output** flow.

### Input
- Student ID (from AEye's own login)
- Front-facing camera feed (face, eye gaze, head pose, facial landmarks)
- Elevated side camera feed (upper-body posture, arm and hand movement)
- Calibration data: areas of interest, environment, and lighting

### Process
1. **Calibration** — MediaPipe records the student's normal face state, the two areas of interest, and the environment.
2. **Signal reading** — MediaPipe reads eye gaze, head pose, and facial expression from the front camera; the side camera provides posture and hand movement, checked against the areas of interest.
3. **Behavior learning** — current signals are compared to the student's saved behavioral profile so personal habits and legitimate actions (like writing within the desk zone) are treated as normal.
4. **Pattern detection & scoring** — repeating patterns over short time windows feed a sequence-based model (with a simpler time-window rule as a fallback), producing a confidence score with a severity level.

### Output
Flagged moments — each with a confidence score, severity level, triggering pattern, and time-stamped screenshot — are sent through the central server to proctor mode. The student's behavioral profile is updated after each exam.

### Feedback Loop
Every completed exam refines the student's behavioral profile, so future sessions have a better sense of what is normal for that student, reducing false alarms over time.

---

## Tech Stack

- **Platform:** Windows desktop application + central server
- **Computer vision:** [MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) (478-point face landmarks, expression scores, head pose)
- **Temporal modeling:** Sequence-based model (LSTM-style) with a rule-based time-window fallback
- **Hardware:** Two ordinary cameras (e.g., a laptop webcam + a phone camera)

---

## Scope & Limitations

AEye is intentionally focused. It is important to understand what it does **not** do:

- **No final verdict on cheating.** It only flags moments and provides scores and evidence — a human proctor decides.
- **No face recognition or biometric authentication.** Identity is trusted from the login.
- **Requires two working cameras.** If one camera is covered, broken, or disconnected, its signals are unavailable. Very dark rooms may reduce accuracy even after calibration.
- **Only detects visible behavior.** It cannot catch memorized notes, silent earpieces, or collusion outside camera view.
- **Windows only.** macOS, Linux, ChromeOS, and mobile platforms are out of scope.
- **No session-level e-class integration.** AEye displays the e-class exam page in its controlled window but does not exchange login sessions; a proctor requires students to use AEye, as with a lockdown browser.
- **Not clinically tested with medical patient groups.** Fairness is demonstrated through accuracy and false-positive measurements rather than clinical validation.

---

## Project Team

**Authors**
- Christian V. Demetillo
- Adrian T. Villarte
- Allain James O. Ybañez

**Adviser**
- Engr. Carmel Tejana

A thesis project presented to the Faculty of the School of Computer Studies, in fulfillment of the requirements for the degree of Bachelor of Science in Computer Science.

---

## References

Key works informing AEye's design:

- Atoum et al. (2017) — *Automated Online Exam Proctoring*, IEEE Transactions on Multimedia
- Dilini et al. (2021) — Eye-gaze cheating detection in browser-based exams
- Al-Mukhtar et al. (2024) — Deep learning-based multimodal cheating detection
- Naveen et al. (2025) — *AutoOEP*, a multimodal dual-camera framework
- Khalil et al. (2025) — Review of deep learning models for exam cheating detection
- Njeru et al. (2025) — Surveillance and disability in online proctored exams
- Coghlan et al. (2021) — Ethics of online exam supervision technologies
- Ansari et al. (2023) — Person-specific gaze estimation from low-quality webcam images

*A full bibliography is available in the thesis manuscript.*
