# Head Pose, Explained Simply 🧠📷

> ## ⚠️ The file this describes was deleted
>
> `headpose_worker.py` is gone. The yaw / pitch / roll maths below is unchanged
> though — it now lives inside `front_cam_worker.py` (search for
> `facial_transformation_matrixes`), alongside the eye-gaze code. Everything
> this document explains is still correct; only the filename changed.

> A super-simple walkthrough of [`headpose_worker.py`](headpose_worker.py).
> Goal: after reading this you can explain, out loud, *exactly* what this file does —
> no scary words left behind.

---

## 1. What is "head pose"?

**Head pose = which way your head is pointing.** That's it.

Imagine your head is a little toy on a stick. It can move in **3 ways**:

| Name | Kid version | Real-life move |
|------|-------------|----------------|
| **Yaw** | Shaking your head "**no**" | turning left ↔ right |
| **Pitch** | Nodding "**yes**" | looking up ↕ down |
| **Roll** | Tilting your ear toward your shoulder | leaning your head sideways ↻ |

This file watches your face through the camera and figures out those 3 numbers
(**yaw, pitch, roll**) every moment. Then it decides a simple sentence like
*"looking left"* or *"looking down"*.

---

## 2. The big picture (one breath)

```
Camera  →  find the face  →  measure the 3 head angles  →  write them on screen
```

That's the whole job. Everything below is just *how* each arrow works.

---

## 3. The helper that finds your face

Computers can't "see" a face on their own. So we give them a **cheat sheet** — a file
that was trained on millions of faces and already knows how to spot one.

- [headpose_worker.py:10](headpose_worker.py):
  ```python
  MODEL = ... / 'head_pose' / 'face_landmarker.task'
  ```
  This points to that cheat sheet: **`../head_pose/face_landmarker.task`**.
  It's Google's **MediaPipe FaceLandmarker** — it finds **478 tiny dots** all over your face
  (eyes, nose, jaw, lips…). Those dots are called **landmarks**.

Think of it like a **connect-the-dots** puzzle that the computer solves instantly on your face.

---

## 4. What a "worker" is (and why it's on its own thread)

The screen (the window you click) has one job: **stay smooth and not freeze.**

Reading a camera and doing face-math is slow-ish. If the window did that itself, it would
**freeze** like a laggy video game. So we hand that heavy job to a **worker** who does it
**in the background** and just **shouts the results back**.

- [headpose_worker.py:12](headpose_worker.py): `class HeadPoseWorker(QThread)`
  → "QThread" = *a helper that runs in the background.*
- The worker can **shout two things** (these are called **signals**):
  - [headpose_worker.py:13](headpose_worker.py) `frame_ready` → "here's a **picture** to show!"
  - [headpose_worker.py:14](headpose_worker.py) `stats_ready` → "here are the **numbers**!"

The window listens for those shouts and updates itself. 🗣️→🖼️

---

## 5. Getting ready — `run()` starts up

`run()` is the background job. Before the loop, it sets up two things:

**a) Turn on the face-finder** ([headpose_worker.py:26-27](headpose_worker.py)):
```python
options = vision.FaceLandmarkerOptions(
    base_options=... model_asset_path=str(MODEL),   # load the cheat sheet
    running_mode=vision.RunningMode.VIDEO,           # we're watching a video, not one photo
    output_facial_transformation_matrixes=True)      # ALSO tell me which way the face is turned
landmarker = vision.FaceLandmarker.create_from_options(options)
```
That last line, `output_facial_transformation_matrixes=True`, is the magic switch that makes
it give us the **head direction**, not just the dots.

**b) Open the camera** ([headpose_worker.py:28](headpose_worker.py)):
```python
cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)   # camera 0 = your webcam
```
If the camera won't open, it politely says so and stops
([headpose_worker.py:29-31](headpose_worker.py)) — no crash.

---

## 6. The loop — the part that repeats forever ♻️

`while self._running:` ([headpose_worker.py:33](headpose_worker.py)) means
"**keep doing this over and over** until someone says stop." Each lap does the same steps:

### Step 1 — Grab one picture
```python
ok, frame = cap.read()                 # take a snapshot from the camera
frame = cv2.flip(frame, 1)             # flip it like a MIRROR so it feels natural
```
([headpose_worker.py:34-37](headpose_worker.py)) The mirror flip is why, when you move
right, the screen-you also moves right — like looking in an actual mirror.

### Step 2 — Hand the picture to the face-finder
```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # fix the colors to the format it wants
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = landmarker.detect_for_video(mp_image, timestamp_ms)
timestamp_ms += 33
```
([headpose_worker.py:38-41](headpose_worker.py))

- **Colors:** the camera gives colors in a weird order (BGR); we flip them to normal (RGB).
- **`timestamp_ms += 33`:** the video runs like a **flipbook** — about **30 pictures per second**,
  and `1 second ÷ 30 ≈ 33 milliseconds` per picture. We tell the finder "this is the picture at
  time 33ms… 66ms… 99ms…" so it knows the order.

### Step 3 — Did we find a face?
```python
if result.face_landmarks and result.facial_transformation_matrixes:
```
([headpose_worker.py:43](headpose_worker.py)) Only do the math **if a face is actually there.**
No face → we skip and just show the plain picture.

### Step 4 — Turn "face direction" into 3 angle numbers 🎯
This is the only mathy bit. Stay with me — the idea is simple.

The face-finder gives us a little **direction chart** for the face (called a *rotation matrix*, `R`).
It's just numbers describing which way the face is turned in 3D space.

```python
R = np.array(result.facial_transformation_matrixes[0])[:3, :3]   # grab the direction chart
yaw   = math.degrees(math.atan2(-R[2, 0], sy))
pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
```
([headpose_worker.py:45-50](headpose_worker.py))

- `atan2(...)` is just a **math tool that reads the direction chart and spits out an angle.**
  You don't need to memorize it — think of it as a **protractor** 📐 that measures the turn.
- `math.degrees(...)` turns the answer into normal **degrees** (like 0°, 45°, 90°) instead of
  math-radians.
- Result: three numbers — **yaw, pitch, roll** — the shake, the nod, and the tilt from Section 1.

### Step 5 — Turn numbers into a plain-English sentence
Numbers are precise but boring. So we make a friendly label:
```python
TH = 10                                   # ignore tiny wobbles under 10 degrees
vert  = 'down' if pitch > TH else 'up'   if pitch < -TH else ''
horiz = 'right' if yaw  > TH else 'left' if yaw  < -TH else ''
direction = 'looking ' + (words) or 'looking center'
```
([headpose_worker.py:51-56](headpose_worker.py))

- **`TH = 10` (a threshold):** a "**don't be too picky**" rule. If your head moves less than 10°,
  we call it *center* — because everybody's head jiggles a little, and that's not cheating.
- Combine the up/down word + the left/right word → e.g. `"looking down right"`, or `"looking center"`
  if the head is basically straight.

### Step 6 — Draw the words on the picture
```python
cv2.putText(frame, direction, ...)                       # e.g. "looking left"
cv2.putText(frame, f'Y {yaw:+.0f}  P {pitch:+.0f}  R {roll:+.0f}', ...)  # the raw angles
```
([headpose_worker.py:57-58](headpose_worker.py)) This paints the sentence and the 3 numbers
**right onto the video** so you can see them.

### Step 7 — Shout the results back to the window
```python
qimg = QImage(...)                 # package the picture in a form the window understands
self.frame_ready.emit(qimg)        # "here's the picture!"
self.stats_ready.emit(stats)       # "here are the numbers!"
```
([headpose_worker.py:59-63](headpose_worker.py)) The window catches these and updates the screen.
Then the loop starts over with a fresh picture. ♻️

---

## 7. Stopping cleanly 🛑
- `stop()` ([headpose_worker.py:21](headpose_worker.py)) sets `self._running = False`.
- Next time the loop checks `while self._running:`, it's `False`, so the loop ends.
- `cap.release()` ([headpose_worker.py:64](headpose_worker.py)) **lets go of the camera** so other
  tabs can use it. (Very important — see the camera-sharing note in the front-end notes.)

---

## 8. What lands on screen (the stats)
The `stats` dictionary it shouts contains:

| Stat | Means |
|------|-------|
| **Direction** | the friendly sentence, e.g. *"looking down"* |
| **Yaw** | shake-your-head angle (left/right), in degrees |
| **Pitch** | nod angle (up/down), in degrees |
| **Roll** | tilt angle (ear-to-shoulder), in degrees |
| **Landmarks detected (/478)** | how many face-dots it found (should be 478 when it sees you) |

---

## 9. The whole thing in 6 kid-sentences
1. A background helper opens the webcam. 📷
2. It uses a trained "face cheat sheet" to find 478 dots on your face. 🔵
3. It also asks *which way the face is turned*. ↻
4. It turns that into 3 angles: shake (**yaw**), nod (**pitch**), tilt (**roll**). 🎯
5. If the turn is bigger than 10°, it says *left/right/up/down*; otherwise *center*. 📐
6. It draws that on the video and shouts the picture + numbers to the screen — 30 times a second. 🗣️

---

## 10. Words you might forget (mini-dictionary)
- **Landmark** = a tiny dot the AI places on a face feature (eye corner, nose tip…).
- **Model / `.task` file** = the pre-trained "brain" that knows how to find faces.
- **QThread / worker** = a background helper so the window doesn't freeze.
- **Signal (`emit`)** = the worker "shouting" data to the window.
- **Frame** = one single picture from the camera.
- **BGR / RGB** = two different orders of Red-Green-Blue; we convert between them.
- **Rotation matrix** = a grid of numbers describing which way something is turned.
- **`atan2` / degrees** = math that reads that grid and gives back a normal angle.
- **Threshold (`TH`)** = a "don't be too picky" cutoff so small wobbles are ignored.
