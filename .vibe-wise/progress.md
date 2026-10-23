# Learning Progress

## Pending decision
- Study style changed 2026-10-19 (his choice): jump to PROCESS, less typing, warm-up each session.
- Stage (2026-10-24): FULL DATA JOURNEY practice, he asked to practice the whole system incl. teammates' input code. 9 stations on C:\AEye feature/isolation line numbers: 1 camera in, 2 face points, 3 eye+head numbers, 4 package+send (signals), 5 calibration save+train, 6 exam check, 7 side camera, 8 database, 9 proctor. Station 1 FIND asked. Hull view H4 blanks still open.
- 2026-10-22: at his request changed REAL C:\AEye (feature/isolation) ui/views.py POINTS_9/POINTS_5 to .02/.98 edges, center/middles kept at .5. His own attempt had replaced .5 with .2 (misread .5 as edge distance) and left (.98, .95). Flagged: MARGIN comment math (exact edge now 1/0.96 = 1.04), top dots overlap the top message text, old .05 runs won't average with new .02 dots (recalibrate fresh).
- Stage (2026-10-22): HULL VIEW H4 drawing placed in practice ui/hull_view.py lines 68-94 (blanks: hull, scan, off, -1). H1-H3 done (asked why h - for y but not x, re-explained with pixel direction). After H4, hull_view.py should be zero diff.
- Also open: POSTURE P1 placed in practice posture_worker.py lines 101-119 (blanks: MIN_VISIBILITY, hidden, None None), unanswered. Warm-up (iriun 3rd time, glance reset, phone missed check) unanswered. Branch question unanswered: C:\AEye now on feature/isolation (4eb6d4d, from f101892, adds train_gaze ridge model), answer key is feature/refactor.
- His hull-view label attempt in C:\AEye hull_view.py is commented out and in the wrong place (after the blue outline comment, not inside the grey-dots loop).
- Explained on request 2026-10-21: graham_scan by hand on demo 9 points (rest order, 2 pops, 7 corners), distance tie-breaker (test square), calibration code map with owners, teammate's _direction_ok line by line.
- MOCK PANEL round 1 done 2026-10-21, score 2.5/5.
- Round 1 results: Q1 screen area ½ (missed train/grow/save, "pop which point", .py). Q2 1-second glance ½ (missed the reset _anom_since=None, mixed screenshot vs started_at). Q3 MARGIN 1.20 full (add 1.11 math). Q4 phone false box ½ (">=" not ">", missed PHONE_END_SECONDS/phone_last_seen half). Q5 isolation forest: said "skip, teammate's", coached the hand-off line.
- Bring back in round 2: Q2 reset, Q4 missed check, Q5 hand-off line, Q1 full chain.
- 7c train() done (blanks right). Real demo check: recomputed hull matches saved screen_area_demo.json exactly, 9 dots, 7 corners, test dots inside 306/361. Corrected my own false claim: validation % is only shown (views.py:609-618), never blocks. "Calibrate again" checks live in teammate's CalibrationView._finish (28 of 31 lines lildelrep) plus train's <3 points check.
- train_cheat_model.py main() (command-line printing) not done, low priority.
- 7b dot_points/average_dots done (median reason explained well). AI filled imports and train_forest (teammate's forest).
- 7a calibration_store done, zero diff (explained why several runs are kept).
- Parked at his request 2026-10-21: ProctorView (6a scaffold placed with blanks at practice views.py 1030 refresh, 1063 3000, 1067 stop, explanation given once) and database tables (schema.sql, db_config). Also loggers. He skips DB/UI-heavy OUTPUT pieces, prefers logic.
- Warm-up 2026-10-21: Q2 cross and Q3 started_at right. Q1 'iriun' in VIRTUAL_CAMERAS wrong a 2nd time ("we use the other iriun index"), re-taught via what breaks if removed. Ask again next warm-up.
- Loggers parked at his request ("skip loggers, come back later"): cheat_logger scaffold already placed with blanks at practice lines 48, 51, 53, 60 (start, lastrowid, pop, values), explanation given once. front_cam_logger and posture_logger not started. OUTPUT 1 save_screenshot done (imWrite capital fixed), front_cam_worker.py zero diff. PROCESS complete. Safe point d6ac25e.
- OUTPUT plan (group 1, must know): save_screenshot, cheat_logger, front_cam_logger, posture_logger, schema.sql, db_config, ProctorView, calibration_store + train. Group 2 (UI, explain-only offered): gaze_graph, hull_view, DetectionView layout, ScreenBorder, web_tab. Asked re-defense date, no answer yet.
- Posture setup TODOs at practice 196/208 left for parked INPUT 8.
- PROCESS 4 (gaze 2-second rule, front_cam_worker) done, zero diff except save_screenshot (OUTPUT). started_at reason needed a retry (2 s is only the wait to be sure).
- 2026-10-21: fixed broken docstrings in practice front_cam_worker.py (save_screenshot) and views.py (DetectionView) caused by TODO markers. His editor showed stale buffer, told him to reopen without saving. Safe point commit 3078072.
- Isolation forest (PROCESS 3) skipped: he says it is a teammate's. Git: lildelrep first added it 2026-09-04, he removed it 2026-09-16 (01a4f83) and re-added it the same day (01a0a4f "added back isolation forest to pair with graham scan"), so blame shows him. AI filled FEATURES in practice (his v_opennes typo fixed). Skip train_forest and is_anomaly when reached, keep the pairing defense line.
- Done: PROCESS 1 Graham scan, PROCESS 2 outside() zero diff. Mistakes fixed: sorted vs min, MARGIN vs margin parameter. FIND answers keep missing folders (workers/, cheat/).
- Warm-up redo next session: why 'iriun' is in VIRTUAL_CAMERAS (he confused it with the [::-1] flip). Path habit: forgot cheat/ folder.
- Parked for later: INPUT 6b (practice file line 444, real views.py:484-488), 7 calibration_store, 8 posture constants. DotWindow 5a-5g done, zero diff.

## Answer key moved to refactor (2026-10-18)
- He chose to practice on C:\AEye feature/refactor 78674d8. Old practice work saved on branch practice-old (9becc80).
- New practice branch from answer, commit fea6312 "practice start (refactor)", 91 TODO markers, ownership by git blame -w -M -C. Finished pieces carried over.
- Open question to him: meaning of the "# a" lines he added

## Camera auto-pick (integration/app/ui/phone_camera.py:40-63)
- Introduced: tuple constant, list comprehension, enumerate, any, [::-1], one-line if, default parameter None
- Demonstrated understanding: why lowercase words, why keep index not name (OpenCV opens by number), why phones list is flipped (Iriun spare #2 slot), fallbacks 0 and 1, populate_camera_combo uses [0] for face cam
- Needs reinforcement: is None vs empty list needed 2 tries, "why names=None helps testing" not finished

## Calibration constants (integration/app/ui/views.py:101-113)
- Introduced: yaw and pitch, alpha darkness, smoothing step
- Demonstrated understanding: .05/.95 edge inset, VAL_POINTS kept out of training, AWAY_DEG 25 and CLOSED 0.12 as gates, GAZE_SMOOTH math
- Needs reinforcement: READY_SEC is unused (leftover), AWAY_DEG comment says "from the start" but code compares to 0

## person_path (integration/app/ui/views.py:143-151)
- Introduced: QPainterPath, addEllipse, addRoundedRect, united
- Demonstrated understanding: fractions to camera pixels, radius uses h so head is same share of frame height (after retry)
- Needs reinforcement: neck overlap 0.9 and hidden bottom corners (+ r), quiz skipped

## Graham scan (integration/app/cheat/cheat_detector.py:28-55)
- Introduced: screen area as a fence around calibration eye points (h_ratio, v_openness), cross() sign = turn direction
- Demonstrated understanding: explained cross positive = left, negative = right = dent, dent point removed (said "b" is removed, corrected to the middle point a)

## DotWindow.paintEvent (integration/app/ui/views.py:241-262)
- Introduced: paintEvent called by Qt, QPainter, KeepAspectRatio, QTransform order (last step applies first)
- Demonstrated understanding: black fill so dot is brightest, in_place 0.5 s check (stale True), frame None guard, stretched picture makes outline taller than face
- Needs reinforcement: transform quiz skipped
- 5d-5f (dark layer, green/red outline, paused message): demonstrated subtract shape, NoBrush, DIM 255 effect. if/elif stops at first match needed a retry. Path typo "intergration/app/views.py" corrected.
