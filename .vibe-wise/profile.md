# Learner Profile

Learning mode: active
Onboarding: complete

## Project
Situation: Existing
Building: AEye thesis exam-proctoring app. Re-learning his own part of the code for the thesis re-defense.
Codebase familiarity: New (said AI wrote his part and he never checked it)
Learning scope: Parts we touch, limited to his own code by git blame, in IPO order (INPUT -> PROCESS -> OUTPUT)

## Experience
Overall programming: Beginner (self-reported "only know basic Python")
Stack familiarity: Beginner in PySide6/Qt, OpenCV, MediaPipe, MySQL (self-reported)

## Goals
Primary: Find and explain his own AEye code at the re-defense (where it is, what each line does, why each value)
Capability goal: Rebuild his code in C:\AEye_Clone branch practice until `git diff answer -- <file>` is empty

## Preferences
Checkpoint frequency: Frequent (2-6 lines per piece)
Question style: Mixed. Since 2026-10-19: 5-min warm-up (3 recall questions on old pieces) at the start of each session, then 1-2 defense quiz questions per piece
Implementation style: Less typing (since 2026-10-19). He types only the 1-2 key lines per piece, explains the rest in his own words. AI never writes it unless he gives up
Other: plain beginner words always (re-asked 2026-10-18), short sentences, few tables, explain every term. Full folder paths with line numbers, defense answers use AEye examples only (prof rule), no semicolons in text, progress list at top of each reply, short answers. Skips self-test blocks and some quizzes: give short answers and move on.

## Strong Concepts
- Reading a git diff (- missing, + extra, whitespace and duplicate lines)
- Constants and tuples, lowercase matching (VIRTUAL_CAMERAS)
- List comprehension as a filter with enumerate and any (pick_cameras)
- Screen positions as 0-1 fractions (POINTS_9, VAL_POINTS) and why validation dots stay separate
- Calibration quality gates (AWAY_DEG, CLOSED)

## Developing Concepts
- Exact paths: drops the ui/ folder, typos in file names
- Finding code by search: answered "idk" for lines and names, search trick introduced
- Answering "why" and "what breaks", not only "what it does"
- Qt drawing: QPainter, QPainterPath, QTransform, paintEvent

## Revisit
- integration/app/ui/phone_camera.py:240-246 self-test block (skipped, still a diff)
- names=None test reason (fill-in sentence skipped)
- person_path quiz Q2-Q3 and outline transform quiz (skipped, answers given)
