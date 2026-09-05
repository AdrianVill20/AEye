-- AEye database schema
--
-- REFERENCE ONLY. The app does not run this file. db_config.ensure_database()
-- creates these same tables on startup and is the authoritative version -
-- if you change a column here, change it there too.
--
-- WARNING: this file is destructive. Each table is dropped before it is created,
-- so running this wipes everything that was logged before.
--
-- Run it by pasting into MySQL Workbench / the mysql client.

CREATE DATABASE IF NOT EXISTS aeye_db
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE aeye_db;


-- ---------------------------------------------------------------------------
-- Side camera: upper-body posture and hand position
-- ---------------------------------------------------------------------------
--
-- Written by PostureLogWriter, ~2 rows/sec, while DetectionView is tracking.
--
-- The x/y columns are NULL-able on purpose. MediaPipe always returns all 33
-- landmarks, and the hidden ones are interpolated guesses that look exactly like
-- real readings, so a joint below the visibility cutoff is stored as NULL rather
-- than as a number nobody can tell apart from a real one later.
--
-- The _vis columns keep the raw visibility score either way. That is what makes
-- it possible to check the cutoff against real side-camera recordings instead of
-- guessing at it.

DROP TABLE IF EXISTS posture_logs;

CREATE TABLE posture_logs (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id     VARCHAR(64)  NOT NULL,
    captured_at         DATETIME(3)  NOT NULL,

    left_shoulder_x     FLOAT        NULL,
    left_shoulder_y     FLOAT        NULL,
    left_shoulder_vis   FLOAT        NULL,
    right_shoulder_x    FLOAT        NULL,
    right_shoulder_y    FLOAT        NULL,
    right_shoulder_vis  FLOAT        NULL,
    left_wrist_x        FLOAT        NULL,
    left_wrist_y        FLOAT        NULL,
    left_wrist_vis      FLOAT        NULL,
    right_wrist_x       FLOAT        NULL,
    right_wrist_y       FLOAT        NULL,
    right_wrist_vis     FLOAT        NULL,

    -- How many of the 33 landmarks were really visible, and whether the camera
    -- had a usable view at all. Together these say whether a gap in the data was
    -- actual behaviour or just a bad view (dark room, occluded arm).
    visible_landmarks   TINYINT UNSIGNED NULL,
    signal_ok           TINYINT(1)   NOT NULL DEFAULT 0,

    created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_time (session_user_id, captured_at)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- Front camera: eye gaze + head pose, combined per frame (FrontCamWorker)
-- ---------------------------------------------------------------------------
--
-- Written by FrontCamLogWriter, one row per frame (~30/sec, batched), while
-- DetectionView is tracking.

DROP TABLE IF EXISTS gaze_logs;

CREATE TABLE gaze_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id   VARCHAR(64)   NULL,
    captured_at       DATETIME(3)   NOT NULL,

    -- eye gaze
    h_direction       ENUM('left', 'center', 'right') NULL,
    v_direction       ENUM('up', 'center', 'down', 'calibrating') NULL,
    h_ratio           FLOAT         NULL,   -- self._prev_h
    v_openness        FLOAT         NULL,   -- avg_open
    is_blinking       TINYINT(1)    NULL,   -- not populated yet, no blink detection in worker

    -- head pose
    yaw               FLOAT         NULL,
    pitch             FLOAT         NULL,
    roll              FLOAT         NULL,
    head_direction    VARCHAR(32)   NULL,   -- e.g. 'looking down left', 'looking center'

    landmarks_detected SMALLINT UNSIGNED NULL,   -- out of 478
    signal_ok         TINYINT(1)    NOT NULL DEFAULT 0,

    created_at        TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_time (session_user_id, captured_at)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- Confirmed cheating episodes (what the proctor sees)
-- ---------------------------------------------------------------------------
--
-- Written by CheatEventLogger, one row per episode (not per frame) once the
-- model + rule + 2-second hold all agree. ProctorView reads this table.

DROP TABLE IF EXISTS cheating_events;

CREATE TABLE cheating_events (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id   VARCHAR(64)   NOT NULL,
    detected_at       DATETIME(3)   NOT NULL,

    created_at        TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_time (session_user_id, detected_at)
) ENGINE=InnoDB;
