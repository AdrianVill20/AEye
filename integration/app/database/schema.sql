-- AEye database schema
--
-- WARNING: this file is destructive. Each table is dropped before it is created,
-- so running this wipes everything that was logged before.
--
-- Run it with:  python integration/app/database/init_db.py
-- (or paste it into MySQL Workbench / the mysql client)

CREATE DATABASE IF NOT EXISTS aeye_db
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE aeye_db;


-- ---------------------------------------------------------------------------
-- Side camera: upper-body posture and hand position
-- ---------------------------------------------------------------------------
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
-- Front camera: eye gaze
-- ---------------------------------------------------------------------------
--
-- Unchanged from v1. This table belongs to the eye-gaze module; nothing writes
-- to it yet, so expect it to stay empty until that side is wired up.

DROP TABLE IF EXISTS gaze_logs;

CREATE TABLE gaze_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id   VARCHAR(64)   NULL,
    captured_at       DATETIME(3)   NOT NULL,
    direction         ENUM('left', 'center', 'right') NOT NULL,
    gaze_ratio        FLOAT         NULL,
    is_blinking       TINYINT(1)    NOT NULL DEFAULT 0,
    created_at        TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_captured_at (captured_at)
) ENGINE=InnoDB;
