CREATE DATABASE IF NOT EXISTS aeye_db
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE aeye_db;

CREATE TABLE IF NOT EXISTS posture_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id   VARCHAR(64)  NULL,
    captured_at       DATETIME(3)  NOT NULL,
    left_shoulder_x   FLOAT        NOT NULL,
    left_shoulder_y   FLOAT        NOT NULL,
    right_shoulder_x  FLOAT        NOT NULL,
    right_shoulder_y  FLOAT        NOT NULL,
    left_wrist_x      FLOAT        NOT NULL,
    left_wrist_y      FLOAT        NOT NULL,
    right_wrist_x     FLOAT        NOT NULL,
    right_wrist_y     FLOAT        NOT NULL,
    created_at        TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_captured_at (captured_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS gaze_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_user_id   VARCHAR(64)   NULL,
    captured_at       DATETIME(3)   NOT NULL,
    direction         ENUM('left', 'center', 'right') NOT NULL,
    gaze_ratio        FLOAT         NULL,
    is_blinking       TINYINT(1)    NOT NULL DEFAULT 0,
    created_at        TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_captured_at (captured_at)
) ENGINE=InnoDB;