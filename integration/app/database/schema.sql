CREATE DATABASE IF NOT EXISTS aeye_db
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE aeye_db;

-- ============================================================
-- Users & Sessions (Authentication)
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id              VARCHAR(64)  PRIMARY KEY,       -- school-issued ID
    password_hash   VARCHAR(255) NOT NULL,           -- bcrypt hash
    full_name       VARCHAR(128) NOT NULL,
    role            ENUM('student', 'proctor') NOT NULL DEFAULT 'student',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sessions (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(64)  NOT NULL,
    role            ENUM('student', 'proctor') NOT NULL,
    login_time      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    logout_time     DATETIME(3)  NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================================
-- Monitoring Logs
-- ============================================================

CREATE TABLE IF NOT EXISTS gaze_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id        INT UNSIGNED   NOT NULL,
    session_user_id   VARCHAR(64)    NOT NULL,
    captured_at       DATETIME(3)    NOT NULL,
    direction         VARCHAR(32)    NOT NULL,
    gaze_ratio        FLOAT          NULL,
    is_blinking       TINYINT(1)     NOT NULL DEFAULT 0,
    created_at        TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_captured_at (captured_at),
    INDEX idx_session_id (session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (session_user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS posture_logs (
    id                BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id        INT UNSIGNED   NOT NULL,
    session_user_id   VARCHAR(64)    NOT NULL,
    captured_at       DATETIME(3)    NOT NULL,
    left_shoulder_x   FLOAT          NOT NULL,
    left_shoulder_y   FLOAT          NOT NULL,
    right_shoulder_x  FLOAT          NOT NULL,
    right_shoulder_y  FLOAT          NOT NULL,
    left_wrist_x      FLOAT          NOT NULL,
    left_wrist_y      FLOAT          NOT NULL,
    right_wrist_x     FLOAT          NOT NULL,
    right_wrist_y     FLOAT          NOT NULL,
    created_at        TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_captured_at (captured_at),
    INDEX idx_session_id (session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (session_user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Default proctor account (password: admin)
-- Password hash generated with: bcrypt.hashpw(b'admin', bcrypt.gensalt())
INSERT IGNORE INTO users (id, password_hash, full_name, role)
VALUES ('admin', '$2b$12$uyQzolmTGfDWIMa0fvrnLu4yoORc2.qu2ji.uQeOBPQjyBCSyLxI.', 'Administrator', 'proctor');
