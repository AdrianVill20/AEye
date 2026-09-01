import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'ninja123',
    'database': 'aeye_db',
}


def ensure_database():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            connect_timeout=5,
        )
        cursor = conn.cursor()
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS aeye_db "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute("USE aeye_db")

        cursor.execute("SHOW TABLES")
        existing = {row[0] for row in cursor.fetchall()}

        if 'posture_logs' not in existing:
            cursor.execute("""
                CREATE TABLE posture_logs (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    session_user_id VARCHAR(64) NOT NULL,
                    captured_at DATETIME(3) NOT NULL,
                    left_shoulder_x FLOAT NULL,
                    left_shoulder_y FLOAT NULL,
                    left_shoulder_vis FLOAT NULL,
                    right_shoulder_x FLOAT NULL,
                    right_shoulder_y FLOAT NULL,
                    right_shoulder_vis FLOAT NULL,
                    left_wrist_x FLOAT NULL,
                    left_wrist_y FLOAT NULL,
                    left_wrist_vis FLOAT NULL,
                    right_wrist_x FLOAT NULL,
                    right_wrist_y FLOAT NULL,
                    right_wrist_vis FLOAT NULL,
                    visible_landmarks TINYINT UNSIGNED NULL,
                    signal_ok TINYINT(1) NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_time (session_user_id, captured_at)
                ) ENGINE=InnoDB
            """)

        if 'gaze_logs' not in existing:
            cursor.execute("""
                CREATE TABLE gaze_logs (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    session_user_id VARCHAR(64) NULL,
                    captured_at DATETIME(3) NOT NULL,
                    h_direction ENUM('left', 'center', 'right') NULL,
                    v_direction ENUM('up', 'center', 'down', 'calibrating') NULL,
                    h_ratio FLOAT NULL,
                    v_openness FLOAT NULL,
                    is_blinking TINYINT(1) NULL,
                    yaw FLOAT NULL,
                    pitch FLOAT NULL,
                    roll FLOAT NULL,
                    head_direction VARCHAR(32) NULL,
                    landmarks_detected SMALLINT UNSIGNED NULL,
                    signal_ok TINYINT(1) NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_time (session_user_id, captured_at)
                ) ENGINE=InnoDB
            """)

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as exc:
        print(f'[DB] Schema bootstrap failed: {exc}')
        return False


def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as exc:
        print(f'[DB] Connection failed: {exc}')
        return None