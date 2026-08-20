import queue
from PySide6.QtCore import QThread
from db_config import get_connection

HEAD_POSE_INSERT_SQL = (
    "INSERT INTO gaze_logs "
    "(session_user_id, captured_at, source, yaw, pitch, roll, "
    "head_h_direction, head_v_direction, landmarks_detected, signal_ok) "
    "VALUES (%s, %s, 'head_pose', %s, %s, %s, %s, %s, %s, %s)"
)


class HeadPoseLogWriter(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._running = False

    def enqueue(self, record):
        """record = (session_user_id, captured_at, yaw, pitch, roll,
                      head_h_direction, head_v_direction,
                      landmarks_detected, signal_ok)"""
        self._queue.put(record)

    def stop(self):
        self._running = False
        self._queue.put(None)

    def run(self):
        self._running = True
        conn = get_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        while self._running:
            record = self._queue.get()
            if record is None:
                break
            try:
                cursor.execute(HEAD_POSE_INSERT_SQL, record)
                conn.commit()
            except Exception as exc:
                print(f'[DB] Insert failed: {exc}')
                conn.rollback()
        cursor.close()
        conn.close()