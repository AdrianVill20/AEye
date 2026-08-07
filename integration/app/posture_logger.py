import queue
from PySide6.QtCore import QThread
from db_config import get_connection

INSERT_SQL = (
    "INSERT INTO posture_logs "
    "(session_user_id, captured_at, "
    "left_shoulder_x, left_shoulder_y, "
    "right_shoulder_x, right_shoulder_y, "
    "left_wrist_x, left_wrist_y, "
    "right_wrist_x, right_wrist_y) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


class PostureLogWriter(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._running = False

    def enqueue(self, record):
        """record = (session_user_id, captured_at, l_sh_x, l_sh_y,
                      r_sh_x, r_sh_y, l_wr_x, l_wr_y, r_wr_x, r_wr_y)"""
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
                cursor.execute(INSERT_SQL, record)
                conn.commit()
            except Exception as exc:
                print(f'[DB] Insert failed: {exc}')
                conn.rollback()
        cursor.close()
        conn.close()