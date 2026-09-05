import queue
import time
from PySide6.QtCore import QThread
from db_config import get_connection

INSERT_SQL = (
    "INSERT INTO gaze_logs "
    "(session_user_id, captured_at, h_direction, v_direction, h_ratio, v_openness, "
    "is_blinking, yaw, pitch, roll, head_direction, landmarks_detected, signal_ok) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)

# The front cam sends a row every frame (~30/sec), so commit in batches
# instead of one round trip per row.
BATCH_SIZE = 30
FLUSH_SECONDS = 2.0

# Cap the queue so a dead database cannot slowly eat memory over a long exam.
MAX_QUEUED = 3000


class FrontCamLogWriter(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue(maxsize=MAX_QUEUED)
        self._running = False

    def enqueue(self, record):
        """record = (session_user_id, captured_at, h_direction, v_direction,
                      h_ratio, v_openness, is_blinking, yaw, pitch, roll,
                      head_direction, landmarks_detected, signal_ok)"""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass   # DB is not keeping up - drop the sample, keep the camera running

    def stop(self):
        self._running = False
        self._queue.put(None)

    def _flush(self, conn, cursor, batch):
        if not batch:
            return
        try:
            cursor.executemany(INSERT_SQL, batch)
            conn.commit()
        except Exception as exc:
            print(f'[DB] Insert failed: {exc}')
            conn.rollback()
        batch.clear()

    def run(self):
        self._running = True
        conn = get_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        batch = []
        last_flush = time.time()
        while self._running:
            try:
                record = self._queue.get(timeout=0.5)
            except queue.Empty:
                # Nothing arriving - write whatever is waiting rather than
                # holding it until the next frame shows up.
                self._flush(conn, cursor, batch)
                last_flush = time.time()
                continue
            if record is None:
                break
            batch.append(record)
            if len(batch) >= BATCH_SIZE or time.time() - last_flush > FLUSH_SECONDS:
                self._flush(conn, cursor, batch)
                last_flush = time.time()
        self._flush(conn, cursor, batch)   # write the last few before closing
        cursor.close()
        conn.close()
