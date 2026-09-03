"""Writes a cheating-detection event to MySQL.

This is the ONLY thing that touches MySQL during a live exam - it fires just
once per detected episode (not per frame), so MySQL stays idle unless the
student is actually flagged. Each row is what the proctor sees: who, and when.
"""

import queue
from PySide6.QtCore import QThread
from db_config import get_connection

INSERT_SQL = (
    "INSERT INTO cheating_events (session_user_id, detected_at) VALUES (%s, %s)"
)


class CheatEventLogger(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()
        self._running = False

    def enqueue(self, event):
        """event = {'user': str, 'timestamp': datetime}"""
        self._queue.put(event)

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
            event = self._queue.get()
            if event is None:
                break
            try:
                cursor.execute(INSERT_SQL, (event['user'], event['timestamp']))
                conn.commit()
                print(f"[CHEAT] Logged event: {event['user']} @ {event['timestamp']}")
            except Exception as exc:
                print(f'[DB] Cheat insert failed: {exc}')
                conn.rollback()
        cursor.close()
        conn.close()
