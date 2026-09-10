"""Writes cheating episodes to MySQL, one row per episode.

A 'start' event inserts the row: who, when it started, why, and the
screenshot. The matching 'end' event fills in ended_at on that same row.
Until then ended_at is NULL, which the proctor sees as "ongoing".

Events come from two cameras ('gaze' = front, 'phone' = side). Each source
keeps its own open row, so a phone ending never closes a gaze episode.
"""

import queue
from datetime import datetime
from PySide6.QtCore import QThread
from db_config import get_connection

INSERT_SQL = (
    "INSERT INTO cheating_events (session_user_id, started_at, reason, screenshot_path) "
    "VALUES (%s, %s, %s, %s)"
)
END_SQL = "UPDATE cheating_events SET ended_at = %s WHERE id = %s"


class CheatEventLogger(QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = queue.Queue()

    def enqueue(self, event):
        """event = {'kind': 'start', 'source', 'user', 'started_at', 'reason', 'screenshot'}
                or {'kind': 'end', 'source', 'ended_at'}"""
        self._queue.put(event)

    def stop(self):
        self._queue.put(None)

    def run(self):
        conn = get_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        open_ids = {}   # source -> id of its row whose episode has not ended yet
        while True:
            event = self._queue.get()
            if event is None:
                break
            try:
                if event['kind'] == 'start':
                    cursor.execute(INSERT_SQL, (event['user'], event['started_at'],
                                                event['reason'], event['screenshot']))
                    open_ids[event['source']] = cursor.lastrowid
                elif event['source'] in open_ids:
                    cursor.execute(END_SQL, (event['ended_at'], open_ids.pop(event['source'])))
                conn.commit()
                print(f"[CHEAT] Logged {event['kind']} of episode")
            except Exception as exc:
                print(f'[DB] Cheat event failed: {exc}')
                conn.rollback()
        # Tracking was stopped while still flagged - close those episodes now.
        for row_id in open_ids.values():
            cursor.execute(END_SQL, (datetime.now(), row_id))
        conn.commit()
        cursor.close()
        conn.close()
