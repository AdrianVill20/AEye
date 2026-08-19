from datetime import datetime
from db_config import get_connection


class Session:
    """Persistent session bound to a user. Inserts a row into sessions on
    login and updates logout_time on close. Each session ID is passed to
    loggers so evidence (gaze/posture) is tagged to the right session."""

    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role
        self.session_id = None
        self.login_time = datetime.now()
        self.logout_time = None
        self._save()

    def _save(self):
        conn = get_connection()
        if conn is None:
            return
        try:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO sessions (user_id, role, login_time) VALUES (%s, %s, %s)',
                (self.user_id, self.role, self.login_time),
            )
            conn.commit()
            self.session_id = cursor.lastrowid
            cursor.close()
        except Exception as exc:
            print(f'[Session] Save failed: {exc}')
        finally:
            conn.close()

    def close(self):
        if self.session_id is None:
            return
        self.logout_time = datetime.now()
        conn = get_connection()
        if conn is None:
            return
        try:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE sessions SET logout_time = %s WHERE id = %s',
                (self.logout_time, self.session_id),
            )
            conn.commit()
            cursor.close()
        except Exception as exc:
            print(f'[Session] Close failed: {exc}')
        finally:
            conn.close()
