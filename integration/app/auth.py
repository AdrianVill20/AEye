import bcrypt
from db_config import get_connection


def authenticate(user_id: str, password: str, role: str) -> bool:
    """Verify credentials against the users table."""
    user_id = user_id.strip()
    password = password.strip()
    role = role.strip().lower()

    if not user_id or not password:
        return False

    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            'SELECT id, password_hash, role FROM users WHERE id = %s',
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()

        if row is None:
            return False

        stored_hash = row['password_hash'].encode('utf-8')
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return False

        if row['role'] != role:
            return False

        return True
    finally:
        conn.close()


def create_user(user_id: str, password: str, full_name: str, role: str) -> str:
    """Insert a new user. Returns '' on success, error message on failure."""
    user_id = user_id.strip()
    password = password.strip()
    full_name = full_name.strip()
    role = role.strip().lower()

    if not user_id or not password or not full_name:
        return 'All fields are required.'

    conn = get_connection()
    if conn is None:
        return 'Database connection failed.'

    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE id = %s', (user_id,))
        if cursor.fetchone():
            cursor.close()
            return 'User ID already exists.'

        pw_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute(
            'INSERT INTO users (id, password_hash, full_name, role) VALUES (%s, %s, %s, %s)',
            (user_id, pw_hash, full_name, role),
        )
        conn.commit()
        cursor.close()
        return ''
    except Exception as exc:
        return str(exc)
    finally:
        conn.close()


def get_user(user_id: str) -> dict | None:
    """Return user record or None."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id, full_name, role FROM users WHERE id = %s', (user_id.strip(),))
        row = cursor.fetchone()
        cursor.close()
        return row
    finally:
        conn.close()


def get_all_sessions() -> list[dict]:
    """Return active sessions (no logout_time) with user info."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT s.id, s.user_id, u.full_name, s.role, s.login_time
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.logout_time IS NULL
            ORDER BY s.login_time DESC
        ''')
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        conn.close()
