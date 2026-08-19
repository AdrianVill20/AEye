import mysql.connector
from mysql.connector import Error
from pathlib import Path

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'aeye_db',
}

SCHEMA_PATH = Path(__file__).resolve().parent / 'database' / 'schema.sql'

_schema_initialized = False


def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as exc:
        print(f'[DB] Connection failed: {exc}')
        return None


def init_schema():
    """Run schema.sql to create all tables on first launch."""
    global _schema_initialized
    if _schema_initialized:
        return True

    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
        )
        cursor = conn.cursor()

        sql = SCHEMA_PATH.read_text(encoding='utf-8')
        for statement in sql.split(';'):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)

        conn.commit()
        cursor.close()
        conn.close()
        _schema_initialized = True
        print('[DB] Schema initialized successfully.')
        return True
    except Error as exc:
        print(f'[DB] Schema init failed: {exc}')
        return False
