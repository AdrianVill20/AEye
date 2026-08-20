import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'ninja123',
    'database': 'aeye_db',
}

def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as exc:
        print(f'[DB] Connection failed: {exc}')
        return None