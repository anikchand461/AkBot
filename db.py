# database.py
import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat(user_message: str, bot_response):
    print("========== save_chat ==========")
    print("user_message:", type(user_message), user_message)
    print("bot_response:", type(bot_response), bot_response)
    print("===============================")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chats (user_message, bot_response, timestamp)
        VALUES (?, ?, ?)
    """, (
        user_message,
        str(bot_response),      # temporary
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

def get_chats(limit: int = 20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response, timestamp
        FROM chats ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows