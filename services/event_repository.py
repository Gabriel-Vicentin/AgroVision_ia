import sqlite3
from datetime import datetime
from typing import List, Dict

from services import config


def init_db():
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_time TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def save_event(event_id: str, label: str, confidence: float, image_path: str):
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (id, event_time, label, confidence, image_path)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            event_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            confidence,
            image_path,
        ),
    )
    conn.commit()
    conn.close()


def list_events(limit: int = 50) -> List[Dict]:
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, event_time, label, confidence, image_path
        FROM events
        ORDER BY event_time DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "event_time": row[1],
            "label": row[2],
            "confidence": row[3],
            "image_path": row[4],
        }
        for row in rows
    ]


def count_events() -> int:
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0
