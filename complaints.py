
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "roadcare.db"


def create_complaints_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                annotated_path TEXT NOT NULL,
                location_text TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                damage_summary TEXT NOT NULL,
                status TEXT DEFAULT 'Pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


def save_complaint(
    complaint_id,
    user_id,
    image_path,
    annotated_path,
    location_text,
    latitude,
    longitude,
    damage_summary
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO complaints (
                complaint_id, user_id, image_path, annotated_path,
                location_text, latitude, longitude, damage_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            complaint_id, user_id, image_path, annotated_path,
            location_text, latitude, longitude, damage_summary
        ))


def get_all_complaints():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT
                c.id, c.complaint_id, u.name, u.email,
                c.image_path, c.annotated_path, c.location_text,
                c.latitude, c.longitude, c.damage_summary,
                c.status, c.created_at
            FROM complaints c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.id DESC
        """).fetchall()


def get_user_complaints(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT complaint_id, location_text, damage_summary,
                   status, created_at
            FROM complaints
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()


def update_complaint_status(complaint_id, status):
    allowed = ("Pending", "In Progress", "Resolved")

    if status not in allowed:
        raise ValueError("Invalid complaint status")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE complaints
            SET status = ?
            WHERE complaint_id = ?
        """, (status, complaint_id))
