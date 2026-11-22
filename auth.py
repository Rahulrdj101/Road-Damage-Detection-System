
import sqlite3
import bcrypt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "roadcare.db"


def create_users_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                role TEXT NOT NULL
                    CHECK(role IN ('citizen', 'admin', 'official')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def register_user(name, email, password, role="citizen"):
    if role not in ("citizen", "admin", "official"):
        return False, "Invalid role."

    if not name.strip() or not email.strip():
        return False, "Name and email are required."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO users (name, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            """, (name.strip(), email.strip().lower(),
                  password_hash, role))

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "This email is already registered."


def login_user(email, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""
            SELECT id, name, email, password_hash, role
            FROM users
            WHERE email = ?
        """, (email.strip().lower(),)).fetchone()

    if row is None:
        return None

    if bcrypt.checkpw(password.encode("utf-8"), row[3]):
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "role": row[4]
        }

    return None
