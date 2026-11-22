
import sqlite3
import bcrypt
import getpass

from auth import DB_PATH


def main():
    email = input("Enter Admin email: ").strip().lower()

    with sqlite3.connect(DB_PATH) as conn:
        user = conn.execute(
            "SELECT id, password_hash FROM users WHERE email=? AND role='admin'",
            (email,)
        ).fetchone()

        if not user:
            print("Admin account not found.")
            return

        current_password = getpass.getpass("Current password: ")

        if not bcrypt.checkpw(current_password.encode(), user[1]):
            print("Incorrect current password.")
            return

        new_password = getpass.getpass("New password: ")

        if len(new_password) < 8:
            print("Password must be at least 8 characters.")
            return

        confirm_password = getpass.getpass("Confirm new password: ")

        if new_password != confirm_password:
            print("Passwords do not match.")
            return

        new_hash = bcrypt.hashpw(
            new_password.encode(),
            bcrypt.gensalt()
        )

        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (new_hash, user[0])
        )

        print("Admin password changed successfully!")


if __name__ == "__main__":
    main()
