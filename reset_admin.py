
import sqlite3
import bcrypt
import getpass

from auth import DB_PATH, create_users_table


def main():
    create_users_table()

    print("\n===== ROADCARE AI ADMIN RESET =====")

    name = input("Enter new Admin name: ").strip()
    email = input("Enter new Admin email: ").strip().lower()

    if not name or not email or "@" not in email:
        print("Please enter a valid name and email.")
        return

    password = getpass.getpass("Enter new Admin password: ")
    confirm = getpass.getpass("Confirm new Admin password: ")

    if len(password) < 8:
        print("Password must contain at least 8 characters.")
        return

    if password != confirm:
        print("Passwords do not match.")
        return

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    try:
        with sqlite3.connect(DB_PATH) as conn:

            # Do not overwrite a Citizen or Official account.
            existing = conn.execute(
                "SELECT role FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            if existing and existing[0] != "admin":
                print("This email belongs to another account.")
                return

            # Replace existing Admin accounts.
            conn.execute(
                "DELETE FROM users WHERE role = 'admin'"
            )

            conn.execute("""
                INSERT INTO users
                (name, email, password_hash, role)
                VALUES (?, ?, ?, 'admin')
            """, (name, email, password_hash))

        print("\nAdmin account recreated successfully!")
        print("New Admin name:", name)
        print("New Admin email:", email)
        print("Password: Saved securely as a bcrypt hash.")

    except sqlite3.Error as error:
        print("Database error:", error)


if __name__ == "__main__":
    main()
