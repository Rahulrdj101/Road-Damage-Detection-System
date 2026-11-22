
import getpass
import sqlite3

from auth import create_users_table, register_user, DB_PATH


def main():
    create_users_table()

    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'"
        ).fetchone()[0]

    if count > 0:
        print("An administrator account already exists.")
        return

    print("\n===== ROADCARE AI: INITIAL ADMIN SETUP =====")

    name = input("Admin name: ").strip()
    email = input("Admin email: ").strip()
    password = getpass.getpass("Create password (minimum 8 characters): ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Passwords do not match.")
        return

    success, message = register_user(
        name, email, password, role="admin"
    )

    print(message)


if __name__ == "__main__":
    main()
