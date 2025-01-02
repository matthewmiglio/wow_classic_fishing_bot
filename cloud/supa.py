import time

from supabase import create_client


import os
import platform
import socket


def get_current_os():
    system_name = platform.system()

    if system_name == "Windows":
        version = platform.version()  # Get the detailed version (e.g., 10.0.19041)
        if "10" in version:
            return "Windows 10"
        elif "11" in version:
            return "Windows 11"
        else:
            return f"Windows ({version})"
    elif system_name == "Darwin":
        return "macOS"
    elif system_name == "Linux":
        return "Linux"
    else:
        return system_name


def get_system_uid():
    username = "NULL"
    pc_name = "NULL"
    current_os = "NULL"

    try:
        username = os.getlogin()
    except:
        pass
    try:
        pc_name = socket.gethostname()
    except:
        pass
    try:
        current_os = get_current_os()
    except:
        pass

    return f"{username}_{pc_name}_{current_os}".replace("-", "_").replace(" ", "_")


class Supa:
    def __init__(self):
        self.url = "https://ubingyvgrlgoknfkhknc.supabase.co"
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InViaW5neXZncmxnb2tuZmtoa25jIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU4NTQxNjIsImV4cCI6MjA1MTQzMDE2Mn0.ntMn7QiENPLSkPqO2U_4mYpAwmtLO69cIrZBgEQSM9I"
        self.supabase = create_client(self.url, self.key)

    def insert(self, table_name: str, data: dict):
        try:
            response = self.supabase.table(table_name).insert(data).execute()
        except Exception as e:
            return e
        return response

    def fetch_all_data(self, table_name: str):
        try:
            response = self.supabase.table(table_name).select("*").execute()
            return response.data
        except Exception as e:
            return e


class UsersTable:
    def __init__(self):
        self.supa = Supa()
        self.table_name = "fishbot-users-table"

    def get_users(self):
        return self.supa.fetch_all_data(self.table_name)

    def add_user(self):
        user_data = {
            "uuid": get_system_uid(),
            "timestamp": time.time(),
        }

        out = self.supa.insert(self.table_name, user_data)
        if "'code': '23505'" in str(out):
            print("User already exists")
            return "user already exists"

        if "data=[{'uuid':" in str(out):
            print("User successfully added")
            return "user successfully added"

        print("unknown response from add_user:", out)


class StatsTable:
    def __init__(self):
        self.supa = Supa()
        self.table_name = "fishbot-stats-table"

    def get_all_stats(self):
        return self.supa.fetch_all_data(self.table_name)

    def add_stats(self, runtime, reels, casts, loots):
        stats_data = {
            "uuid": get_system_uid(),
            "runtime": runtime,
            "reels": reels,
            "casts": casts,
            "loots": loots,
            "timestamp": time.time(),
        }

        out = self.supa.insert(self.table_name, stats_data)

        if "'code': '22P02'" in str(out):
            print("Type mismatch for this attempted stats addition")
            return

        if "data=[" in str(out):
            print("Stats added")
            return

        print("unexpected response from add_stats:", out)


def test():
    import random

    print("\n\n\ninserting:")
    print("---" * 30)

    ut = UsersTable()
    st = StatsTable()

    runtime = random.randint(200, 9000)
    reels = random.randint(200, 500)
    casts = reels + random.randint(0, 100)
    loots = reels - random.randint(0, 100)

    ut.add_user()
    st.add_stats(runtime, reels, casts, loots)

    print("\n\n\n\n\n\nQuerying:")
    print("---" * 30)

    users = ut.get_users()
    print(f"Existing users:")
    for user in users:
        print(f"\t{user['uuid']} : {user['timestamp']}")

    all_stats = st.get_all_stats()
    print(f"The stats:")
    for stat_row in all_stats:
        print(
            f"\t{stat_row['uuid']} : {stat_row['runtime']}s : {stat_row['casts']} casts : {stat_row['loots']} loots : {stat_row['timestamp']}"
        )


if __name__ == "__main__":
    test()
