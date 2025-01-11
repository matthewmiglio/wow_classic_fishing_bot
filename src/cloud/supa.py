import pickle
import time

from supabase import create_client


import os
import platform
import socket

MODULE_LEVEL_PRINT = False


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
    un, pcn, cos = "NULL", "NULL", "NULL"

    try:
        un = os.getlogin()
    except:
        pass
    try:
        pcn = socket.gethostname()
    except:
        pass
    try:
        cos = get_current_os()
    except:
        pass

    return f"{un}_{pcn}_{cos}".replace("-", "_").replace(" ", "_")


class Supa:
    def __init__(self):
        pklr = Pickler()
        self.supabase = create_client(pklr.get("u"), pklr.get("k"))

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
            if MODULE_LEVEL_PRINT:
                print("User already exists")
            return "user already exists"

        if "data=[{'uuid':" in str(out):
            if MODULE_LEVEL_PRINT:
                print("User successfully added")
            return "user successfully added"

        if MODULE_LEVEL_PRINT:
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
            if MODULE_LEVEL_PRINT:
                print("Type mismatch for this attempted stats addition")
            return

        if "data=[" in str(out):
            if MODULE_LEVEL_PRINT:
                print("Stats added")
            return

        if MODULE_LEVEL_PRINT:
            print("unexpected response from add_stats:", out)


class UsageTable:
    def __init__(self):
        self.supa = Supa()
        self.table_name = "fishbot-usage-table"

    def get_user_usage(self, uuid):
        """Fetches the usage data for a specific user using uuid"""
        try:
            response = (
                self.supa.supabase.table(self.table_name)
                .select("*")
                .eq("uuid", uuid)
                .execute()
            )
            if response.data:
                return response.data[0]  # Assuming there's only one record per uuid
            else:
                return None
        except Exception as e:
            return e

    def increment_uses(self):
        """Increments the 'starts' counter for a user, creates a new user if necessary"""
        user_uuid = get_system_uid()
        user_usage = self.get_user_usage(user_uuid)

        if user_usage is None:
            # If the user does not exist, create an entry
            usage_data = {
                "uuid": user_uuid,
                "starts": 1,
                "last_use_time": time.time(),
            }
            out = self.supa.insert(self.table_name, usage_data)
            if "data=[" in str(out):
                if MODULE_LEVEL_PRINT:
                    print("New user usage added")
                return "new user usage added"
            if MODULE_LEVEL_PRINT:
                print("Unexpected response from increment_uses:", out)
            return

        # If the user already exists, increment the 'starts' value
        new_starts = user_usage["starts"] + 1
        updated_usage_data = {
            "starts": new_starts,
        }

        out = (
            self.supa.supabase.table(self.table_name)
            .update(updated_usage_data)
            .eq("uuid", user_uuid)
            .execute()
        )
        if "data=[" in str(out):
            if MODULE_LEVEL_PRINT:
                print("User usage incremented")
            return "user usage incremented"
        if MODULE_LEVEL_PRINT:
            print("Unexpected response from increment_uses:", out)

    def set_last_use_time(self):
        """Sets the 'last_use_time' for a user"""
        user_uuid = get_system_uid()
        user_usage = self.get_user_usage(user_uuid)

        if user_usage is None:
            # If the user does not exist, create an entry
            usage_data = {
                "uuid": user_uuid,
                "starts": 1,
                "last_use_time": time.time(),
            }
            out = self.supa.insert(self.table_name, usage_data)
            if "data=[" in str(out):
                if MODULE_LEVEL_PRINT:
                    print("New user usage added")
                return "new user usage added"
            if MODULE_LEVEL_PRINT:
                print("Unexpected response from set_last_use_time:", out)
            return

        # If the user exists, update the 'last_use_time'
        updated_usage_data = {
            "last_use_time": time.time(),
        }

        out = (
            self.supa.supabase.table(self.table_name)
            .update(updated_usage_data)
            .eq("uuid", user_uuid)
            .execute()
        )
        if "data=[" in str(out):
            if MODULE_LEVEL_PRINT:
                print("User last use time updated")
            return "user last use time updated"
        if MODULE_LEVEL_PRINT:
            print("Unexpected response from set_last_use_time:", out)


def test():
    import random

    if MODULE_LEVEL_PRINT:
        print("\n\n\ninserting:")
    if MODULE_LEVEL_PRINT:
        print("---" * 30)

    ut = UsersTable()
    st = StatsTable()

    runtime = random.randint(200, 9000)
    reels = random.randint(200, 500)
    casts = reels + random.randint(0, 100)
    loots = reels - random.randint(0, 100)

    ut.add_user()
    st.add_stats(runtime, reels, casts, loots)

    if MODULE_LEVEL_PRINT:
        print("\n\n\n\n\n\nQuerying:")
    if MODULE_LEVEL_PRINT:
        print("---" * 30)

    users = ut.get_users()
    if MODULE_LEVEL_PRINT:
        print(f"Existing users:")
    for user in users:
        if MODULE_LEVEL_PRINT:
            print(f"\t{user['uuid']} : {user['timestamp']}")

    all_stats = st.get_all_stats()
    if MODULE_LEVEL_PRINT:
        print(f"The stats:")
    for stat_row in all_stats:
        if MODULE_LEVEL_PRINT:
            print(
                f"\t{stat_row['uuid']} : {stat_row['runtime']}s : {stat_row['casts']} casts : {stat_row['loots']} loots : {stat_row['timestamp']}"
            )


def test_usage_table():
    ut = UsageTable()
    ut.increment_uses()
    ut.set_last_use_time()


class Pickler:
    def __init__(self):
        self.kfp = os.path.join(os.getcwd(), 'src',"cloud", "data", "XbsgAJG8sbA2.pkl")
        self.ufp = os.path.join(os.getcwd(), 'src',"cloud", "data", "iKnsfabt73hB.pkl")
        os.makedirs(os.path.join(os.getcwd(), 'src',"cloud", "data"), exist_ok=True)

    def place(self, string: str, type: str):
        if type not in ["k", "u"]:
            if MODULE_LEVEL_PRINT:
                print("invalid type for key.place()")
            return

        if type == "k":
            with open(self.kfp, "wb") as file:
                pickle.dump(string, file)
        else:
            with open(self.ufp, "wb") as file:
                pickle.dump(string, file)

    def get(self, type: str):
        if type not in ["k", "u"]:
            if MODULE_LEVEL_PRINT:
                print("invalid type for key.get()")
            return

        if type == "k":
            with open(self.kfp, "rb") as file:
                return pickle.load(file)
        else:
            with open(self.ufp, "rb") as file:
                return pickle.load(file)


if __name__ == "__main__":
    # test()
    test_usage_table()
