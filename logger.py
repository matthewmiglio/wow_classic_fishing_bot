from cloud.supa import StatsTable, UsersTable, UsageTable
from _FEATURE_FLAGS import SAVE_LOGS_FEATURE, CLOUD_STATS_FEATURE

import os
import time

class Logger:
    def __init__(self):
        print("Initializing Logger...")
        self.logs_folder = r"logs"
        self.fishing_attempts_log = os.path.join(
            self.logs_folder, "fishing_attempts.txt"
        )
        self.loot_log = os.path.join(self.logs_folder, "loot_log.txt")
        self.init_log_files()

        if CLOUD_STATS_FEATURE is True:
            self.cloud_update_increment = 1 * 60 * 60  # update every N hours
            self.first_cloud_update_buffer = 1 * 60  # do first update after N minutes
            self.stats_table: StatsTable = StatsTable()
            self.users_table: UsersTable = UsersTable()
            self.usage_table = UsageTable()
            self.users_table.add_user()

            # init time_of_last_cloud_update so we update it
            # self.first_cloud_update_buffer seconds after starting
            self.time_of_last_cloud_update = (
                time.time()
                - self.cloud_update_increment
                + self.first_cloud_update_buffer
            )

    def should_cloud_update(self):
        if not CLOUD_STATS_FEATURE:
            return False

        if time.time() - self.time_of_last_cloud_update > self.cloud_update_increment:
            self.time_of_last_cloud_update = time.time()
            return True

        return False

    def init_log_files(self):
        if SAVE_LOGS_FEATURE is not True:
            return
        os.makedirs(self.logs_folder, exist_ok=True)

        if not os.path.exists(self.fishing_attempts_log):
            with open(self.fishing_attempts_log, "w") as f:
                f.write("")

        if not os.path.exists(self.loot_log):
            with open(self.loot_log, "w") as f:
                f.write("")

    def add_to_fishing_log(self, type: str):
        if SAVE_LOGS_FEATURE is not True:
            return
        with open(self.fishing_attempts_log, "a") as f:
            log_string = f"{time.time()} {type}\n"
            f.write(log_string)

    def add_to_loot_log(self, loot: str):
        if SAVE_LOGS_FEATURE is not True:
            return
        with open(self.loot_log, "a") as f:
            f.write(f"{time.time()} {loot}\n")

    def get_fishing_log(self):
        with open(self.fishing_attempts_log, "r") as f:
            return f.read()

    def get_loot_log(self):
        with open(self.loot_log, "r") as f:
            return f.read()

