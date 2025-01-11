import os
import time

from _FEATURE_FLAGS import CLOUD_STATS_FEATURE
from cloud.supa import StatsTable, UsageTable, UsersTable


class Logger:
    def __init__(self):
        print("Initializing Logger...")
        self.logs_folder = r"logs"
        self.fishing_attempts_log = os.path.join(
            self.logs_folder, "fishing_attempts.txt"
        )
        self.loot_log = os.path.join(self.logs_folder, "loot_log.txt")

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
