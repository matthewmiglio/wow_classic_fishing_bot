import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import numpy as np


GUI_WINDOW_NAME = "WoW Fishing Bot Display"
from constants import BLACKLIST_STRINGS


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title(GUI_WINDOW_NAME)

        # Create frames for images and stats
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Create labels for images
        self.raw_image_label = tk.Label(self.image_frame, text="Raw Image")
        self.raw_image_label.grid(row=0, column=0)

        self.bobber_image_label = tk.Label(self.image_frame, text="Bobber Image")
        self.bobber_image_label.grid(row=0, column=1)

        self.raw_image_display = tk.Label(self.image_frame)
        self.raw_image_display.grid(row=1, column=0)

        self.bobber_image_display = tk.Label(self.image_frame)
        self.bobber_image_display.grid(row=1, column=1)

        # Create stats table
        self.stats = {
            "bobber_detected": tk.StringVar(value="No"),
            "splash_detected": tk.StringVar(value="No"),
            "casts": tk.StringVar(value=0),
            "reels": tk.StringVar(value=0),
            "runtime": tk.StringVar(value=0),
        }

        ttk.Label(self.stats_frame, text="Bobber Detected").grid(row=0, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["bobber_detected"]).grid(
            row=0, column=1
        )

        ttk.Label(self.stats_frame, text="Splash Detected").grid(row=1, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["splash_detected"]).grid(
            row=1, column=1
        )

        ttk.Label(self.stats_frame, text="Casts").grid(row=2, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["casts"]).grid(
            row=2, column=1
        )

        ttk.Label(self.stats_frame, text="Reels").grid(row=3, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["reels"]).grid(
            row=3, column=1
        )

        ttk.Label(self.stats_frame, text="Runtime").grid(row=4, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["runtime"]).grid(
            row=4, column=1
        )

        # save settings location
        self.save_settings_path = r"settings.txt"

        # Create Start and Stop buttons
        self.start_button = ttk.Button(root, text="Start", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_bot)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create the "Open Blacklist Settings" button
        self.blacklist_button = ttk.Button(
            root, text="Blacklist Settings", command=self.open_blacklist_gui
        )
        self.blacklist_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.bot = None  # Reference to the bot

    def set_bot(self, bot):
        """Set the bot instance for starting and stopping."""
        self.bot = bot

    def start_bot(self):
        """Start the bot."""
        if (
            self.bot and not self.bot.running_event.is_set()
        ):  # Check if bot is not already running
            self.bot.running_event.set()  # Signal to start running
            bot_thread = threading.Thread(target=self.bot.run, daemon=True)
            bot_thread.start()  # Start the bot in a separate thread

    def stop_bot(self):
        """Stop the bot."""
        if self.bot:
            self.bot.stop()  # Signal to stop running

    def update_image(self, image_array, image_type="raw"):
        """Update the displayed image."""
        image = Image.fromarray(image_array)
        image = ImageTk.PhotoImage(image)

        if image_type == "raw":
            self.raw_image_display.config(image=image)
            self.raw_image_display.image = (
                image  # Keep a reference to avoid garbage collection
            )
        elif image_type == "bobber":
            self.bobber_image_display.config(image=image)
            self.bobber_image_display.image = (
                image  # Keep a reference to avoid garbage collection
            )

    def update_stat(self, stat_name, value):
        """Update the stats display."""
        if stat_name in self.stats:
            self.stats[stat_name].set(value)

    def open_blacklist_gui(self):
        """Open the popup window for blacklist settings."""
        self.blacklist_window = tk.Toplevel(self.root)
        self.blacklist_window.title("Blacklist Settings")

        self.checkboxes = {}

        # Create checkboxes dynamically
        for index, text in enumerate(BLACKLIST_STRINGS):
            var = tk.BooleanVar()
            checkbox = tk.Checkbutton(self.blacklist_window, text=text, variable=var)
            checkbox.grid(row=index, column=0, sticky="w")
            self.checkboxes[text] = var

        # Save and Close buttons
        self.save_button = ttk.Button(
            self.blacklist_window,
            text="Save Settings",
            command=self.save_blacklist_settings,
        )
        self.save_button.grid(row=len(BLACKLIST_STRINGS), column=0, padx=5, pady=5)

        self.close_button = ttk.Button(
            self.blacklist_window, text="Close", command=self.close_blacklist_gui
        )
        self.close_button.grid(row=len(BLACKLIST_STRINGS), column=1, padx=5, pady=5)

    def save_blacklist_settings(self):
        """Placeholder method for saving blacklist settings."""
        selected_items = [fish for fish, var in self.checkboxes.items() if var.get()]
        with open(self.save_settings_path, "w") as f:
            for item in selected_items:
                f.write("%s," % item)

        self.blacklist_window.destroy()

    def get_blacklist_settings(self):
        """Placeholder method for getting blacklist settings."""
        with open(self.save_settings_path, "r") as f:
            selected_items = f.read().split(",")
        return selected_items

    def close_blacklist_gui(self):
        """Close the blacklist settings window."""
        self.blacklist_window.destroy()
