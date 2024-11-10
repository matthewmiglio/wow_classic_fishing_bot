import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from loot_constants import BLACKLIST_STRINGS


GUI_WINDOW_NAME = "WoW Fishing Bot Display"
GUI_SIZE = "450x720"

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.geometry(GUI_SIZE)
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

        # Create frame for the loot history bar graph
        self.graph_frame = tk.Frame(self.image_frame)
        self.graph_frame.grid(row=2, column=0, columnspan=2)
        self.figure, self.ax = plt.subplots(figsize=(4, 3.2))

        self.figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.3)

        # pack that graph
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack()

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

    def configure_graph(self):
        """Configure the aesthetics and settings of the graph."""
        self.ax.set_title("Seen Fish")
        self.ax.set_ylabel("#")

        # Set up y-axis formatting
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        # Configure tick settings
        self.ax.tick_params(axis="x", labelsize=8)
        for label in self.ax.get_xticklabels():
            label.set_rotation(45)  # Rotate labels to avoid overlap
            label.set_wrap(True)    # Enable text wrapping

        # Ensure x-axis labels don't overlap
        self.ax.set_xticklabels(
            self.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

    def update_loot_history(self, loot_history):
        """Update the loot history bar graph."""
        # Count occurrences of each fish
        fish_counts = {fish: loot_history.count(fish) for fish in set(loot_history)}

        # Clear the previous graph
        self.ax.clear()

        # Apply graph configuration
        self.configure_graph()

        # Plot the new data
        bars = self.ax.bar(fish_counts.keys(), fish_counts.values())

        # Set x-ticks to be at the center of each bar (this is important if you have non-numeric categories)
        self.ax.set_xticks([bar.get_x() + bar.get_width() / 2 for bar in bars])

        # Set x-tick labels to the fish names
        self.ax.set_xticklabels(fish_counts.keys(), rotation=45, ha="right")

        # Redraw the canvas
        self.canvas.draw()





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

        self.whitelist_frame = tk.Frame(self.blacklist_window)
        self.whitelist_frame.grid(row=0, column=0, padx=10, pady=10)

        self.blacklist_frame = tk.Frame(self.blacklist_window)
        self.blacklist_frame.grid(row=0, column=1, padx=10, pady=10)

        # Create the listboxes
        self.whitelist_listbox = tk.Listbox(
            self.whitelist_frame, height=15, width=30, selectmode=tk.SINGLE
        )
        self.whitelist_listbox.pack(side=tk.LEFT, padx=5, pady=5)

        self.blacklist_listbox = tk.Listbox(
            self.blacklist_frame, height=15, width=30, selectmode=tk.SINGLE
        )
        self.blacklist_listbox.pack(side=tk.LEFT, padx=5, pady=5)

        # Buttons to move items
        self.move_to_blacklist_button = ttk.Button(
            self.blacklist_window,
            text="Add to Blacklist",
            command=self.move_to_blacklist,
        )
        self.move_to_blacklist_button.grid(row=1, column=0, padx=5, pady=5)

        self.move_to_whitelist_button = ttk.Button(
            self.blacklist_window,
            text="Add to Whitelist",
            command=self.move_to_whitelist,
        )
        self.move_to_whitelist_button.grid(row=1, column=1, padx=5, pady=5)

        # Buttons to move all items
        self.move_all_to_blacklist_button = ttk.Button(
            self.blacklist_window,
            text="Move All to Blacklist",
            command=self.move_all_to_blacklist,
        )
        self.move_all_to_blacklist_button.grid(row=2, column=0, padx=5, pady=5)

        self.move_all_to_whitelist_button = ttk.Button(
            self.blacklist_window,
            text="Move All to Whitelist",
            command=self.move_all_to_whitelist,
        )
        self.move_all_to_whitelist_button.grid(row=2, column=1, padx=5, pady=5)

        # Load the blacklist settings and populate the lists
        self.load_blacklist_settings()

        # Save and Close buttons
        self.save_button = ttk.Button(
            self.blacklist_window,
            text="Save Settings",
            command=self.save_blacklist_settings,
        )
        self.save_button.grid(row=3, column=0, padx=5, pady=5)

        self.close_button = ttk.Button(
            self.blacklist_window, text="Close", command=self.close_blacklist_gui
        )
        self.close_button.grid(row=3, column=1, padx=5, pady=5)

    def move_to_blacklist(self):
        """Move selected item from whitelist to blacklist."""
        selected_item = self.whitelist_listbox.get(tk.ACTIVE)
        if selected_item:
            self.whitelist_listbox.delete(tk.ACTIVE)
            self.blacklist_listbox.insert(tk.END, selected_item)

    def move_to_whitelist(self):
        """Move selected item from blacklist to whitelist."""
        selected_item = self.blacklist_listbox.get(tk.ACTIVE)
        if selected_item:
            self.blacklist_listbox.delete(tk.ACTIVE)
            self.whitelist_listbox.insert(tk.END, selected_item)

    def move_all_to_blacklist(self):
        """Move all items from whitelist to blacklist."""
        for item in self.whitelist_listbox.get(0, tk.END):
            self.blacklist_listbox.insert(tk.END, item)
        self.whitelist_listbox.delete(0, tk.END)

    def move_all_to_whitelist(self):
        """Move all items from blacklist to whitelist."""
        for item in self.blacklist_listbox.get(0, tk.END):
            self.whitelist_listbox.insert(tk.END, item)
        self.blacklist_listbox.delete(0, tk.END)

    def load_blacklist_settings(self):
        """Initialize the positions of each string according to settings.txt."""
        if os.path.exists(self.save_settings_path):
            with open(self.save_settings_path, "r") as f:
                saved_settings = f.read().split(",")
                saved_settings = [
                    item.strip() for item in saved_settings if item.strip()
                ]

            # Populate the whitelist and blacklist based on saved settings
            for item in BLACKLIST_STRINGS:
                if item in saved_settings:
                    self.blacklist_listbox.insert(tk.END, item)
                else:
                    self.whitelist_listbox.insert(tk.END, item)
        else:
            # If settings.txt does not exist, assume all items are whitelisted
            for item in BLACKLIST_STRINGS:
                self.whitelist_listbox.insert(tk.END, item)

    def save_blacklist_settings(self):
        """Save the current blacklist settings."""
        whitelist_items = list(self.whitelist_listbox.get(0, tk.END))
        blacklist_items = list(self.blacklist_listbox.get(0, tk.END))

        # Save the blacklist items to settings.txt
        with open(self.save_settings_path, "w") as f:
            for item in blacklist_items:
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
