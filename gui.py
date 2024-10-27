import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class SimpleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WoW Fishing Bot Display")

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

        # Create labels for images
        self.raw_image_display = tk.Label(self.image_frame)
        self.raw_image_display.grid(row=1, column=0)

        self.bobber_image_display = tk.Label(self.image_frame)
        self.bobber_image_display.grid(row=1, column=1)

        # Create stats table
        self.stats = {
            "bobber_detected": tk.StringVar(value="No"),
            "splash_detected": tk.StringVar(value="No"),
        }

        ttk.Label(self.stats_frame, text="Bobber Detected").grid(row=0, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["bobber_detected"]).grid(row=0, column=1)

        ttk.Label(self.stats_frame, text="Splash Detected").grid(row=1, column=0)
        ttk.Label(self.stats_frame, textvariable=self.stats["splash_detected"]).grid(row=1, column=1)

    def update_image(self, image_array, image_type='raw'):
        """Update the displayed image."""
        image = Image.fromarray(image_array)
        image = ImageTk.PhotoImage(image)

        if image_type == 'raw':
            self.raw_image_display.config(image=image)
            self.raw_image_display.image = image  # Keep a reference to avoid garbage collection
        elif image_type == 'bobber':
            self.bobber_image_display.config(image=image)
            self.bobber_image_display.image = image  # Keep a reference to avoid garbage collection

    def update_stat(self, stat_name, value):
        """Update the stats display."""
        if stat_name in self.stats:
            self.stats[stat_name].set(value)

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    gui = SimpleGUI(root)

    # Example to update values and images
    gui.update_stat("bobber_detected", "Yes")
    gui.update_stat("splash_detected", "No")

    # Example image updates (use random numpy arrays for demonstration)
    raw_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    bobber_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    gui.update_image(raw_image, 'raw')
    gui.update_image(bobber_image, 'bobber')

    root.mainloop()
