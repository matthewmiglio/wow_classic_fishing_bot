import threading
import pyautogui
import numpy as np
import time
import os
import tkinter as tk
import cv2
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier
from gui import SimpleGUI  # Make sure this import matches your GUI file

class WoWFishBot:
    def __init__(self, gui, save_splash_images_toggle=False):
        self.gui = gui
        self.bobber_detector = BobberDetector(r"inference\bobber_finder.onnx")
        self.splash_classifier = SplashClassifier(r"inference\splash_classifier.onnx")
        self.save_splash_images_toggle = save_splash_images_toggle
        self.splash_images_dir = "splash_images"
        self.splash_labels = ["not", ""]

        if self.save_splash_images_toggle:
            os.makedirs(self.splash_images_dir, exist_ok=True)
            os.makedirs(os.path.join(self.splash_images_dir, "not"), exist_ok=True)
            os.makedirs(os.path.join(self.splash_images_dir, "splash"), exist_ok=True)

    def screenshot_bobber_roi(self):
        image = pyautogui.screenshot()
        image = image.crop((700, 100, 1300, 900)).resize((256, 256))
        image = np.array(image)
        image = image[:, :, ::-1].copy()  # Convert BGR to RGB
        return image

    def make_bobber_image(self, bbox, img):
        xywh = bbox
        x_pad, y_pad = 10, 10
        x1 = (xywh[0] - xywh[2] // 2) - x_pad
        y1 = (xywh[1] - xywh[3] // 2) - y_pad
        x2 = (xywh[0] + xywh[2] // 2) + x_pad
        y2 = (xywh[1] + xywh[3] // 2) + y_pad
        bbox_image = img[y1:y2, x1:x2]
        return bbox_image

    def focus_wow(self):
        wow_deadspace = (1800, 100)
        pyautogui.click(*wow_deadspace)
        time.sleep(0.2)

    def start_fishing(self):
        self.focus_wow()
        pyautogui.press("z")
        time.sleep(2)

    def click_bbox(self, bbox):
        coord = self.calculate_coordinates(bbox)
        print(f"CLICKING THIS COORD: {coord}")
        self.send_delayed_click(*coord, wait=1)

    def send_delayed_click(self, x, y, wait):
        def _to_wrap(x, y, wait):
            time.sleep(wait)
            pyautogui.click(x, y, button="right")

        t = threading.Thread(target=_to_wrap, args=(x, y, wait))
        t.start()

    def calculate_coordinates(self, bbox):
        x, y, w, h = bbox
        this_image_dim = 256
        x_ratio = x / this_image_dim
        y_ratio = y / this_image_dim
        prev_image_dim_x = 600
        prev_image_dim_y = 800
        start_x = 700
        start_y = 100
        real_y = start_y + (y_ratio * prev_image_dim_y)
        real_x = start_x + (x_ratio * prev_image_dim_x)
        return (real_x, real_y)

    def run(self):
        MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25
        print("Initializing Bobber Detector and Splash Classifier modules!")

        while True:
            base_image = self.screenshot_bobber_roi()
            self.gui.update_image(base_image, 'raw')
            bbox, score = self.bobber_detector.detect_object_in_image(base_image, draw_result=False)

            if score > MIN_CONFIDENCE_FOR_BOBBER_DETECTION:
                try:
                    bobber_image = self.make_bobber_image(bbox, base_image)
                    self.gui.update_image(bobber_image, 'bobber')
                    bobber_image = self.splash_classifier.preprocess(bobber_image)
                except Exception as e:
                    continue

                is_splash = self.splash_classifier.run(bobber_image, draw_result=False)
                is_splash = self.splash_classifier.postprocess(is_splash)

                self.gui.update_stat("bobber_detected", "Yes")
                self.gui.update_stat("splash_detected", "Yes" if "splash" in is_splash else "No")

                if "splash" in is_splash:
                    self.click_bbox(bbox)

                if self.save_splash_images_toggle:
                    self.save_splash_images(bobber_image, is_splash)

            else:
                print("Bobber is not detected!\nStarting fishing...")
                self.gui.update_stat("bobber_detected", "No")
                self.start_fishing()

    def save_splash_images(self, bbox_image, is_splash) -> bool:
        uid = str(time.time()).replace(".", "") + ".png"
        path = os.path.join(self.splash_images_dir, "splash" if "splash" in is_splash else "not", uid)

        try:
            cv2.imwrite(path, bbox_image)

        except Exception as e:
            print(f"Failed to save this splash image:\n{e}")
            return False

        return True

def run_bot_with_gui():
    root = tk.Tk()
    gui = SimpleGUI(root)
    bot = WoWFishBot(gui,save_splash_images_toggle=False)

    # Run the bot in a separate thread
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()

    root.mainloop()

if __name__ == "__main__":
    run_bot_with_gui()
