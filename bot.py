from PIL import Image
import threading
import pyautogui
import numpy as np
import time
import os
import tkinter as tk
import cv2
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier
from gui import GUI


class WoWFishBot:
    def __init__(self, gui, save_images=False):
        self.gui = gui
        self.bobber_detector = BobberDetector(r"inference\bobber_finder.onnx")
        self.splash_classifier = SplashClassifier(r"inference\splash_classifier.onnx")
        self.save_splash_images_toggle = save_images
        self.save_images_dir = "save_images"

        self.casts = 0
        self.reels = 0
        self.prediction_history = []  # used to store the last 8 predictions
        self.splash_prediction_history_limit = 12

        self.running_event = threading.Event()  # Control the bot running state

    def add_splash_prediction(self, prediction):
        # add the prediction to the history while removing the oldest
        if "not" in prediction:
            prediction = "not"
        elif "splash" in prediction:
            prediction = "splash"
        self.prediction_history.append(prediction)

        # if its longer than limit remove the oldest one
        if len(self.prediction_history) > self.splash_prediction_history_limit:
            self.prediction_history.remove(self.prediction_history[0])

    def last_predictions_equal(self, prediction_string, equal_count):
        if prediction_string not in self.prediction_history[-1]:
            return False

        this_count = 0
        for i in self.prediction_history:
            if prediction_string in i:
                this_count += 1

        if this_count > equal_count:
            self.prediction_history = []
            return True

        return False

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
        self.casts += 1
        self.update_gui("casts", self.casts)

    def click_bbox(self, bbox):
        coord = self.calculate_coordinates(bbox)
        print("Reeling in the bobber with this coord: ", coord)
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

    def update_gui(self, stat, value):
        if stat == "raw_image":
            try:
                value = numpy_img_bgr_to_rgb(value)
                self.gui.update_image(value, "raw")
            except:
                pass
        if stat == "bobber_image":
            try:
                value = numpy_img_bgr_to_rgb(value)
                value = cv2.resize(value, (256, 256))
                self.gui.update_image(value, "bobber")
            except:
                pass
        if stat == "bobber_detected":
            self.gui.update_stat("bobber_detected", value)
        if stat == "splash_detected":
            self.gui.update_stat(
                "splash_detected", "Yes" if "splash" in value else "No"
            )
        if stat == "casts":
            self.gui.update_stat("casts", value)
        if stat == "reels":
            self.gui.update_stat("reels", value)

    def run(self):
        MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25
        print("Initializing Bobber Detector and Splash Classifier modules!")

        self.running_event.set()  # Start the event

        while self.running_event.is_set():
            base_image = self.screenshot_bobber_roi()
            self.save_roi_image(base_image)
            self.update_gui("raw_image", base_image)

            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # if a bobber detected
            if score > MIN_CONFIDENCE_FOR_BOBBER_DETECTION:
                # get the image of the bobber based on the bbox we infered
                bobber_image = self.make_bobber_image(bbox, base_image)
                self.update_gui("bobber_image", bobber_image)

                # get the bobber image ready for the splash classifier
                bobber_image = self.splash_classifier.preprocess(bobber_image)
                if bobber_image is False:
                    continue

                # classify that bobber image as either a 'splash' or 'not' a splash
                is_splash = self.splash_classifier.run(bobber_image, draw_result=False)
                is_splash = self.splash_classifier.postprocess(is_splash)

                # show the result in the gui
                self.update_gui("bobber_detected", "Yes")
                self.update_gui("splash_detected", is_splash)

                # if the bobber is a splash, we click it to reel in the fish
                self.add_splash_prediction(is_splash)
                if self.last_predictions_equal("splash", 6):
                    print(
                        "Splash detected! Reeling in fish...\nSelf.splash_prediction_history: ",
                        self.prediction_history,
                    )
                    self.reels += 1
                    self.update_gui("reels", self.reels)
                    self.click_bbox(bbox)

                # if we want to save the splash images
                if self.save_splash_images_toggle:
                    self.save_splash_images(bbox, base_image, is_splash)

            # if none detected at all
            else:
                self.gui.update_stat("bobber_detected", "No")
                self.gui.update_image(base_image, "bobber")
                self.gui.update_stat("splash_detected", "No")
                self.add_splash_prediction("none")
                if self.last_predictions_equal("none", 11):
                    print("Starting fishing...")
                    self.start_fishing()
                    time.sleep(2)

    def stop(self):
        self.running_event.clear()  # Stop the bot

    def save_roi_image(self, img):
        uid = "roi_" + str(time.time()).replace(".", "") + ".png"
        path = os.path.join(self.save_images_dir, "roi", uid)
        img = numpy_img_bgr_to_rgb(img)

        img = Image.fromarray(img)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

    def save_splash_images(self, bbox, img, is_splash) -> bool:
        uid = str(time.time()).replace(".", "") + ".png"
        path = os.path.join(
            self.save_images_dir, "splash" if "splash" in is_splash else "not", uid
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # convert it from BGR to RGB
        img = numpy_img_bgr_to_rgb(img)
        img = Image.fromarray(img)
        bbox = xywy2xyxy(*bbox)
        img = img.crop(bbox)
        img.save(path)
        return True


def numpy_img_bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def xywy2xyxy(x, y, w, h):
    return x - w // 2, y - h // 2, x + w // 2, y + h // 2


def run_bot_with_gui():
    root = tk.Tk()
    gui = GUI(root)
    bot = WoWFishBot(gui, save_images=True)
    gui.set_bot(bot)  # Pass the bot to the GUI for control
    root.mainloop()


if __name__ == "__main__":
    run_bot_with_gui()
