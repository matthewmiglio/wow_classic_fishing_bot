from PIL import Image
import threading
import pyautogui
import numpy as np
import time
import pygetwindow
import os
import tkinter as tk
import cv2
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier
from gui import GUI

START_FISHING_COORD = (930, 1400)
FISHING_BUFF_COORD = (470, 1330)
FISHING_POLE_COORD = (520,1330)
BUFF_INCREMENT = 11 # minutes

class WoWFishBot:
    def __init__(self, gui, save_images=False,apply_buff=False):
        #gui
        self.gui = gui
        self.running_event = threading.Event()  # Control the bot running state

        #ai models
        self.bobber_detector = BobberDetector(
            r"inference\bobber_models\bobber_finder3.0.onnx"
        )
        self.splash_classifier = SplashClassifier(
            r"inference\splash_models\splash_classifier4.0.onnx"
        )  # test this. otherwise use 1.0

        #saving images
        self.save_splash_images_toggle = save_images
        self.save_images_dir = "save_images"

        #user toggles
        self.apply_buff = apply_buff

        #stats
        self.casts = 0
        self.reels = 0
        self.buffs = 0
        self.time_running = 0

        #prediction storage
        self.prediction_history = []  # used to store the last 8 predictions
        self.splash_prediction_history_limit = 12

    def should_buff(self):
        minutes_ran = self.time_running / 60

        target_count = (minutes_ran // BUFF_INCREMENT) +1

        if target_count > self.buffs:
            self.buffs += 1
            return True

        return False

    def apply_fishing_buff(self):
        pyautogui.click(*FISHING_BUFF_COORD,clicks=2,interval=0.22)
        time.sleep(0.1)
        pyautogui.click(*FISHING_POLE_COORD,clicks=2,interval=0.22)
        time.sleep(5.5)
        self.update_gui("buffs", self.buffs)

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
        close_sponsored_session_teamviewer()
        name = "World of Warcraft"
        try:
            window = pygetwindow.getWindowsWithTitle(name)[0]
            window.activate()
        except:
            return False

        return True

    def start_fishing(self):
        if self.apply_buff and self.should_buff():
            self.apply_fishing_buff()

        pyautogui.click(*START_FISHING_COORD)
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
        if self.gui is None:
            return

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
        if stat == "buffs":
            self.gui.update_stat("buffs", value)
        if stat == 'runtime':
            self.gui.update_stat("runtime", value)

    def run(self):
        MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25
        print("Initializing Bobber Detector and Splash Classifier modules!")

        self.running_event.set()  # Start the event

        while self.running_event.is_set():
            start_time = time.time()
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
                    self.add_time_taken(time.time() - start_time)
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
                self.gui.update_image(self.make_no_bobber_image(), "bobber")
                self.gui.update_stat("splash_detected", "No")
                self.add_splash_prediction("none")
                if self.last_predictions_equal("none", 11):
                    print("Starting fishing...")
                    self.start_fishing()
                    time.sleep(2)

            self.add_time_taken(time.time() - start_time)

    def add_time_taken(self,time_taken):
        def format_timestamp(timestamp):
            def format_digit(num):
                num=str(num)
                while len(num) < 2:
                    num = '0' + num
                return num
            remainder = timestamp
            hours = int(timestamp // 3600)
            remainder = remainder % 3600
            minutes = int(remainder // 60)
            remainder = remainder % 60
            seconds = int(remainder)
            return f"{format_digit(hours)}:{format_digit(minutes)}:{format_digit(seconds)}"

        self.time_running += time_taken
        self.update_gui('runtime', format_timestamp(self.time_running))

    def make_no_bobber_image(self):
        img = np.ones((256, 256, 3), np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 256 // 2)
        fontScale = 1
        fontColor = (135, 0, 0)
        lineType = 2
        cv2.putText(
            img,
            "No bobber",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        return img

    def stop(self):
        self.running_event.clear()  # Stop the bot

    def save_roi_image(self, img):
        uid = "roi_" + str(time.time()).replace(".", "") + ".png"
        path = os.path.join(self.save_images_dir, "roi", uid)
        img = numpy_img_bgr_to_rgb(img)

        img = Image.fromarray(img)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image_to_save_images(path, img)

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
        save_image_to_save_images(path, img)
        return True


def close_sponsored_session_teamviewer():
    name = "Sponsored session"
    try:
        window = pygetwindow.getWindowsWithTitle(name)[0]
        window.close()
    except:
        return False

    return True


def save_image_to_save_images(path: str, image: Image.Image):
    # make the dir if it doesnt exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # if there more than 1000 images in the folder, delete a random one
    if len(os.listdir(os.path.dirname(path))) > 1000:
        os.remove(
            os.path.join(
                os.path.dirname(path),
                np.random.choice(os.listdir(os.path.dirname(path))),
            )
        )

    # save this image
    image.save(path)


def numpy_img_bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def xywy2xyxy(x, y, w, h):
    return x - w // 2, y - h // 2, x + w // 2, y + h // 2


def run_bot_with_gui():
    root = tk.Tk()
    gui = GUI(root)
    bot = WoWFishBot(gui, save_images=False)
    gui.set_bot(bot)
    root.mainloop()


if __name__ == "__main__":
    run_bot_with_gui()

