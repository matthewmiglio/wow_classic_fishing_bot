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
FISHING_POLE_COORD = (520, 1330)
BUFF_INCREMENT = 11  # minutes


class WoWFishBot:
    MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25

    def __init__(self, gui, save_images=False, apply_buff=False):
        # gui
        self.gui = gui
        self.running_event = threading.Event()  # Control the bot running state

        # ai models
        self.bobber_detector = BobberDetector(
            r"inference\bobber_models\bobber_finder3.0.onnx"
        )
        self.splash_classifier = SplashClassifier(
            r"inference\splash_models\splash_classifier4.0.onnx"
        )  # test this. otherwise use 1.0

        # saving images
        self.save_splash_images_toggle = save_images
        self.save_images_dir = "save_images"

        # user toggles
        self.apply_buff = apply_buff

        # stats
        self.casts = 0
        self.reels = 0
        self.buffs = 0
        self.time_running = 0

        # prediction storage
        self.prediction_history = []  # used to store the last 8 predictions
        self.splash_prediction_history_limit = 100

        # vars related to dynamic roi image
        self.dynamic_image_topleft = (0, 0)
        self.dynamic_image_crop_region = (0, 0, 0, 0)  # x1,y1,x2,y2
        self.stretched_size = (256, 256)

    def should_buff(self):
        minutes_ran = self.time_running / 60

        target_count = (minutes_ran // BUFF_INCREMENT) + 1

        if target_count > self.buffs:
            self.buffs += 1
            return True

        return False

    def apply_fishing_buff(self):
        pyautogui.click(*FISHING_BUFF_COORD, clicks=2, interval=0.22)
        time.sleep(0.1)
        pyautogui.click(*FISHING_POLE_COORD, clicks=2, interval=0.22)
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

    def last_predictions_equal(
        self, prediction, mind_prediction_count, prediction_history_length=12
    ):
        # most recent prediction should be the target prediction
        if prediction not in self.prediction_history[-1]:
            return False

        # count up the number of times the target prediction has been made in the recent history_length count
        cut_prediction_history = self.prediction_history[-prediction_history_length:]
        count = 0
        for prediction in cut_prediction_history:
            if prediction in prediction:
                count += 1

        # if the count is greater than the equal_count, return True
        if count > mind_prediction_count:
            self.prediction_history = []
            return True

        return False

    def get_roi_image(self):
        # get the wow image
        window_name = "World of Warcraft"
        window = pygetwindow.getWindowsWithTitle(window_name)[0]
        self.dynamic_image_topleft = (window.left, window.top)
        region = (window.left, window.top, window.width, window.height)
        wow_image = pyautogui.screenshot(region=region)

        # calculate the crop
        crop_ratios = (
            0.3645833333333333,
            0.06944444444444445,
            0.6770833333333334,
            0.625,
        )

        this_crop = (
            crop_ratios[0] * wow_image.width,
            crop_ratios[1] * wow_image.height,
            crop_ratios[2] * wow_image.width,
            crop_ratios[3] * wow_image.height,
        )
        self.dynamic_image_crop_region = this_crop

        # crop the image
        wow_image = wow_image.crop(this_crop).resize((256, 256))

        # convert to numpy array
        wow_image = np.array(wow_image)
        return wow_image

    def convert_bbox_to_usable(self, bbox):
        # Extract the bounding box components
        center_x, center_y, width, height = bbox

        # Step 1: Unstretch the bounding box
        # Convert the center back to the crop coordinates
        scale_x = self.stretched_size[0] / (
            self.dynamic_image_crop_region[2] - self.dynamic_image_crop_region[0]
        )
        scale_y = self.stretched_size[1] / (
            self.dynamic_image_crop_region[3] - self.dynamic_image_crop_region[1]
        )

        # Convert to crop coordinates
        crop_center_x = center_x / scale_x
        crop_center_y = center_y / scale_y
        crop_width = width / scale_x
        crop_height = height / scale_y

        # Step 2: Uncrop to original window coordinates
        # The crop coordinates are relative to the crop region
        original_x = crop_center_x - crop_width / 2
        original_y = crop_center_y - crop_height / 2

        # Now adjust for the original window position
        original_x += self.dynamic_image_crop_region[0] + self.dynamic_image_topleft[0]
        original_y += self.dynamic_image_crop_region[1] + self.dynamic_image_topleft[1]

        # Return the usable bounding box in the format [x1, y1, x2, y2]
        x1 = original_x
        y1 = original_y
        x2 = original_x + crop_width
        y2 = original_y + crop_height

        return (x1, y1, x2, y2)

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

        self.focus_wow()
        pyautogui.press("z")
        self.casts += 1
        self.update_gui("casts", self.casts)

    def click_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        self.send_delayed_click(center_x, center_y, wait=1)

    def send_delayed_click(self, x, y, wait):
        def _to_wrap(x, y, wait):
            time.sleep(wait)
            pyautogui.click(x, y, button="right")

        t = threading.Thread(target=_to_wrap, args=(x, y, wait))
        t.start()

    def update_gui(self, stat, value):
        if "image" in stat:
            # convert value to rgb
            value = numpy_img_bgr_to_rgb(value)

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
        if stat == "runtime":
            self.gui.update_stat("runtime", value)

    def run(self):
        self.running_event.set()  # Start the event

        while self.running_event.is_set():
            start_time = time.time()
            base_image = self.get_roi_image()

            if self.save_splash_images_toggle:
                self.save_roi_image(base_image)

            self.update_gui("raw_image", base_image)

            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # if a bobber detected
            if score > self.MIN_CONFIDENCE_FOR_BOBBER_DETECTION:
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
                if self.last_predictions_equal(
                    prediction="splash",
                    mind_prediction_count=6,
                    prediction_history_length=12,
                ):
                    print(
                        "Splash detected! Reeling in fish...\nSelf.splash_prediction_history: ",
                        self.prediction_history,
                    )
                    self.reels += 1
                    self.update_gui("reels", self.reels)
                    bbox = self.convert_bbox_to_usable(bbox)
                    self.click_bbox(bbox)

                # if we want to save the splash images
                if self.save_splash_images_toggle:
                    self.save_splash_images(bbox, base_image, is_splash)

            # if none detected at all
            else:
                self.gui.update_stat("bobber_detected", "No")
                self.gui.update_image(self.make_no_bobber_image(), "bobber")
                self.gui.update_image(self.get_roi_image(), "raw_image")
                self.gui.update_stat("splash_detected", "No")
                self.add_splash_prediction("none")
                if self.last_predictions_equal(
                    prediction="none",
                    mind_prediction_count=20,
                    prediction_history_length=50,
                ):
                    print('20 of the last 50 predictions were "none".')
                    print("Starting fishing...")
                    self.start_fishing()

            self.add_time_taken(time.time() - start_time)

    def add_time_taken(self, time_taken):
        def format_timestamp(timestamp):
            def format_digit(num):
                num = str(num)
                while len(num) < 2:
                    num = "0" + num
                return num

            remainder = timestamp
            hours = int(timestamp // 3600)
            remainder = remainder % 3600
            minutes = int(remainder // 60)
            remainder = remainder % 60
            seconds = int(remainder)
            return (
                f"{format_digit(hours)}:{format_digit(minutes)}:{format_digit(seconds)}"
            )

        self.time_running += time_taken
        self.update_gui("runtime", format_timestamp(self.time_running))

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
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f'\nError occured in numpy_img_bgr_to_rgb(): {e}'*50)

    return img


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
