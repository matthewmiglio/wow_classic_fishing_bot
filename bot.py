print("Starting python...")
import os
import random
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import pyautogui
import pygetwindow
from numpy import ndarray
from PIL import Image

from _FEATURE_FLAGS import (
    BLACKLIST_FEATURE_FLAG,
    SAVE_IMAGES_FEATURE,
    SAVE_LOGS_FEATURE,
)
from gui import GUI, GUI_WINDOW_NAME
from image_rec import classification_scorer, get_color_frequencies
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier

START_FISHING_COORD = (930, 1400)
FISHING_POLE_COORD = (520, 1330)
DISPLAY_IMAGE_SIZE = 200
WOW_CLIENT_RESIZE = (800, 600)
WOW_WINDOW_NAME = "World of Warcraft"
TOP_SAVE_DIR = "save_images"


def close_sponsored_session_teamviewer():
    name = "Sponsored session"
    try:
        window = pygetwindow.getWindowsWithTitle(name)[0]
        window.close()
    except:
        return False

    return True


def all_pixels_equal(pixels, colors, tol=30):
    for i, pixel in enumerate(pixels):
        if not pixel_is_equal(pixel, colors[i], tol=tol):
            return False
    return True


def pixel_is_equal(p1, p2, tol=30):
    for i, pixel1_value in enumerate(p1):
        if abs(pixel1_value - p2[i]) > tol:
            return False
    return True


def numpy_img_bgr_to_rgb(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"\nError occured in numpy_img_bgr_to_rgb(): {e}" * 50)

    return img


def xywh2xyxy(x, y, w, h):
    return x - w // 2, y - h // 2, x + w // 2, y + h // 2


def run_bot_with_gui():
    print("Initalizing bot and GUI...")
    bot_root = tk.Tk()
    bot_gui = GUI(bot_root)
    bot = WoWFishBot(bot_gui, save_images=True)
    bot_gui.set_bot(bot)
    bot_root.mainloop()


class WoWFishBot:
    MIN_CONFIDENCE_FOR_BOBBER_DETECTION = (
        -1
    )  # determined in the bobber model's non max suppression
    CAST_TIMEOUT = 7  # s
    BOBBER_ROI_IMAGE_RESIZE = 640

    def __init__(self, gui, save_images=False):
        print("Initializing WoWFishBot...")

        # stats
        self.casts = 0
        self.reels = 0
        self.time_of_last_reel = None
        self.time_of_last_cast = None
        self.start_time = time.time()
        self.time_running = 0

        # loot stuff
        self.loot_classifier = LootClassifier()
        self.ignore_blacklist = False

        # gui
        self.running_event = threading.Event()  # Control the bot running state
        self.gui = gui

        if self.gui is not None:
            self.gui_orientation_thread()
            self.gui.update_image(
                self.make_image("  Waiting for start..."), "bobber"
            )  # TODO idfk why its different
            self.update_gui(
                "raw_image",
                self.make_image("  Waiting for start..."),
            )  # TODO idfk why its different
        self.update_loot_history_stats()

        # wow client
        self.wow_orientation_thread()

        # ai models
        self.bobber_detector = BobberDetector(
            r"inference\bobber_models\bobber_finder7.0.onnx"
        )
        self.splash_classifier = SplashClassifier(
            r"inference\splash_models\splash_classifier6.0.onnx"
        )

        # saving images
        self.save_splash_images_toggle = save_images
        self.save_images_dir = TOP_SAVE_DIR
        self.save_images_folder_file_limit = 5000
        self.start_file_clean_thread(
            self.save_images_dir, self.save_images_folder_file_limit
        )

        # logger
        self.logger = Logger()

        # printing
        self.print_mode_enabled = False

        # prediction storage
        self.predictions = []  # used to store the last predictions
        self.splash_prediction_history_limit = 1000

        # vars related to dynamic roi image
        self.dynamic_image_topleft = (0, 0)
        self.dynamic_image_crop_region = (0, 0, 0, 0)  # x1,y1,x2,y2

    # prediction stuff
    def print_predictions(self):
        def format_ts(this_ts, first_ts):
            diff = this_ts - first_ts
            integer, decimals = str(diff).split(".")
            number = str(integer) + "." + str(decimals)[:2]
            return number

        if self.print_mode_enabled is False:
            return

        if len(self.predictions) == 0:
            return
        first_ts = self.predictions[0]["time"]
        for prediction in self.predictions:
            bobber_exists = str(prediction["bobber_exists"])
            bobber_position = str(prediction["bobber_position"])
            is_splash = str(prediction["is_splash"])
            ts = prediction["time"]
            ts = format_ts(ts, first_ts)
            print(
                "bobber?: {:^6} splash?: {:^6} {:^20} ts: {}".format(
                    bobber_exists, is_splash, bobber_position, ts
                )
            )
        print(f"Holding {len(self.predictions)} predictions")

    def last_predictions_equal(
        self,
        prediction_key,
        target_prediction_value,
        history_in_seconds=3,
        occurence_ratio=0.5,
    ) -> bool:
        """
        Example prediction datum:
        {
            "bobber_exists": bobber_exists,
            "bobber_position": bobber_xywh,
            "is_splash": is_splash,
            "time": time.time(),
        }
        """

        # if prediction history is empty, return False
        if len(self.predictions) == 0:
            return False

        # Get the latest timestamp from the last prediction
        last_ts = self.predictions[-1]["time"]
        first_ts = (
            last_ts - history_in_seconds
        )  # Time window starts from `history_in_seconds` ago

        positives = 0
        total = 0

        for prediction in self.predictions:
            if (
                prediction["time"] < first_ts
            ):  # Stop processing if we're outside the time window
                continue

            this_prediction_value = prediction[prediction_key]
            if this_prediction_value == target_prediction_value:
                positives += 1
            total += 1

        # Ensure there were some predictions within the history window
        if total == 0:
            return False

        ratio = positives / total
        return ratio > occurence_ratio

    def add_to_prediction_history(
        self, is_splash: bool, bobber_exists: bool, bobber_xywh: list[float]
    ):
        # if we're missing the bbox, use the previous one
        if bobber_xywh is None and len(self.predictions) > 0:
            bobber_xywh = self.predictions[-1]["bobber_position"]

        # if we're missing the is_splash, use the previous one
        if is_splash is None:
            is_splash = (
                self.predictions[-1]["is_splash"]
                if len(self.predictions) != 0
                else False
            )

        self.predictions.append(
            {
                "bobber_exists": bobber_exists,
                "bobber_position": bobber_xywh,
                "is_splash": is_splash,
                "time": time.time(),
            }
        )

    # wow client stuff
    def focus_wow(self):
        close_sponsored_session_teamviewer()
        name = WOW_WINDOW_NAME

        try:
            window = pygetwindow.getWindowsWithTitle(name)[0]
            window.activate()
        except:
            return False

        return True

    def start_fishing(self):
        self.focus_wow()
        pyautogui.press("z")
        self.casts += 1
        self.update_gui("casts", self.casts)
        self.logger.add_to_fishing_log("attempt")

    def click_bobber_bbox(self, bbox):
        random_deviation = 10
        x1, y1, x2, y2 = bbox
        for _ in range(5):
            center_x = int((x1 + x2) / 2) + (
                random.choice([-1, 1]) * random.randint(0, random_deviation)
            )
            center_y = int((y1 + y2) / 2) + (
                random.choice([-1, 1]) * random.randint(0, random_deviation)
            )
            self.send_delayed_click(center_x, center_y, wait=1)

    def send_delayed_click(self, x, y, wait):
        def _to_wrap(x, y, wait):
            time.sleep(wait)
            pyautogui.click(x, y, button="right")

        t = threading.Thread(target=_to_wrap, args=(x, y, wait))
        t.start()

    # window orientation stuff
    def should_clear_background_threads(self) -> bool:
        # if the bot has been running for less than 20 seconds, dont kill the gui yet
        if time.time() - self.start_time < 10:
            return False

        # if gui window is missing, we should kill things
        try:
            pygetwindow.getWindowsWithTitle(GUI_WINDOW_NAME)[0]
        except:
            return True

        return False

    def gui_orientation_thread(self):
        def valid_position():
            try:
                window = pygetwindow.getWindowsWithTitle(GUI_WINDOW_NAME)[0]
                if window.left == 0 and window.top == 0:
                    return True
            except:
                return False
            return False

        def _to_wrap():
            gui_window_name = GUI_WINDOW_NAME
            while not self.should_clear_background_threads():
                try:
                    if not valid_position():
                        window: pygetwindow.Window = pygetwindow.getWindowsWithTitle(
                            gui_window_name
                        )[0]
                        window.moveTo(0, 0)
                        print("Moved window!")
                    else:
                        self.background_thread_wait(5)
                except Exception as e:
                    print(f"Error moving window: {e}")
                    self.background_thread_wait(3)

            print("Exiting gui_orientation_thread()")

        if self.gui is None:
            return
        t = threading.Thread(target=_to_wrap)
        t.start()

    def wow_orientation_thread(self):
        def valid_position():
            try:
                screen_dims = pyautogui.size()
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                valid_left_position = screen_dims[0] - 815
                if abs(window.left - valid_left_position) > 0:
                    print(
                        f"This window's horizontal position is incorrect. Expected: {valid_left_position} real {window.left}"
                    )
                    return False
                if window.top != 0:
                    print(
                        f"This window's vertical position is incorrect. Expected: {0} real {window.top}"
                    )
                    return False
            except:
                return True

            return True

        def move_wow():
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                screen_dims = pyautogui.size()
                left = screen_dims[0] - WOW_CLIENT_RESIZE[0] - 15
                window.moveTo(left, 0)
            except:
                pass

        def _to_wrap():
            while not self.should_clear_background_threads():
                try:
                    if not valid_position():
                        move_wow()
                        print("Resized wow window!")
                    else:
                        self.background_thread_wait(5)
                except Exception as e:
                    print(f"Error moving wow window: {e}")
                    self.background_thread_wait(3)

            print("Exiting wow_orientation_thread()")

        if self.gui is None:
            return

        t = threading.Thread(target=_to_wrap)
        t.start()

    # gui stuff
    def update_gui(self, stat, value):
        # convert value o rgb
        if self.gui is None:
            return

        if stat == "raw_image":
            try:
                value = cv2.resize(value, (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE))
                self.gui.update_image(value, "raw")
            except:
                pass
        elif stat == "bobber_image":
            try:
                value = cv2.resize(value, (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE))
                self.gui.update_image(value, "bobber")
            except:
                pass
        elif stat == "bobber_detected":
            self.gui.update_stat("bobber_detected", value)
        elif stat == "splash_detected":
            self.gui.update_stat(
                "splash_detected", "Yes" if "splash" in value else "No"
            )
        elif stat == "casts":
            self.gui.update_stat("casts", value)
        elif stat == "reels":
            self.gui.update_stat("reels", value)
        elif stat == "runtime":
            self.gui.update_stat("runtime", value)
        elif stat == "loots":
            self.gui.update_stat("loots", value)

    def update_loot_history_stats(self):
        def example_loot_history():
            example_loot_list = []
            number_of_example_fish = 10
            random_counts = sorted(
                [random.randint(0, 100) for _ in range(number_of_example_fish)]
            )
            for example_fish_index in range(number_of_example_fish):
                for _ in range(random_counts[example_fish_index]):
                    example_loot_list.append(f"Example Fish #{example_fish_index}")

            return example_loot_list

        if self.gui is None:
            return

        loot_history = (
            self.loot_classifier.history
            if self.loot_classifier.history != []
            else example_loot_history()
        )

        self.gui.update_loot_history((loot_history))

    def add_reel(self):
        def should_add_reel():
            last_time = self.time_of_last_reel
            if last_time is None:
                return True
            current_time = time.time()
            threshold = 10  # s
            if current_time - last_time > threshold:
                return True
            return False

        if should_add_reel() is True:
            self.time_of_last_reel = time.time()
            self.reels += 1
            self.logger.add_to_fishing_log("success")
            self.update_gui("reels", self.reels)

    def make_image(self, text):
        bg_color = 0
        img = np.ones((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE, 3), np.uint8) * bg_color
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, DISPLAY_IMAGE_SIZE // 2)
        fontScale = 0.5
        fontColor = (200, 200, 200)
        lineType = 2
        cv2.putText(
            img,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        return img

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

    def background_thread_wait(self, dur):
        start_time = time.time()
        while 1:
            if time.time() - start_time > dur:
                return
            if self.should_clear_background_threads():
                return
            time.sleep(1)

    # file stuff
    def start_file_clean_thread(self, top_dir, file_limit):
        def clean_excess_pngs():
            for root, dirs, files in os.walk(top_dir):
                # Count the number of PNG files directly within the current folder
                png_files = [f for f in files if f.lower().endswith(".png")]
                current_count = len(png_files)

                # Check if the current count exceeds the file limit
                if current_count > file_limit:
                    # Calculate how many files need to be deleted
                    excess_count = current_count - file_limit
                    files_to_delete = random.sample(png_files, excess_count)

                    # Delete the selected files
                    for file_to_delete in files_to_delete:
                        file_path = os.path.join(root, file_to_delete)
                        os.remove(file_path)
                        # print(f"Deleted: {file_path}")

                # If no more PNGs are in excess, we can break early
                if current_count <= file_limit:
                    continue

        def _to_wrap():
            while not self.should_clear_background_threads():
                clean_excess_pngs()
                self.background_thread_wait(120)
            print("Exiting file_clean_thread()")

        t = threading.Thread(target=_to_wrap)
        t.start()

    def save_bobber_roi_image(self, img):
        if SAVE_IMAGES_FEATURE is not True:
            return

        uid = "roi_" + str(time.time()).replace(".", "") + ".png"
        path = os.path.join(self.save_images_dir, "bobber_roi", uid)

        img = Image.fromarray(img)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.save_image_to_save_images(path, img)

    def save_splash_images(self, bbox, img, is_splash) -> bool:
        if SAVE_IMAGES_FEATURE is not True:
            return

        uid = str(time.time()).replace(".", "") + ".png"
        path = os.path.join(
            self.save_images_dir,
            "splash" if "splash" in is_splash else "not_splash",
            uid,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # convert it from BGR to RGB
        img = numpy_img_bgr_to_rgb(img)
        img = Image.fromarray(img)
        bbox = xywh2xyxy(*bbox)
        img = img.crop(bbox)
        self.save_image_to_save_images(path, img)
        return True

    def save_image_to_save_images(self, path: str, image: Image.Image):
        if SAVE_IMAGES_FEATURE is not True:
            return

        # make the dir if it doesnt exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # if there more than 1000 images in the folder, delete a random one
        if len(os.listdir(os.path.dirname(path))) > self.save_images_folder_file_limit:
            os.remove(
                os.path.join(
                    os.path.dirname(path),
                    np.random.choice(os.listdir(os.path.dirname(path))),
                )
            )

        # save this image
        image.save(path)

    # image stuff
    def get_roi_image(self):
        # get the wow image
        window_name = WOW_WINDOW_NAME
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
        wow_image = wow_image.crop(this_crop).resize((640, 640))

        # convert to numpy array
        wow_image = np.array(wow_image)
        return wow_image

    def make_bobber_image(self, bbox, img):
        xywh = bbox
        x_pad, y_pad = 10, 10
        x1 = (xywh[0] - xywh[2] // 2) - x_pad
        y1 = (xywh[1] - xywh[3] // 2) - y_pad
        x2 = (xywh[0] + xywh[2] // 2) + x_pad
        y2 = (xywh[1] + xywh[3] // 2) + y_pad
        bbox_image = img[y1:y2, x1:x2]
        return bbox_image

    def convert_bbox_to_usable(self, xywh_bbox):
        # Step 1: Unstretch the bounding box
        # Convert the center back to the crop coordinates
        scale_x = self.BOBBER_ROI_IMAGE_RESIZE / (
            self.dynamic_image_crop_region[2] - self.dynamic_image_crop_region[0]
        )
        scale_y = self.BOBBER_ROI_IMAGE_RESIZE / (
            self.dynamic_image_crop_region[3] - self.dynamic_image_crop_region[1]
        )

        center_x, center_y, width, height = xywh_bbox

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

    def get_wow_image(self):
        # get the wow image
        window_name = WOW_WINDOW_NAME
        window = pygetwindow.getWindowsWithTitle(window_name)[0]
        self.dynamic_image_topleft = (window.left, window.top)
        region = (window.left, window.top, window.width, window.height)
        wow_image = pyautogui.screenshot(region=region)
        wow_image_np = np.array(wow_image)
        return wow_image_np

    # settings stuff
    def set_blacklist(self):
        self.loot_classifier.blacklist_loot = self.gui.get_blacklist_settings()

    # bot stuff
    def stop(self):
        self.running_event.clear()  # Stop the bot

    def run(self):
        self.running_event.set()  # Start the event
        print(
            f"user just clicked run.\nThe blacklist enabled checkbox value is {self.gui.blacklist_mode_toggle_input.get()}"
        )
        if int(self.gui.blacklist_mode_toggle_input.get()) == 1:
            self.ignore_blacklist = True
        else:
            self.ignore_blacklist = False
        print(f"Thus, the ignore_blacklist value is {self.ignore_blacklist}")

        # main loop
        while (
            self.running_event.is_set() and not self.should_clear_background_threads()
        ):
            start_time = time.time()
            base_image = self.get_roi_image()

            if self.save_splash_images_toggle:
                self.save_bobber_roi_image(base_image)

            self.update_gui("raw_image", base_image)

            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # if a bobber detected
            if score >= self.MIN_CONFIDENCE_FOR_BOBBER_DETECTION and bbox != []:
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

                # show the result in the guiz
                self.update_gui("bobber_detected", "Yes")
                self.update_gui("splash_detected", is_splash)

                # if the bobber is a splash, we click it to reel in the fish
                if "splash" in is_splash:
                    is_splash_bool = True
                else:
                    is_splash_bool = False

                self.add_to_prediction_history(
                    is_splash_bool, bobber_exists=True, bobber_xywh=bbox
                )
                self.print_predictions()
                if self.last_predictions_equal(
                    "is_splash",
                    True,
                    history_in_seconds=1,
                    occurence_ratio=0.3,
                ):
                    print("Splash detected! Reeling in fish!")
                    self.add_reel()
                    bbox = self.convert_bbox_to_usable(bbox)
                    self.click_bobber_bbox(bbox)
                    self.set_blacklist()

                    # if blacklist feature is off,  or untoggled by user
                    # we assume autoloot is on, so skip collect_loot()
                    if (
                        BLACKLIST_FEATURE_FLAG is not True
                        or self.ignore_blacklist is True
                    ):
                        print(
                            f"Skipping loot collection step because BLACKLIST_FEATURE_FLAG={BLACKLIST_FEATURE_FLAG} and WoWFishBot.ignore_blacklist={WoWFishBot.ignore_blacklist}"
                        )
                        continue

                    loot = self.loot_classifier.collect_loot()
                    if loot:
                        self.update_gui(
                            stat="loots", value=len(self.loot_classifier.history)
                        )
                        self.logger.add_to_loot_log(loot)
                        self.update_loot_history_stats()

                # if we want to save the splash images
                if self.save_splash_images_toggle:
                    self.save_splash_images(bbox, base_image, is_splash)

            # if none detected at all
            else:
                self.gui.update_stat("bobber_detected", "No")
                self.gui.update_image(self.make_image("No bobber"), "bobber")
                self.gui.update_image(self.get_roi_image(), "raw_image")
                self.gui.update_stat("splash_detected", "No")
                self.add_to_prediction_history(
                    None, bobber_exists=False, bobber_xywh=None
                )
                if self.last_predictions_equal(
                    "bobber_exists",
                    False,
                    history_in_seconds=3,
                    occurence_ratio=0.5,
                ):
                    if (self.time_of_last_cast is None) or (
                        time.time() - self.time_of_last_cast > self.CAST_TIMEOUT
                    ):
                        print("-" * 50 + "\n" + "Starting fishing...")
                        self.start_fishing()
                        self.time_of_last_cast = time.time()
                        self.predictions = []  # clear the predictions for new fishing session
                    else:
                        print("Bobber not found, but just casted...")

            self.add_time_taken(time.time() - start_time)

        print("Exiting wfb.run()")


class Logger:
    def __init__(self):
        print("Initializing Logger...")
        self.logs_folder = r"logs"
        self.fishing_attempts_log = os.path.join(
            self.logs_folder, "fishing_attempts.txt"
        )
        self.loot_log = os.path.join(self.logs_folder, "loot_log.txt")
        self.init_log_files()

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


class LootClassifier:
    POSITIVE_DETECTION_THRESHOLD = 0.99

    def __init__(self):
        print("Initializing LootClassifier...")
        self.blacklist_loot = []  # let gui set this
        self.history: list[str] = []

    def wait_for_loot_window(self):
        wait_timeout = 3  # s
        wait_start_time = time.time()
        while time.time() - wait_start_time < wait_timeout:
            if self.loot_window_exists():
                return True
            time.sleep(0.1)
        return False

    def collect_loot(self) -> bool:
        def calc_loot_coord():
            wow_window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
            coord = (wow_window.left + 44, wow_window.top + 196)
            return coord

        def calc_close_loot_coord():
            wow_window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
            coord = (wow_window.left + 144, wow_window.top + 129)
            return coord

        # wait for loot window to appear
        if self.wait_for_loot_window() is False:
            print("Loot window did not appear in time.")
            return False

        # classify that loot
        loot_classification, score = self.classify_loot()
        if score < self.POSITIVE_DETECTION_THRESHOLD:
            print("This is a low score. is this a new loot type?")
            # loot_classification = f"unknown_{score}_" + loot_classification
            loot_classification = f"unknown_fish_{random.randint(100,999)}"

        self.history.append(loot_classification)

        # if its blacklist loot, close the loot window to skip it
        close_loot_coords = calc_close_loot_coord()
        if loot_classification in self.blacklist_loot:
            print(f"{loot_classification} is blacklisted! Skipping it...")
            pyautogui.click(*close_loot_coords, clicks=3, interval=0.5)

        # else click the loot to collect it
        else:
            print(f"{loot_classification} is whitelisted! Collecting it...")
            collect_loot_coords = calc_loot_coord()
            pyautogui.click(
                *collect_loot_coords, button="right", clicks=3, interval=0.3
            )

        return loot_classification

    def loot_window_exists(self):
        # grab wow image
        wow_window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
        wow_image_region = [
            wow_window.left,
            wow_window.top,
            wow_window.width,
            wow_window.height,
        ]
        wow_image = pyautogui.screenshot(region=wow_image_region)
        wow_image = np.array(wow_image)

        pixels = [
            wow_image[138][23],
            wow_image[121][38],
            wow_image[139][56],
            wow_image[155][45],
            wow_image[132][34],
            wow_image[156][36],
        ]
        colors = [
            "black",
            "black",
            "black",
            "black",
            "white",
            "white",
        ]

        def is_white(pixel):
            return pixel[0] > 100 and pixel[1] > 100 and pixel[2] > 100

        def is_black(pixel):
            return pixel[0] < 50 and pixel[1] < 50 and pixel[2] < 50

        for i, pixel in enumerate(pixels):
            color = colors[i]
            # print(pixel,color)
            if color == "white" and not is_white(pixel):
                return False
            elif color == "black" and not is_black(pixel):
                return False

        return True

    def loot_window_exists_old(self):
        image = np.asarray(pyautogui.screenshot())
        pixels = [
            image[112][938],
            image[103][951],
            image[119][967],
            image[128][958],
            image[124][941],
            image[112][949],
            image[111][957],
            image[131][948],
            image[112][958],
        ]

        colors = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [234, 233, 223],
            [167, 166, 161],
            [193, 192, 186],
            [151, 149, 143],
        ]
        return all_pixels_equal(pixels, colors)

    def save_unknown_loot_image(self, image, dict):
        def make_new_text_file(fp, text):
            with open(fp, "w") as f:
                f.write(text)

        top_save_dir = os.path.join(TOP_SAVE_DIR, "unknown_loot")
        this_uid = str(time.time()).replace(".", "")
        this_export_dir = os.path.join(top_save_dir, this_uid)
        os.makedirs(TOP_SAVE_DIR, exist_ok=True)
        os.makedirs(top_save_dir, exist_ok=True)
        os.makedirs(this_export_dir, exist_ok=True)

        this_text_file_path = os.path.join(this_export_dir, "color_dict.txt")
        this_image_file_path = os.path.join(this_export_dir, "loot_image.png")

        dict = str(dict).strip()
        make_new_text_file(this_text_file_path, dict)
        image = Image.fromarray(image)
        image.save(this_image_file_path)

    def classify_loot(self):
        def loot_colors_to_printable(loot_colors):
            # get a list of values
            loot_colors = list(loot_colors.values())
            return loot_colors

        def get_highest_score(scores):
            scores = {
                k: v
                for k, v in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )
            }
            return list(scores.keys())[0], scores[list(scores.keys())[0]]

        loot_image = self.get_loot_image()

        class_scores = classification_scorer(loot_image)
        # print("\nRaw class scores\n", class_scores)

        best_label, best_score = get_highest_score(class_scores)

        if best_score < self.POSITIVE_DETECTION_THRESHOLD:
            color_dict = get_color_frequencies(loot_image)
            self.save_unknown_loot_image(loot_image, color_dict)

        print("Loot:", best_label, str(float(best_score) * 100).split(".")[0] + "%")

        return (best_label, best_score)

    def get_loot_image(self) -> ndarray:
        # grab wow image
        wow_window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
        wow_image_region = [
            wow_window.left,
            wow_window.top,
            wow_window.width,
            wow_window.height,
        ]
        wow_image = pyautogui.screenshot(region=wow_image_region)
        wow_image = np.array(wow_image)

        # plt.imshow(wow_image)
        # plt.show()

        # crop to the little loot image
        loot_image = wow_image[175:203, 28:55]
        return loot_image

    def get_loot_image_old(self) -> ndarray:
        image = pyautogui.screenshot(region=[944, 147, 20, 20])
        image = np.array(image)
        return image


if __name__ == "__main__":
    run_bot_with_gui()

    # wfb = WoWFishBot(None, save_images=False)
    # roi = wfb.get_roi_image()
