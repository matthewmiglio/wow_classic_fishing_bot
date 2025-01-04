print("Starting python...")

from _FEATURE_FLAGS import (
    BLACKLIST_FEATURE_FLAG,
    CLOUD_STATS_FEATURE,
    SAVE_IMAGES_FEATURE,
)
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier
from cloud.supa import UsersTable, StatsTable, UsageTable
from logger import Logger
from looter import LootClassifier
from constants import WOW_WINDOW_NAME, TOP_SAVE_DIR, WOW_CLIENT_RESIZE, DISPLAY_IMAGE_SIZE

import os
import random
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import pyautogui
import pygetwindow
from PIL import Image


from debug import collect_all_system_info, get_folder_size
from gui import GUI, GUI_WINDOW_NAME
from image_rec import classification_scorer, get_color_frequencies





def show_popup(display_text, title_text):
    window_width = 150
    text_wrap_len = window_width - 20
    line_count = 1 + (len(display_text) // text_wrap_len)
    window_height = 90 + (line_count * 20)

    # Create a new tkinter window
    root = tk.Tk()
    root.title(title_text)

    root.overrideredirect(True)  # This removes the window title bar

    # Set up the label with the text
    label = tk.Label(
        root, text=display_text, padx=10, pady=10, wraplength=text_wrap_len
    )
    label.pack()

    # Variable to store the result
    result = None

    # Function to handle the OK button click event
    def on_ok():
        nonlocal result
        result = True
        root.destroy()  # Close the popup

    # Function to handle the Cancel button click event
    def on_cancel():
        nonlocal result
        result = False
        root.destroy()  # Close the popup

    # Create the OK and Cancel buttons
    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.pack(side="left", padx=10, pady=10)

    cancel_button = tk.Button(
        root, text="Cancel", command=on_cancel, bg="red", fg="white"
    )
    cancel_button.pack(side="right", padx=10, pady=10)

    # Center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    # Start the Tkinter event loop and wait for the user interaction
    root.mainloop()

    return result


def close_sponsored_session_teamviewer():
    name = "Sponsored session"
    try:
        window = pygetwindow.getWindowsWithTitle(name)[0]
        window.close()
    except:
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


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
        self.logger: Logger = Logger()

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
            except Exception as e:
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
                foreground_wow()
            except:
                pass

        def foreground_wow():
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                window.activate()
            except:
                pass

        def _to_wrap():
            foreground_wow()

            while not self.should_clear_background_threads():
                try:
                    if not valid_position():
                        move_wow()
                        print("Resized wow window!")
                    else:
                        self.background_thread_wait(5)
                        foreground_wow()
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
        try:
            window = pygetwindow.getWindowsWithTitle(window_name)[0]
        except:
            print('You need to have WoW open to get a ROI image!')
            return False
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

    def run_debug_mode(self):
        def make_uid():
            return str(time.time()).replace(".", "") + random.choice(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )

        def _to_wrap():
            if (
                show_popup("Are you sure you want to run debug mode?", "Debug Mode")
                is False
            ):
                return

            # get appdata dir
            appdata_dir = os.getenv("APPDATA")
            date_readable = time.strftime("%Y_%m_%d_%H_%M_%S")
            self.debug_data_export_folder = os.path.join(
                appdata_dir,
                "WoWFishBot",
                "debug",
                date_readable + random.choice("ABCDEF"),
            )
            os.makedirs(self.debug_data_export_folder, exist_ok=True)
            os.makedirs(
                os.path.join(self.debug_data_export_folder, "predictions"),
                exist_ok=True,
            )

            # write system info as a txt
            system_info = collect_all_system_info()
            system_info_export_path = os.path.join(
                self.debug_data_export_folder, "system_info.txt"
            )
            with open(system_info_export_path, "w") as f:
                f.write(system_info)

            # stats
            valid_predicts = 0
            invalid_predicts = 0

            # for 10 seconds, test the bot, saving related info
            start_time = time.time()
            debug_duration = 60  # s
            casts = 0
            cast_every = 20  # s
            print("Running debug mode for {debug_duration} seconds...")
            while time.time() - start_time < debug_duration:
                # print
                time_left = int(debug_duration - (time.time() - start_time))
                print(f"DEBUG MODE: {time_left}s until complete...")

                # handle casting logic
                time_of_next_cast = casts * cast_every
                time_taken = time.time() - start_time
                if time_taken > time_of_next_cast:
                    self.start_fishing()
                    casts += 1
                    time.sleep(3)

                this_uid = make_uid()

                # grab a raw image of the entire desktop
                screen_image = pyautogui.screenshot()
                screen_image = np.array(screen_image)
                screen_image_save_path = os.path.join(
                    self.debug_data_export_folder,
                    "predictions",
                    this_uid + "_screen_image" + ".png",
                )
                screen_image = bgr2rgb(screen_image)
                cv2.imwrite(screen_image_save_path, screen_image)

                # grab roi image, save it
                roi_image = self.get_roi_image()
                if roi_image is not False:
                    roi_image_save_path = os.path.join(
                        self.debug_data_export_folder,
                        "predictions",
                        this_uid + "_roi_image" + ".png",
                    )
                    roi_image = bgr2rgb(roi_image)
                    cv2.imwrite(roi_image_save_path, roi_image)

                # predict, write prediction
                bbox, score = self.bobber_detector.detect_object_in_image(
                    roi_image, draw_result=False
                )
                annotation_line = f"{time.time()} {bbox} {score}"
                annotation_file_path = os.path.join(
                    self.debug_data_export_folder,
                    "predictions",
                    this_uid + "_prediction" + ".txt",
                )
                with open(annotation_file_path, "w") as f:
                    f.write(annotation_line)

                # save bobber image and prediction if its good
                if bbox != []:
                    bobber_image_save_path = os.path.join(
                        self.debug_data_export_folder,
                        "predictions",
                        this_uid + "_bobber_image" + ".png",
                    )
                    try:
                        bobber_image = self.make_bobber_image(bbox, roi_image)
                        bobber_image = bgr2rgb(bobber_image)
                        cv2.imwrite(bobber_image_save_path, bobber_image)
                    except:
                        pass
                    valid_predicts += 1

                else:
                    invalid_predicts += 1

                # save the whole wow image
                wow_image = self.get_wow_image()
                wow_image_save_path = os.path.join(
                    self.debug_data_export_folder,
                    "predictions",
                    this_uid + "_wow_image" + ".png",
                )
                wow_image = bgr2rgb(wow_image)
                cv2.imwrite(wow_image_save_path, wow_image)

            # save the results
            total_predicts = valid_predicts + invalid_predicts
            valid_percent = round((valid_predicts / total_predicts) * 100, 2)
            invalid_percent = round((invalid_predicts / total_predicts) * 100, 2)
            results_string = f"total casts {casts}"
            results_string += f"\ntotal predictions {total_predicts}"
            results_string += f"\nvalid predictions {valid_predicts}"
            results_string += f"\nvalid percent {valid_percent}"
            results_string += f"\ninvalid predictions {invalid_predicts}"
            results_string += f"\ninvalid percent {invalid_percent}"
            results_string += f"\nduration {debug_duration}"
            results_save_path = os.path.join(
                self.debug_data_export_folder, "results.txt"
            )
            with open(results_save_path, "w") as f:
                f.write(results_string)

            print(
                f"Created a debug folder of size {get_folder_size(self.debug_data_export_folder)} at {self.debug_data_export_folder}"
            )

        threading.Thread(target=_to_wrap).start()

    def run(self):
        self.running_event.set()  # Start the event

        if CLOUD_STATS_FEATURE is not False:
            self.logger.usage_table.increment_uses()
            self.logger.usage_table.set_last_use_time()

        self.ignore_blacklist = True if self.gui.blacklist_mode_toggle_input.get() == 1 else False

        # main loop
        while (
            self.running_event.is_set() and not self.should_clear_background_threads()
        ):
            start_time = time.time()
            base_image = self.get_roi_image()
            if base_image is False:
                print('Couldnt capture an image from WoW. Retrying in 10 seconds...')
                time.sleep(10)
                continue

            if self.save_splash_images_toggle:
                self.save_bobber_roi_image(base_image)

            self.update_gui("raw_image", base_image)

            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # update cloud stats
            # if self.logger.should_cloud_update():
            #     self.logger.stats_table.add_stats(
            #         runtime=self.time_running,
            #         reels=self.reels,
            #         casts=self.casts,
            #         loots=len(self.loot_classifier.history),
            #     )

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
                    bbox = self.convert_bbox_to_usable(bbox)
                    self.click_bobber_bbox(bbox)
                    self.set_blacklist()
                    self.add_reel()

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
                        # update gui
                        self.update_gui(
                            stat="loots", value=len(self.loot_classifier.history)
                        )

                        # update logger
                        self.logger.add_to_loot_log(loot)

                        # update loot history
                        self.update_loot_history_stats()

                # if we want to save the splash images
                if self.save_splash_images_toggle:
                    self.save_splash_images(bbox, base_image, is_splash)

            # if none detected at all
            else:
                self.gui.update_stat("bobber_detected", "No")
                self.gui.update_image(self.make_image("No bobber"), "bobber")
                this_roi_image = self.get_roi_image()
                if this_roi_image is not False:
                    self.gui.update_image(this_roi_image, "raw_image")
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
                        # print("Bobber not found, but just casted...")
                        pass

            self.add_time_taken(time.time() - start_time)

        print("Exiting wfb.run()")


if __name__ == "__main__":
    run_bot_with_gui()

    # wfb = WoWFishBot(None, save_images=False)
    # roi = wfb.get_roi_image()
