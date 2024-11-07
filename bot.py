from PIL import Image
import threading
import pyautogui
import numpy as np
from numpy import ndarray
from collections import defaultdict
import random
import time
import pygetwindow
import os
import tkinter as tk
import cv2
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier
from gui import GUI, GUI_WINDOW_NAME
from constants import LOOT_COLOR_DATA

START_FISHING_COORD = (930, 1400)
FISHING_POLE_COORD = (520, 1330)
DISPLAY_IMAGE_SIZE = 100
WOW_CLIENT_RESIZE = (1000, 700)
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


def xywy2xyxy(x, y, w, h):
    return x - w // 2, y - h // 2, x + w // 2, y + h // 2


def run_bot_with_gui():
    bot_root = tk.Tk()
    bot_gui = GUI(bot_root)
    bot = WoWFishBot(bot_gui, save_images=True)
    bot_gui.set_bot(bot)
    bot_root.mainloop()


class WoWFishBot:
    MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25

    def __init__(self, gui, save_images=False):
        # gui
        self.gui = gui
        self.running_event = threading.Event()  # Control the bot running state
        self.gui_orientation_thread()

        # wow client
        self.wow_orientation_thread()

        # loot detection module
        self.loot_classifier = LootClassifier()

        # ai models
        self.bobber_detector = BobberDetector(
            r"inference\bobber_models\bobber_finder3.0.onnx"
        )
        self.splash_classifier = SplashClassifier(
            r"inference\splash_models\splash_classifier4.0.onnx"
        )  # test this. otherwise use 1.0

        # saving images
        self.save_splash_images_toggle = save_images
        self.save_images_dir = TOP_SAVE_DIR
        self.save_images_folder_file_limit = 5000
        self.start_file_clean_thread(
            self.save_images_dir, self.save_images_folder_file_limit
        )

        # stats
        self.casts = 0
        self.reels = 0
        self.time_of_last_reel = None
        self.time_running = 0

        # prediction storage
        self.prediction_history = []  # used to store the last 8 predictionsz
        self.splash_prediction_history_limit = 100

        # vars related to dynamic roi image
        self.dynamic_image_topleft = (0, 0)
        self.dynamic_image_crop_region = (0, 0, 0, 0)  # x1,y1,x2,y2
        self.stretched_size = (256, 256)

    # prediction stuff
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
        self, prediction, prediction_count, prediction_history_length=12
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
        if count > prediction_count:
            self.prediction_history = []
            return True

        return False

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

    def click_bobber_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        for _ in range(5):
            center_x = int((x1 + x2) / 2) + (
                random.choice([-1, 1]) * random.randint(0, 10)
            )
            center_y = int((y1 + y2) / 2) + (
                random.choice([-1, 1]) * random.randint(0, 10)
            )
            self.send_delayed_click(center_x, center_y, wait=1)

    def wow_orientation_thread(self):
        def valid_position():
            position_tol = 3
            screen_dims = pyautogui.size()
            left = screen_dims[0] - WOW_CLIENT_RESIZE[0]
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                if (
                    abs(window.left - left) < position_tol
                    and abs(window.top - 0) < position_tol
                ):
                    return True
            except:
                return False

        def valid_size():
            tol = 5
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                if (
                    abs(window.width - WOW_CLIENT_RESIZE[0]) < tol
                    and abs(window.height - WOW_CLIENT_RESIZE[1]) < tol
                ):
                    return True
            except:
                return False

        def resize_wow():
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                window.resizeTo(WOW_CLIENT_RESIZE[0], WOW_CLIENT_RESIZE[1])
            except:
                pass

        def move_wow():
            try:
                window = pygetwindow.getWindowsWithTitle(WOW_WINDOW_NAME)[0]
                screen_dims = pyautogui.size()
                left = screen_dims[0] - WOW_CLIENT_RESIZE[0]
                window.moveTo(left, 0)
            except:
                pass

        def _to_wrap():
            while 1:
                try:
                    if not valid_size():
                        resize_wow()
                        print("Moved window!")
                    elif not valid_position():
                        move_wow()
                        print("Resized window!")
                    else:
                        time.sleep(2)
                except Exception as e:
                    print(f"Error moving window: {e}")
                    time.sleep(10)
                    pass

        if self.gui is None:
            return
        t = threading.Thread(target=_to_wrap)
        t.start()

    def send_delayed_click(self, x, y, wait):
        def _to_wrap(x, y, wait):
            time.sleep(wait)
            pyautogui.click(x, y, button="right")

        t = threading.Thread(target=_to_wrap, args=(x, y, wait))
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
        if stat == "bobber_image":
            try:
                value = cv2.resize(value, (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE))
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

        if stat == "runtime":
            self.gui.update_stat("runtime", value)

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
            self.update_gui("reels", self.reels)

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
            while 1:
                try:
                    if not valid_position():
                        window: pygetwindow.Window = pygetwindow.getWindowsWithTitle(
                            gui_window_name
                        )[0]
                        window.moveTo(0, 0)
                        print("Moved window!")
                    else:
                        time.sleep(2)
                except Exception as e:
                    print(f"Error moving window: {e}")
                    time.sleep(10)
                    pass

        if self.gui is None:
            return
        t = threading.Thread(target=_to_wrap)
        t.start()

    def make_no_bobber_image(self):
        img = np.ones((DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE, 3), np.uint8) * 155
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, DISPLAY_IMAGE_SIZE // 2)
        fontScale = 0.5
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
            while 1:
                clean_excess_pngs()
                time.sleep(120)

        t = threading.Thread(target=_to_wrap)
        t.start()

    def save_bobber_roi_image(self, img):
        uid = "roi_" + str(time.time()).replace(".", "") + ".png"
        path = os.path.join(self.save_images_dir, "bobber_roi", uid)

        img = Image.fromarray(img)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.save_image_to_save_images(path, img)

    def save_splash_images(self, bbox, img, is_splash) -> bool:
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
        bbox = xywy2xyxy(*bbox)
        img = img.crop(bbox)
        self.save_image_to_save_images(path, img)
        return True

    def save_image_to_save_images(self, path: str, image: Image.Image):
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
        wow_image = wow_image.crop(this_crop).resize((256, 256))

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

    # settings stuff
    def set_blacklist(self):
        self.loot_classifier.blacklist_loot = self.gui.get_blacklist_settings()

    # bot stuff
    def stop(self):
        self.running_event.clear()  # Stop the bot

    def run(self):
        self.running_event.set()  # Start the event

        # main loop
        while self.running_event.is_set():
            start_time = time.time()
            base_image = self.get_roi_image()

            if self.save_splash_images_toggle:
                self.save_bobber_roi_image(base_image)

            self.update_gui("raw_image", base_image)

            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # if loot window exists, handle that
            if self.loot_classifier.loot_window_exists():
                self.loot_classifier.collect_loot()

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
                    prediction_count=6,
                    prediction_history_length=12,
                ):
                    print("Splash detected! Reeling in fish!")
                    self.add_reel()
                    bbox = self.convert_bbox_to_usable(bbox)
                    self.click_bobber_bbox(bbox)

                    self.set_blacklist()
                    self.loot_classifier.collect_loot()

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
                none_detection_threshold = (10, 20)
                if self.last_predictions_equal(
                    prediction="none",
                    prediction_count=none_detection_threshold[0],
                    prediction_history_length=none_detection_threshold[1],
                ):
                    print(
                        f'{none_detection_threshold[0]} of the last {none_detection_threshold[1]} predictions were "none".'
                    )
                    print("-" * 50 + "\n" + "Starting fishing...")
                    self.start_fishing()

            self.add_time_taken(time.time() - start_time)





class LootClassifier:
    COLOR_PALETTE = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (192, 192, 192),  # Silver
        (128, 128, 128),  # Gray
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (0, 0, 0),  # Black
    ]
    POSITIVE_DETECTION_THRESHOLD = 0.99

    def __init__(self):
        self.blacklist_loot = [
            # "Raw Brillian Smallfish",
            # "Raw Bristle Whisker Catfish",
            # "Raw Longjaw Mud Snapper",
            # "Raw Rockscale Cod",
            "Sturdy Locked Chest",
            "Sealed Crate",
            # "Raw Spotted Yellowtail",
        ]

    def wait_for_loot_window(self):
        wait_timeout = 10  # s
        wait_start_time = time.time()
        while time.time() - wait_start_time < wait_timeout:
            if self.loot_window_exists():
                return True
            time.sleep(0.1)
        return False

    def collect_loot(self) -> bool:
        print(f"This time, blacklist = {self.blacklist_loot}")
        # wait for loot window to appear
        if self.wait_for_loot_window() is False:
            print("Loot window did not appear in time.")
            return False

        # classify that loot
        loot_classification, score = self.classify_loot()
        if score < self.POSITIVE_DETECTION_THRESHOLD:
            # input(f"This is a low score. is this a new loot type?")
            loot_classification = f"unknown_{score}_" + loot_classification

        # if its blacklist loot, close the loot window to skip it
        close_loot_coords = (1035, 110)
        if loot_classification in self.blacklist_loot:
            print("That's a blacklisted loot. Closing loot window.")
            pyautogui.click(*close_loot_coords, clicks=3, interval=0.5)
            return False

        # click the loot
        collect_loot_coords = (950, 160)
        pyautogui.click(*collect_loot_coords, button="right", clicks=3, interval=0.3)

        return True

    def loot_window_exists(self):
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

    def get_color_frequencies(self, image: np.ndarray) -> dict:
        """
        Returns a dictionary of 12 colors and the frequency of pixels that align with that color.

        Args:
            image (np.ndarray): A numpy array representing the image, expected to be of shape (height, width, 3).

        Returns:
            dict: A dictionary where keys are RGB color tuples and values are their pixel frequencies.
        """
        # Initialize the dictionary to count frequencies
        color_count = defaultdict(int)

        # inti it will all the colors at zero
        for color in self.COLOR_PALETTE:
            color_count[color] = 0

        # Reshape the image to a 2D array of pixels (each pixel is a 3-element RGB tuple)
        height, width, _ = image.shape
        pixels = image.reshape(-1, 3)

        # Iterate over each pixel
        for pixel in pixels:
            # Find the closest color in the COLOR_PALETTE
            min_distance = float("inf")
            closest_color = None

            for color in self.COLOR_PALETTE:
                # Calculate Euclidean distance between the pixel and each color
                distance = np.linalg.norm(np.array(pixel) - np.array(color))

                if distance < min_distance:
                    min_distance = distance
                    closest_color = color

            # Increment the frequency for the closest color
            color_count[closest_color] += 1

        return dict(color_count)

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
        print(
            "Loot colors:",
            loot_colors_to_printable(self.get_color_frequencies(loot_image)),
        )

        class_scores = self.classification_scorer(loot_image)
        # print("\nRaw class scores\n", class_scores)

        best_label, best_score = get_highest_score(class_scores)

        if best_score < self.POSITIVE_DETECTION_THRESHOLD:
            color_dict = self.get_color_frequencies(loot_image)
            self.save_unknown_loot_image(loot_image, color_dict)

        formatted_best_score = str(float(best_score) * 100).split(".")[0] + "%"
        print("Class:", best_label)
        print("Score:", formatted_best_score)

        return (best_label, best_score)

    def classification_scorer(self, image: np.ndarray) -> list:
        """
        Classifies the image and returns a list of tuples (class_name, score) based on color similarity.

        Args:
            image (np.ndarray): The image to classify.

        Returns:
            list: A list of tuples in the form (class_name, score).
        """
        # Step 1: Get the color frequencies of the image
        image_color_frequencies = self.get_color_frequencies(image)

        # Step 2: Compare the image color frequencies against each class in loot_dict
        scores = {}
        for class_name, class_color_frequencies in LOOT_COLOR_DATA.items():
            # Calculate the score for this class
            score = self.calculate_class_score(
                image_color_frequencies, class_color_frequencies
            )
            scores[class_name] = score

        return scores

    def calculate_class_score(
        self, image_frequencies: dict, class_frequencies: dict
    ) -> float:
        """
        Calculate a score for a class based on how well its color frequencies match the image.

        Args:
            image_frequencies (dict): The color frequencies of the image.
            class_frequencies (dict): The color frequencies for the class.

        Returns:
            float: A score between 0 and 1.
        """
        # We can compute the similarity as an inverse of the Euclidean distance
        distance = 0.0
        total_pixels = sum(image_frequencies.values())

        # Iterate through all color keys (we assume the color palette is the same for all classes)
        for color in self.COLOR_PALETTE:
            image_count = image_frequencies.get(color, 0)
            class_count = class_frequencies.get(color, 0)

            # We calculate the squared difference of the relative frequencies of each color
            image_relative_frequency = (
                image_count / total_pixels if total_pixels > 0 else 0
            )
            class_relative_frequency = (
                class_count / total_pixels if total_pixels > 0 else 0
            )

            distance += (image_relative_frequency - class_relative_frequency) ** 2

        # Step 4: Convert the distance to a score (using the inverse of the distance)
        score = 1 / (1 + distance)  # Higher scores mean more similar

        return score

    def get_loot_image(self) -> ndarray:
        image = pyautogui.screenshot(region=[944, 147, 20, 20])
        image = np.array(image)
        return image


if __name__ == "__main__":
    run_bot_with_gui()

    # bot = WoWFishBot(None, save_images=False,)
    # lc =  LootClassifier()
    # image = lc.get_loot_image()
    # plt.imshow(image)
    # plt.show()
