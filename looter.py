from constants import WOW_WINDOW_NAME, TOP_SAVE_DIR
from image_rec import classification_scorer, get_color_frequencies

import time
import pyautogui
import pygetwindow
import random
import numpy as np
import os
from PIL import Image

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

    def get_loot_image(self):
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

    def get_loot_image_old(self):
        image = pyautogui.screenshot(region=[944, 147, 20, 20])
        image = np.array(image)
        return image

