import threading
import pyautogui
import numpy as np
import time
import os
import cv2
from inference.find_bobber import BobberDetector
from inference.splash_classifier import SplashClassifier


class WoWFishBot:
    def __init__(self, save_splash_images_toggle=False):
        # the bobber detector and splash classifier models we'll use this run
        self.bobber_detector = BobberDetector(r"inference\bobber_finder.onnx")
        self.splash_clasisfier = SplashClassifier(r"inference\splash_classifier.onnx")

        # saving images for the splash classifier to learn from in the future
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
        image = image[:, :, ::-1].copy()
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
        # pyautogui send the key z
        self.focus_wow()
        pyautogui.press("z")
        time.sleep(2)

    def send_delayed_click(self, x, y, wait):
        def _to_wrap(x, y, wait):
            time.sleep(wait)
            pyautogui.click(x, y, button="right")

        print(f"Queueing a click for {wait}s in the future...")
        t = threading.Thread(target=_to_wrap, args=(x, y, wait))
        t.start()

    def click_bbox(self, bbox):
        def calc_coord():
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

        coord = calc_coord()
        print(f"CLICKING THIS COORD: {coord}")
        # pyautogui.click(*coord, button="right")
        self.send_delayed_click(*coord, wait=1)

    def save_splash_images(self, bbox_image, is_splash) -> bool:
        def _to_readable(bbox_image):

            # cut the batch dimension of the bbox image
            bbox_image = bbox_image[0]

            # transpose 0,1,2 to 1,2,0
            bbox_image = np.transpose(bbox_image, (1, 2, 0))

            # multiple by 255
            bbox_image = bbox_image * 255

            return bbox_image

        uid = str(time.time()).replace(".", "") + ".png"
        if "splash" in is_splash:
            path = os.path.join(self.splash_images_dir, "splash", uid)
        else:
            path = os.path.join(self.splash_images_dir, "not", uid)

        try:
            bbox_image = _to_readable(bbox_image)
            cv2.imwrite(path, bbox_image)
        except Exception as e:
            print(f"Failed to save this splash image:\n{e}")
            return False

        return True

    def run(self):
        MIN_CONFIDENCE_FOR_BOBBER_DETECTION = 0.25
        print("Initalizing Bobber Detector and Splash Classifier modules!")

        # loop foreverz
        while 1:
            # search for the bobber in the base image's ROI
            base_image = self.screenshot_bobber_roi()
            bbox, score = self.bobber_detector.detect_object_in_image(
                base_image, draw_result=False
            )

            # if we found a bobber confidently enough
            if score > MIN_CONFIDENCE_FOR_BOBBER_DETECTION:
                # create an image of the bobber ROI
                try:
                    bobber_image = self.make_bobber_image(bbox, base_image)
                    bobber_image = self.splash_clasisfier.preprocess(bobber_image)

                # sometimes that fails, so we'll wait for a better image of the ROI
                except Exception as e:
                    print(f"\n{e}\n")
                    print(
                        "This bbox cannot be converted to a splash image.\nSkipping splash_clasisfier prediction"
                    )
                    continue

                # classify that bobber image as either a splash, or not a splash
                is_splash = self.splash_clasisfier.run(bobber_image, draw_result=False)
                is_splash = self.splash_clasisfier.postprocess(
                    is_splash
                )  # either ['splash 0.984','not 0.016']

                # if we see a splash, click it!
                if "splash" in is_splash:
                    self.click_bbox(bbox)

                # save the image, if that's enabled
                if self.save_splash_images_toggle:
                    self.save_splash_images(bobber_image, is_splash)

            else:
                print("Bobber is not detected!\nStarting fishing...")
                self.start_fishing()
                time.sleep(4)


if __name__ == "__main__":
    wfb = WoWFishBot(save_splash_images_toggle=True)
    wfb.run()
