print("Initalizing bobber detector model...")
import cv2
import numpy as np
import onnxruntime
import torch


class MultiBobberDetector:
    def __init__(self, model_paths: list[str]):
        self.bds = [BobberDetector(model_path) for model_path in model_paths]
        self.verbose = False

    def stretch_bbox(self, bbox, prev_image_width, new_image_width):
        normalized_bbox = [i / prev_image_width for i in bbox]
        new_bbox = [int(i * new_image_width) for i in normalized_bbox]
        return new_bbox

    def normalize_bbox(self, bbox, img_width):
        normalized_bbox = [i / img_width for i in bbox]
        return normalized_bbox

    def denormalize_bbox(self, bbox, img_width):
        denormalized_bbox = [i * img_width for i in bbox]
        return denormalized_bbox

    def get_avg_bbox(self, bboxes):
        if not bboxes:  # If bboxes is empty, return an empty list
            return []

        # Convert bboxes to a NumPy array for faster processing
        bboxes = np.array(bboxes)

        # Compute the average bbox by averaging along the 0th axis (rows)
        avg_bbox = np.mean(bboxes, axis=0)

        # Convert the average bounding box values to integers and return as a list
        return [int(coord) for coord in avg_bbox]

    def print_bbox(self, bbox, score):
        def format_score(score):
            score = score * 100
            score = round(score, 2)
            score = f"{score}%"
            return score

        if bbox == []:
            print(" x " * 30)
            return

        s = ""
        for i in bbox:
            s += f"{round(i,2)}, "
        s = s[:-2]
        s += f" score: {format_score(score)}"
        print(s)

    def bboxes_disagree(self, bboxes):

        difference_threshold = 45  # pixels
        all_center_xs = [bbox[0] for bbox in bboxes]
        all_center_ys = [bbox[1] for bbox in bboxes]

        if max(all_center_xs) - min(all_center_xs) > difference_threshold:
            return True

        if max(all_center_ys) - min(all_center_ys) > difference_threshold:
            return True

        return False

    def detect_object_in_image(self, img):
        if self.verbose:
            print("--" * 50)
        start_image_width = img.shape[0]

        best_bboxes = []
        scores = []
        for i, bd in enumerate(self.bds):
            model_img_size = bd.session.get_inputs()[0].shape[2]
            if model_img_size != img.shape[0]:
                input_image = bd.preprocess_image(
                    cv2.resize(img, (model_img_size, model_img_size))
                )
            else:
                input_image = bd.preprocess_image(img)

            best_bbox, max_score = bd.postprocess_output(
                bd.session.run(None, {bd.session.get_inputs()[0].name: input_image})
            )
            scores.append(max_score)
            best_bbox = self.normalize_bbox(best_bbox, model_img_size)
            best_bbox = self.denormalize_bbox(best_bbox, start_image_width)
            if best_bbox != []:
                best_bboxes.append(best_bbox)
            if self.verbose:
                self.print_bbox(best_bbox, max_score)

        if len(best_bboxes) == 0:
            if self.verbose:
                print("No bboxes detected, returning empty bbox")
            return [], 0

        # check for models heavily disagreeing
        if self.bboxes_disagree(best_bboxes):
            if self.verbose:
                print("Bboxes disagree, returning empty bbox")
            return [], 0

        # otherwise, return an avreage between the models
        avg_bbox = self.get_avg_bbox(best_bboxes)
        avg_score = np.mean(scores) if scores else 0
        if self.verbose:
            print(f"Avg bbox: {avg_bbox}")
        return avg_bbox, avg_score


class BobberDetector:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_image_size = None
        self.output_image_size = None

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """Applies Non-Maximum Suppression to filter overlapping detections and return boxes with their scores."""
        boxes = prediction[..., :4]
        scores = prediction[..., 4]

        # Filter by confidence threshold
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return [], []  # Return empty lists for both boxes and scores

        # Sort by confidence
        indices = scores.argsort(descending=True)

        # Apply NMS
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i.item())  # Convert tensor to integer
            iou = self.box_iou(boxes[i], boxes[indices[1:]])  # Calculate IoU
            indices = indices[1:][iou <= iou_thres]

        return boxes[keep], scores[keep]  # Return boxes and their corresponding scores

    def box_iou(self, box1, box2):
        """Calculates IoU (Intersection over Union) between two sets of boxes."""
        x1 = torch.max(box1[0], box2[:, 0])
        y1 = torch.max(box1[1], box2[:, 1])
        x2 = torch.min(box1[2], box2[:, 2])
        y2 = torch.min(box1[3], box2[:, 3])

        inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        return inter_area / (box1_area + box2_area - inter_area)

    def preprocess_image(self, img):
        img_size = img.shape[0]
        self.input_image_size = img_size
        expected_model_dims = self.session.get_inputs()[0].shape[2]
        #resize the image to the expected model dims
        img = cv2.resize(img, (expected_model_dims, expected_model_dims))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1)).reshape(
            1, 3, expected_model_dims, expected_model_dims
        )
        self.output_image_size = expected_model_dims
        return img

    def postprocess_output(self, pred):
        def convertbbox2list(bbox):
            bbox = bbox.tolist()
            return [int(x) for x in bbox]

        def convert_bbox_to_input_dims(bbox,input_image_size,output_image_size):
            normalized_bbox = [i / output_image_size for i in bbox]
            bbox_for_input_image = [int(i * input_image_size) for i in normalized_bbox]
            return bbox_for_input_image

        boxes, scores = self.non_max_suppression(
            torch.tensor(pred[0]), conf_thres=0.25, iou_thres=0.45
        )

        if len(boxes) == 0:
            return [], 0

        scores = np.array(scores)
        max_score = np.max(scores)
        index_of_max_score = np.where(scores == max_score)[0][0]
        best_bbox = boxes[index_of_max_score]
        best_bbox = convertbbox2list(best_bbox)
        best_bbox = convert_bbox_to_input_dims(best_bbox,self.input_image_size,self.output_image_size)
        return best_bbox, max_score

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        label,
        color: tuple[int, int, int],
    ):
        label = str(label)
        bbox = [int(x) for x in bbox]

        x, y, w, h = bbox[:4]
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text_x_pos = x1
        text_y_pos = y1 - 10

        cv2.putText(
            image,
            label,
            (text_x_pos, text_y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    def detect_object_in_image(self, img, draw_result=False):
        img = self.preprocess_image(img)
        best_bbox, max_score = self.postprocess_output(
            self.session.run(None, {self.session.get_inputs()[0].name: img})
        )
        if draw_result:
            self.draw_bbox(img, best_bbox, "best", (255, 0, 0))
        return best_bbox, max_score
