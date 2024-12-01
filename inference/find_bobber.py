print("Initalizing bobber detector model...")
import torch
import cv2
import numpy as np
import onnxruntime


class BobberDetector:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)

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

    def run_detection(self, img, conf_thres=0.25, iou_thres=0.45):
        # Load and preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[100:900, 700:1300]
        # img = cv2.resize(img, (256, 256))  # Resize to 256x256
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1)).reshape(
            1, 3, 256, 256
        )  # Convert to [1, 3, 256, 256]

        # Inference
        pred = self.session.run(None, {self.session.get_inputs()[0].name: img})

        # Apply NMS
        boxes, scores = self.non_max_suppression(
            torch.tensor(pred[0]), conf_thres, iou_thres
        )

        return boxes, scores

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
        boxes, scores = self.run_detection(img, conf_thres=0.25, iou_thres=0.45)
        if len(boxes) == 0:

            return [], 0
        scores = np.array(scores)
        max_score = np.max(scores)
        index_of_max_score = np.where(scores == max_score)[0][0]
        best_box = boxes[index_of_max_score]

        if draw_result:
            self.draw_bbox(img, best_box, "best", (255, 0, 0))

        def convertbbox2list(bbox):
            bbox = bbox.tolist()
            return [int(x) for x in bbox]

        best_box = convertbbox2list(best_box)
        return best_box, max_score
