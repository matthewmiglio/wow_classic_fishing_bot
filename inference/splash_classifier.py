import onnxruntime
import cv2
import numpy as np



def draw_text(
    image,
    text,
    position=(10, 10),
    font_size=40,
    font_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    thickness=2,
):
    """
    Draws text on the given image using cv2.

    Parameters:
    - image: numpy.ndarray representing the image to draw on.
    - text: str, the text to be drawn.
    - position: tuple (x, y), the starting position of the text (default: (10, 10)).
    - font_size: int, font size in pixels (default: 40).
    - font_color: tuple (r, g, b), color of the ext in RGB format (default: white).
    - font: cv2 font type, font for text rendering (default: cv2.FONT_HERSHEY_SIMPLEX).
    - thickness: int, thickness of the text characters (default: 2).

    Returns:
    - numpy.ndarray: Image with text drawn.
    """
    cv2.putText(image, text, position, font, font_size / 10, font_color, thickness)
    return image


class SplashClassifier:
    def __init__(self,model_path):
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path)

    def preprocess(self, image):
        try:

            #just make sure the image is 256,256
            image = cv2.resize(image,(256,256))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
        except:
            return False
        return image

    def postprocess(self, output: np.ndarray):
        def format_confidence(conf):
            conf = float(conf) * 100
            conf = str(conf)[:5]
            return conf

        label1,label2 = output[0][0]
        if label1>label2:
            return f'not {format_confidence(label1)}%'
        else:
            return f'splash {format_confidence(label2)}%'

    def run(self, image,draw_result=False):
        pred = self.session.run(None, {self.session.get_inputs()[0].name: image})
        if draw_result:
            draw_text(
                image,
                pred,
                position=(10, 10),
                font_size=40,
                font_color=(255, 255, 255),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                thickness=2,
            )
        return pred


if __name__ == '__main__':
    pass



