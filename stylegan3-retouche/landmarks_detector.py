import dlib
from typing import Generator, List, Tuple
from pathlib import Path


class LandmarksDetector:
    def __init__(self, predictor_model_path: str):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector()  # cnn_face_detection_model_v1 also possible
        predictor_path = Path(predictor_model_path)
        if not predictor_path.is_file():
            raise FileNotFoundError(f"Shape predictor not found at: {predictor_path}")
        self.shape_predictor = dlib.shape_predictor(str(predictor_path))

    def get_landmarks(self, image_path: str) -> Generator[List[Tuple[int, int]], None, None]:
        """
        Detect facial landmarks from an image.

        :param image_path: Path to the image file.
        :return: Generator of landmarks (list of (x, y) tuples).
        """
        try:
            img = dlib.load_rgb_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return

        dets = self.detector(img, 1)

        for detection in dets:
            try:
                landmarks = self.shape_predictor(img, detection)
                face_landmarks = [(point.x, point.y) for point in landmarks.parts()]
                yield face_landmarks
            except Exception as e:
                print(f"Exception in get_landmarks(): {e}")
