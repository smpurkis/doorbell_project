import face_recognition
from pathlib import Path
import numpy as np
import cv2


class Face:
    def __init__(self, parent_path, name=None, new_face=False):
        self.name = name
        self.folder_path = Path(parent_path, name)
        self.images_path = Path(self.folder_path, "images")
        self.encodings_path = Path(self.folder_path, "encodings")
        self.make_dirs(self.folder_path)
        self.number_of_images = len(list(self.images_path.glob("*")))
        if not new_face:
            self.encodings = self.load_encodings(self.encodings_path)

    def make_dirs(self, folder_path):
        if not folder_path.exists():
            folder_path.mkdir()
        if not self.encodings_path.exists():
            self.encodings_path.mkdir()
        if not self.images_path.exists():
            self.images_path.mkdir()

    def load_encodings(self, encodings_path):
        return [self.load_encoding(encoding_path) for encoding_path in encodings_path.glob("*")]

    def check_encoding(self, encoding):
        if encoding.shape != (128):
            encoding = encoding.squeeze(0)
        return encoding

    def load_encoding(self, encoding_path):
        if encoding_path.exists() and encoding_path.is_file():
            encoding = np.load(str(encoding_path))
            encoding = self.check_encoding(encoding)
            return encoding
        else:
            return self.get_encoding(encoding_path)

    def get_encoding(self, encoding_path):
        image_path = Path(encoding_path.parent, "image", encoding_path.stem)
        if image_path.exists() and image_path.is_file():
            image = self.load_image(image_path)
            return face_recognition.face_encodings(image)

    def load_image(self, image_path=False, image_number=None):
        if image_number is not None:
            image_path = Path(self.images_path, image_number)
        return cv2.imread(str(image_path))

    def save_encoding(self, encoding):
        encoding_path = Path(self.encodings_path, f"{str(self.number_of_images + 1)}")
        np.save(str(encoding_path), encoding)

    def save_image(self, image):
        image_path = Path(self.images_path, f"{str(self.number_of_images + 1)}.png")
        cv2.imwrite(str(image_path), image)
