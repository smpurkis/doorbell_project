import face_recognition
from pathlib import Path
import numpy as np
import cv2
import time
from detect_recognise_faces.face import Face
import io
import httpx


def get_all_faces():
    faces = []
    faces_folder = Path("faces")
    for face_folder in faces_folder.glob("*"):
        face = Face(face_folder.parent, face_folder.stem)
        faces.append(face)
    return faces


def standardize_image_size(image, image_resolution=480):
    img_arr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    standard_shape = (image_resolution, int(image_resolution * (img_arr.shape[0] / img_arr.shape[1])))
    return cv2.resize(img_arr, dsize=standard_shape)


def extract_faces(image, bounding_boxes):
    faces = []
    for box in bounding_boxes:
        t, r, b, l = box[0], box[1], box[2], box[3]
        x = l
        y = t
        w = r - l
        h = b - t
        face = image[y:y + h, x:x + w]
        faces.append(face)
    return faces


def is_face_in_image(image_encoding, face):
    known_encodings = face.encodings
    results = face_recognition.compare_faces(known_encodings, image_encoding)
    if float(results.count(True)) > 0.3 * len(results):
        return True
    else:
        return False


def name_faces_in_photo(image, known_faces):
    unknown_face_encodings = face_recognition.face_encodings(image)
    faces_in_photo = [known_face.name for known_face in known_faces for unknown_face_encoding in unknown_face_encodings
                      if is_face_in_image(unknown_face_encoding, known_face)]
    return faces_in_photo


if __name__ == "__main__":
    known_faces = get_all_faces()

    images = Path("images").glob("*")
    for image_path in images:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = standardize_image_size(image)

        # image = standardize_image_size(image)
        # print(f"{image_path.stem}: {name_faces_in_photo(image, known_faces)}")
