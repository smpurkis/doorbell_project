import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File
from fastapi.logger import logger

from detection_code.face import Face
from detection_code.utils import name_faces_in_photo, standardize_image_size, load_ultra_fast_model


def get_all_faces(faces_path):
    faces = []
    faces_folder = Path(faces_path)
    for face_folder in faces_folder.glob("*"):
        face = Face(face_folder.parent, face_folder.stem)
        faces.append(face)
    logger.info(f"Loaded Faces: {faces}")
    return faces


class FaceRecognitionServer:
    def __init__(self, faces_path, ultra=True, **extra: Dict[str, Any]):
        super().__init__(**extra)
        self.known_faces = get_all_faces(faces_path)
        self.ultra = ultra
        if ultra:
            self.ort_session, self.input_name = load_ultra_fast_model()
        self.recent_faces = {}
        self.refresh_time = 10.
        self.load()

    def _get_new_faces(self, face_names):
        # filter to new faces
        return [name for name in face_names if
                time.time() - self.recent_faces.get(name, time.time()) > self.refresh_time or name not in self.recent_faces]

    def _track_calls(self, new_faces):
        # track known faces in recent names dictionary
        [self.recent_faces.update({name: time.time()}) for name in new_faces if
         name not in self.recent_faces or time.time() - self.recent_faces.get(name, time.time()) > self.refresh_time]

    def load(self):
        self.app = FastAPI()

        @self.app.get("/")
        async def read_root():
            return "hello world"

        @self.app.post("/get_faces")
        async def recognise_faces(file: bytes = File(...)):
            image = cv2.imdecode(np.frombuffer(file, np.uint8), -1)
            image = standardize_image_size(image)
            faces = name_faces_in_photo(image, self.known_faces)
            return faces

        @self.app.post("/get_new_faces")
        async def recognise_faces(file: bytes = File(...)):
            image = cv2.imdecode(np.frombuffer(file, np.uint8), -1)
            image = standardize_image_size(image)
            faces = name_faces_in_photo(image, self.known_faces, self.ultra,
                                        ort_session=self.ort_session,
                                        input_name=self.input_name)
            if len(faces) == 0:
                return faces
            faces = self._get_new_faces(faces)
            self._track_calls(faces)
            return faces


server = FaceRecognitionServer("faces").app


def run_server():
    uvicorn.run("face_recognition_server:server", host="127.0.0.1", port=8000, workers=2)


if __name__ == "__main__":
    run_server()
