from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from detect_recognise_faces.recognise_face import name_faces_in_photo, get_all_faces, standardize_image_size


class FaceRecognitionServer:
    def __init__(self, **extra: Dict[str, Any]):
        super().__init__(**extra)
        self.known_faces = get_all_faces()
        self.load()

    def load(self):
        self.app = FastAPI()

        @self.app.get("/")
        async def read_root():
            return {"Hello": "World"}

        @self.app.get("/items/{item_id}")
        async def read_item(item_id: int, q: Optional[str] = None):
            return {"item_id": item_id, "q": q}

        @self.app.post("/recognise_faces")
        async def recognise_faces(file: bytes = File(...)):
            image = cv2.imdecode(np.frombuffer(file, np.uint8), -1)
            image = standardize_image_size(image)
            faces = name_faces_in_photo(image, self.known_faces)
            return faces


server = FaceRecognitionServer().app
if __name__ == "__main__":
    uvicorn.run("face_recognition_server:server", host="127.0.0.1", port=8000, workers=2)
