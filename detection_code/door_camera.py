import io
from threading import Thread

import cv2
import httpx
import numpy as np
import json

from detection_code.utils import standardize_image_size
from text_to_speech.text_to_speech import tts


class DoorCamera:
    def __init__(self, host="localhost", port=8000, **kwargs):
        self.host_address = f"http://{host}:{str(port)}"
        self.camera = cv2.VideoCapture(0)
        self.width = int(np.ceil(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = int(np.ceil(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.camera_fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.recent_names = {}
        if kwargs.get("motion_settings"):
            self.motion_settings = kwargs.get("motion_settings")

    def request_new_faces(self, image):
        # send request to face recognition
        resp = httpx.post(f"{self.host_address}/get_new_faces", files={'file': self.package_image(image)})
        new_faces = json.loads(resp.content)
        print(new_faces)
        if len(new_faces) > 0:
            line_to_say = self._get_line(new_faces)
            tts(line_to_say, lang="en-au", names=new_faces)

    def _get_line(self, new_faces):
        # determine the line to say
        if len(new_faces) == 1:
            line_to_say = f"Ring ring {' '.join(new_faces)} is at the door!"
        else:
            line_to_say = f"Ring ring, {' and '.join(new_faces)} are at the door!"
        return line_to_say

    def package_image(self, image):
        _, image = cv2.imencode(".JPEG", image)
        f = io.BytesIO(image.tobytes())
        return f

    def detect_motion(self, current_image, previous_image, draw=False):
        d = cv2.absdiff(previous_image, current_image)

        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey, (self.motion_settings.get("blur_kernal_size"),
                                       self.motion_settings.get("blur_kernal_size")), 0)
        ret, th = cv2.threshold(blur, self.motion_settings.get("binary_threshold"), 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((self.motion_settings.get("dilate_kernal_size"),
                                          self.motion_settings.get("dilate_kernal_size")), np.uint8),
                             iterations=self.motion_settings.get("dilate_iterations"))
        c, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if draw:
            cv2.drawContours(previous_image, c, -1, (0, 255, 0), 2)
        return len(c) > self.motion_settings.get("contour_threshold"), previous_image

    def run_request_thread(self, current_image):
        image = standardize_image_size(current_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        thread = Thread(target=self.request_new_faces, args=(image,), daemon=True)
        del image
        thread.start()

    def run_camera(self):
        check = 0
        cap = self.camera
        ret, previous_image = cap.read()
        while cap.isOpened():
            ret, current_image = cap.read()
            detected_motion, previous_image = self.detect_motion(current_image, previous_image,
                                                                 draw=self.motion_settings.get("view_motion"))
            if detected_motion:
                checks_per_second = int(self.camera_fps/1.)
                if check % checks_per_second == 0:
                    self.run_request_thread(current_image)
            check += 1
            cv2.imshow('frame', previous_image)
            previous_image = current_image
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    door_camera = DoorCamera()
    door_camera.run_camera()
    # image = cv2.imread("photo.jpg")
    # door_camera.request_new_faces(image)

