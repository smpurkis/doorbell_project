import cv2
from detect_recognise_faces.recognise_face import standardize_image_size, get_all_faces
from text_to_speech.text_to_speech import tts
from threading import Thread
import io
import httpx
import time
import json

recent_names = {}


def request_who_is_in_image(image, refresh_time=5., repeat=1):
    _, image = cv2.imencode(".JPEG", image)
    f = io.BytesIO(image.tobytes())
    files = {'file': f}
    r = httpx.post("http://127.0.0.1:8000/recognise_faces", files=files)
    face_names = json.loads(r.content)
    print("before", face_names)
    face_names = [name for name in face_names if time.time() - recent_names.get(name, time.time()) > refresh_time or name not in recent_names]
    print("after", face_names)
    if len(face_names) > 0:
        print("before", recent_names)
        [recent_names.update({name: time.time()}) for name in face_names if name not in recent_names or time.time() - recent_names.get(name, time.time()) > refresh_time]
        print("after", recent_names)
        if len(face_names) == 1:
            line_to_say = f"{' '.join(face_names)} is at the door!"
        else:
            line_to_say = f"{' and '.join(face_names)} are at the door!"
        tts(line_to_say, lang="en-au", names=face_names)


known_faces = get_all_faces()

cap = cv2.VideoCapture(0)

check = 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    image = standardize_image_size(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if check % 30 == 0:
        thread = Thread(target=request_who_is_in_image, args=(image,), daemon=True)
        thread.start()
    check += 1

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
