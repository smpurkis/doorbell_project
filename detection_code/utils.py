import cv2
import face_recognition
import numpy as np
import onnx
import onnxruntime as ort
from caffe2.python.onnx import backend

import detection_code.box_utils_numpy as box_utils


def load_ultra_fast_model():
    onnx_path = "version-RFB-320.onnx"
    class_names = ["BACKGROUND", "face"]
    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    ort_session = ort.InferenceSession(onnx_path, session_options)
    input_name = ort_session.get_inputs()[0].name
    return ort_session, input_name


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


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def get_face_encodings_ultra_fast(image, **kwargs):
    ort_session = kwargs.get("ort_session")
    input_name = kwargs.get("input_name")
    # This is an experimental function, approximately 50% faster on low number of faces, much faster on lots of faces
    original_image = image
    original_shape = image.shape
    image_std = 128.0
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    confidences, boxes = ort_session.run(None, {input_name: image})
    boxes, labels, probs = predict(original_shape[1], original_shape[0], confidences, boxes, 0.7)

    bboxes = [(bbox[1], bbox[2], bbox[3], bbox[0]) for bbox in boxes.tolist()]
    ultra_faces = extract_faces(original_image, bboxes)
    ultra_encodings = []
    for face_image in ultra_faces:
        ultra_encodings.extend(face_recognition.face_encodings(face_image))
    return ultra_encodings, ultra_faces


def name_faces_in_photo(image, known_faces, ultra=False, **kwargs):
    if ultra:
        unknown_face_encodings, ultra_faces = get_face_encodings_ultra_fast(image, **kwargs)
        faces_in_photo = []
        for index, unknown_face_encoding in enumerate(unknown_face_encodings):
            for known_face in known_faces:
                if is_face_in_image(unknown_face_encoding, known_face):
                    faces_in_photo.append(known_face.name)
                    known_face.save_encoding(unknown_face_encoding)
                    known_face.save_image(ultra_faces[index])
    else:
        unknown_face_encodings = face_recognition.face_encodings(image)
        faces_in_photo = [known_face.name for known_face in known_faces for unknown_face_encoding in
                          unknown_face_encodings if is_face_in_image(unknown_face_encoding, known_face)]
    return faces_in_photo
