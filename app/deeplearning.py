import cv2
import numpy as np
import pytesseract
import os

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')
print(MODEL_PATH)

# Load YOLO model
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_detections(img):
    # 1. Convert image to YOLO format
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 2. Get prediction from YOLO model
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_suppression(input_image, detections):
    # 3. Filter detections based on confidence and probability score

    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5*w) * x_factor)
                top = int((cy - 0.5*h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 Clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

def yolo_predictions(img):
    # Step-1: detections
    input_image, detections = get_detections(img)

    # Step-2: NMS
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)

    # Step-3: Extract text from detected number plate
    plate_texts = []
    for ind in index:
        bbox = boxes_np[ind]
        plate_text = extract_text(input_image, bbox)
        plate_texts.append(plate_text)

    # Step-4: Drawings (optional, if needed to return annotated image)
    for ind in index:
        x, y, w, h = boxes_np[ind]
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (255, 0, 255), 2)

    return input_image, plate_texts

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pytesseract.image_to_string(roi)
        text = text.strip()

        return text
