import cv2
import numpy as np
import os
import easyocr

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')
print(MODEL_PATH)

# Load YOLO model
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def apply_histogram_equalization(image):
    """Apply histogram equalization to an image."""
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return image

def get_detections(img):
    """Get detections from the YOLO model."""
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Apply histogram equalization
    input_image = apply_histogram_equalization(input_image)

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_suppression(input_image, detections):
    """Perform non-maximum suppression to filter detections."""
    boxes = []
    confidences = []

    image_h, image_w = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # Confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # Probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]

                confidences.append(confidence)
                boxes.append(box)

    # Convert to numpy array for NMS
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # Perform NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

def extract_text(image, bbox):
    """Extract text from the detected region of interest (ROI)."""
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]

    # Apply histogram equalization to ROI
    roi = apply_histogram_equalization(roi)

    if 0 in roi.shape:
        return 'no number'
    else:
        result = reader.readtext(roi, detail=0)
        text = ' '.join(result).strip()
        return text

def yolo_predictions(img):
    """Get YOLO predictions and extract text from detected license plates."""
    # Step 1: Get detections
    input_image, detections = get_detections(img)

    # Step 2: Apply non-maximum suppression
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)

    # Step 3: Extract text from detected number plates
    plate_texts = []
    for ind in index:
        bbox = boxes_np[ind]
        plate_text = extract_text(input_image, bbox)
        plate_texts.append(plate_text)

    # Step 4: Draw bounding boxes (optional)
    for ind in index:
        x, y, w, h = boxes_np[ind]
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return input_image, plate_texts