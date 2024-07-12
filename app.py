from flask import Flask, render_template, request, jsonify
import os
from skimage import io
import cv2
from deeplearning import yolo_predictions  # Assuming you have defined this function

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'project/static/upload/')
PREDICT_PATH = os.path.join(BASE_PATH, 'project/static/predict/')

# Ensure the directories exist
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(PREDICT_PATH, exist_ok=True)

@app.route('/', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = file.filename
    path_save = os.path.join(UPLOAD_PATH, filename)
    file.save(path_save)

    # Load the uploaded image
    img = io.imread(path_save)

    # Make predictions using your model (assuming yolo_predictions function is defined)
    result_img, plate_texts = yolo_predictions(img)

    # Save the predicted image
    predicted_filename = 'predicted_' + filename
    result_path = os.path.join(PREDICT_PATH, predicted_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    # Return the predicted filename or results to the client (chatbot)
    return jsonify({'predicted_filename': predicted_filename, 'plate_texts': plate_texts})

if __name__ == "__main__":
    app.run(debug=True)
