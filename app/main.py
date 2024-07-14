import base64
import os
import tempfile
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage import io
from .deeplearning import yolo_predictions  # Ensure this import works

app = FastAPI()

# CORS middleware configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the current directory is used for file storage
BASE_PATH = os.getcwd()
TEMP_PATH = os.path.join(BASE_PATH, 'temp')

os.makedirs(TEMP_PATH, exist_ok=True)

@app.get("/test")
def health_check():
    return "API is working fine."

@app.post("/")
async def index(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=TEMP_PATH)
        temp_file.close()

        # Write the uploaded file content to the temporary file
        with open(temp_file.name, "wb") as f:
            f.write(await file.read())

        # Read the uploaded image using skimage
        img = io.imread(temp_file.name)

        # Perform YOLO predictions
        result_img, plate_texts = yolo_predictions(img)

        # Encode the resulting image as base64
        retval, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Construct the response object
        response_data = {
            "predicted_image": f"data:image/jpeg;base64,{encoded_image}",
            "plate_texts": plate_texts,
        }

        return response_data

    finally:
        # Clean up: Delete the temporary file after processing
        os.remove(temp_file.name)
