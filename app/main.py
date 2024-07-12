import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from skimage import io
import cv2
from .deeplearning import yolo_predictions  # Ensure this import works

app = FastAPI()

# CORS middleware configuration
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
PREDICT_PATH = os.path.join(BASE_PATH, 'static/predict/')

os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(PREDICT_PATH, exist_ok=True)

@app.get("/test")
def healt_check():
    return "API is working fine."

@app.post("/")
async def index(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_PATH, file.filename)
    
    try:
        with open(file_location, "wb") as file_object:
            file_object.write(file.file.read())
    except Exception as e:
        return {"error": str(e)}
    
    img = io.imread(file.file)
    result_img, plate_texts = yolo_predictions(img)

    # Encode the resulting image as base64
    retval, buffer = cv2.imencode('.jpeg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Construct the response object
    response_data = {
        "predicted_image": f"data:image/jpeg;base64,{encoded_image}",
        "plate_texts": plate_texts,
    }

    return response_data

