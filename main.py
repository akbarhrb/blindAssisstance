from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch
from PIL import Image
import io
import cv2

app = FastAPI()

# CORS for Flutter access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter IP or domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv5 model (PyTorch)
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt", source='local', force_reload=False)

# COCO class names
CLASS_NAMES = model.names  # Automatically loaded from the model

def preprocess(image: Image.Image):
    # Convert the image to a format suitable for YOLOv5
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    return image

def postprocess(predictions, conf_threshold=0.3):
    boxes = []
    for *xyxy, conf, cls_id in predictions.xywh[0]:  # Using xywh format for predictions
        if conf >= conf_threshold:
            boxes.append({
                "class_id": int(cls_id),
                "class_name": CLASS_NAMES[int(cls_id)],
                "confidence": round(float(conf), 3),
                "bbox": [round(float(xyxy[0]), 2), round(float(xyxy[1]), 2),
                         round(float(xyxy[2]), 2), round(float(xyxy[3]), 2)]
            })
    return boxes

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    input_image = preprocess(image)
    results = model(input_image)  # Perform inference with YOLOv5

    # Postprocess the results to get bounding boxes and class names
    detections = postprocess(results)

    return {"detections": detections}

@app.get("/hello")
def hello():
    return JSONResponse(content={"message": "Hello from YOLOv5 FastAPI!"})
