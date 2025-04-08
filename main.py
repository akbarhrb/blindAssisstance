from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow.lite as tflite  # Use tflite_runtime if deploying to Render

app = FastAPI()

# CORS for Flutter access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter IP or domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="yolov8n_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# COCO class names
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def preprocess(image: Image.Image):
    image = image.resize((640, 640)).convert("RGB")
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data.astype(np.float32)

def postprocess(predictions, conf_threshold=0.3):
    boxes = []
    for pred in predictions[0]:
        pred = pred[:6]  # Use only [x1, y1, x2, y2, conf, class_id]
        x1, y1, x2, y2, conf, cls_id = pred
        if conf >= conf_threshold:
            boxes.append({
                "class_id": int(cls_id),
                "class_name": CLASS_NAMES[int(cls_id)],
                "confidence": round(float(conf), 3),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2), 2), round(float(y2), 2)]
            })
    return boxes

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    input_tensor = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    results = postprocess(output_data)

    return {"detections": results}

@app.get("/hello")
def hello():
    return JSONResponse(content={"message": "Hello from YOLO FastAPI!"})
