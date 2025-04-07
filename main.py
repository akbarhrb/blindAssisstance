from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Allow Flutter requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Flutter server IP in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="yolov8n_float16.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Replace with your actual class list
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
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    return input_data

def postprocess(output, conf_thres=0.3):
    results = []
    preds = output[0]
    for det in preds:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_thres:
            continue
        results.append({
            "class_id": int(cls_id),
            "class_name": CLASS_NAMES[int(cls_id)],
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })
    return results

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    input_tensor = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    results = postprocess(output)
    return {"detections": results}


@app.get("/hello")
def sayHello():
    return JSONResponse(content={"message" : "hello ali"})
    
