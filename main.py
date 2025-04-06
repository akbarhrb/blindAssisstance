from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# Load the YOLO model once on startup
model = YOLO("yolov8n.pt")  # You can change to your trained .pt if needed

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        results = model(temp_path)
        detections = results[0].boxes

        response = []
        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                response.append({
                    "class": class_name,
                    "confidence": round(conf, 3)
                })

        # Clean up the temporary file
        os.remove(temp_path)
        return JSONResponse(content={"results": response})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
