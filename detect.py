from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # 'n' = small, fast model

def detect_objects(frame):
    results = model(frame)  # Run detection
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box
            conf = box.conf[0].item()     # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            detections.append(([x1, y1, x2, y2], conf, cls))

    return detections  # Return list of detected objects