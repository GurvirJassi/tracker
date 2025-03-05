import cv2
from detect import detect_objects
from track import track_objects
from utils import draw_boxes

# Load video
video_path = "/Users/lucst/myenv/MarkerDrop.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    detections = detect_objects(frame)

    #Show coordinates of object on screen
    for bbox, conf, cls in detections:
        x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
        print(f"Object detected: Class {cls}, Confidence {conf:.2f}, Position: ({x1}, {y1}), ({x2}, {y2})")

    # Track objects
    tracked_objects = track_objects(detections, frame)

    # Draw bounding boxes
    frame = draw_boxes(frame, detections, tracked_objects)

    # Display frame
    cv2.imshow("Object Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
