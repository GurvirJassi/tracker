import cv2
import os
# print(cv2.getBuildInformation())
from detect import detect_objects
from track import track_objects
from utils import draw_boxes

# Load video
video_path = "/workspaces/tracker/resources/tennisballs.mp4"
cap = cv2.VideoCapture(video_path)


print("File exists:", os.path.exists(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create output directory for frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

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

    # Save each frame with a unique name
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved {frame_filename}")  

    frame_count += 1


cap.release()
print(f"All frames saved in {output_dir}/")

#ffmpeg -framerate 30 -i frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output_video.mp4
