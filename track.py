from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def track_objects(detections, frame):
    tracks = tracker.update_tracks(detections, frame=frame)
    tracked_objects = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Bounding box coordinates
        tracked_objects.append((track_id, ltrb))

    return tracked_objects

