import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device).eval()

deepsort = DeepSort(max_age=10, max_cosine_distance=0.2)



# Define video source
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)
output_path = "updated_output.mp4"



# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = frame[:, :, ::-1]  # Convert BGR to RGB

    # Perform detection
    results = model(img)
    detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
    bbs = []
    for index, row in detections.iterrows():
        if row['confidence'] > 0.5 and row['class'] == 0:  # Only consider 'person' class (class 0)
            x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            width = x_max - x_min
            height = y_max - y_min
            bbs.append(([x_min, y_min, width, height], row['confidence'], row['class']))


    # Track objects
    if bbs:
        tracks = deepsort.update_tracks(bbs, frame=frame)

        # Draw bounding boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Convert to left-top-right-bottom format
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

