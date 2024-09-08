# CogniAble_Assignment

# YOLOv5 and DeepSORT Object Tracking Project

This project uses YOLOv5 for object detection and DeepSORT for real-time multi-object tracking. It processes video frames to detect and track persons in the video and assigns unique IDs to each person detected.

## Features

- Detect persons using YOLOv5.
- Track detected persons across video frames using DeepSORT.
- Output video with tracked persons and their corresponding IDs.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- DeepSORT (deep_sort_realtime)
- Pandas
- Ultralytics (for YOLOv5)

## Setup

1. Clone this repository and navigate to the project directory:
    ```bash
    git clone https://github.com/yourusername/yolov5-deepsort-tracking.git
    cd yolov5-deepsort-tracking
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv5 model:
    ```bash
    python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5', 'ultralytics/yolov5')"
    ```

4. Place your video file in the project directory or update the `video_path` in the script to point to your own video.

5. Run the tracker:
    ```bash
    python your_script.py
    ```

## Usage

- The script processes the video frame by frame, detects persons, and tracks them. It outputs a new video file with bounding boxes and IDs around detected persons.

## Customization

- **Detection Confidence**: You can adjust the detection confidence threshold by changing the `if row['confidence'] > 0.5` condition in the script.
- **Tracking**: Customize the DeepSORT tracker by adjusting parameters such as `max_age`, `max_cosine_distance`, etc.

## Output

- The output video will have the same resolution and frame rate as the input video, with detected persons annotated with unique IDs.

## Example

Here's how to run the tracker on a sample video:
```bash
python your_script.py

```

Please Note : We can still work on the accuracy terms if given more time
