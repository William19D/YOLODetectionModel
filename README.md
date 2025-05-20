YOLOv12 Person Tracking System
<div align="center"> <img src="assets/person_tracking_sample.jpg" width="80%" alt="Person Tracking System Demo"> <p>Real-time person detection and trajectory tracking with YOLOv12</p> </div>
📝 Description
This repository contains an advanced person detection and tracking system built on YOLOv12. The system detects people in video streams and visualizes their movement patterns with trajectory lines, providing valuable insights for surveillance, retail analytics, and crowd monitoring applications.

Developed by William19D based on the YOLOv12 framework.

✨ Features
Real-time Person Detection: Uses YOLOv12's attention-based architecture for accurate people detection
Trajectory Visualization: Tracks individuals with unique IDs and shows their movement paths with red trails
False Positive Filtering: Intelligently distinguishes between people and static objects like poles
Multi-scale Detection: Employs two-pass detection to capture difficult cases and partial occlusions
Movement Analysis: Categorizes objects as static or moving based on trajectory patterns
Visual Statistics: Shows current person count, maximum persons detected, and frame information
🛠️ Installation
bash
# Clone the repository
git clone https://github.com/William19D/person-tracker.git
cd person-tracker

# Create conda environment
conda create -n person-tracker python=3.11
conda activate person-tracker

# Install dependencies
pip install -r requirements.txt

# Install YOLOv12
pip install ultralytics
📊 Usage
Basic Command
bash
python person_tracker.py --video your_video.mp4
Advanced Options
bash
python person_tracker.py --video your_video.mp4 --confidence 0.4 --iou 0.45 --save-video True --show-preview True
Running with Python Code
Python
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")  # or use "yolov12n.pt" if available

# Configure video capture
cap = cv2.VideoCapture("your_video.mp4")

# Run the tracker - see configuration section for parameters
# ...rest of the code as provided in the main implementation
⚙️ Configuration
You can customize the tracking system with these parameters:

Python
# Core settings
VIDEO_PATH = "test.mp4"        # Input video file
MODEL = "yolov8n.pt"           # Detection model (compatible with YOLOv8/YOLOv12)
CONFIDENCE = 0.4               # Confidence threshold (0.0-1.0)
SAVE_VIDEO = True              # Save output video with annotations

# Detection parameters
TRACK_PEOPLE_ONLY = True       # Only detect and track people
IOU_THRESHOLD = 0.45           # IoU threshold for NMS
MAX_DETECTIONS = 100           # Maximum objects to detect per frame

# Display settings
SHOW_LIVE_PREVIEW = True       # Show real-time processing window
MIN_HUMAN_HEIGHT = 40          # Minimum pixel height to consider as human
MIN_HUMAN_WIDTH = 15           # Minimum pixel width to consider as human
📋 Output Examples
The system produces several outputs:

Processed Video: Video file with person detections and trajectories
Detection Images:
Periodic frame captures saved to /detections folder
Maximum person count frame saved as max_N_people_frame_X.jpg
Terminal Statistics:
Processing progress and estimated time
Final count statistics and performance metrics
🌟 Applications
Security & Surveillance: Track individuals across camera views
Retail Analytics: Analyze customer movement patterns in stores
Urban Planning: Study pedestrian flow in public spaces
Event Management: Monitor crowd density and movement at events
Traffic Analysis: Analyze pedestrian crossing patterns
👨‍💻 Code Example
Here's a simplified version of the core tracking logic:

Python
# Process detected persons
for person_box in person_boxes:
    # Extract box coordinates and tracking ID
    x1, y1, x2, y2, track_id, confidence = extract_box_info(person_box)
    
    # Validate if detection is actually a person
    if not is_valid_human(x1, y1, x2, y2, track_id):
        continue
    
    # Calculate center point for tracking
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Add to tracking history
    track_history[track_id].append((center_x, center_y))
    
    # Draw bounding box (green)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw trajectory trail (red)
    for j in range(1, len(track_history[track_id])):
        if len(track_history[track_id]) > 30:  # Limit trail length
            track_history[track_id].pop(0)
        
        # Draw line between consecutive points
        pt1 = track_history[track_id][j-1]
        pt2 = track_history[track_id][j]
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
🙏 Acknowledgements
YOLOv12 algorithm by Yunjie Tian, Qixiang Ye, and David Doermann
Based on ultralytics implementation
OpenCV community for image processing tools
📄 Citation
BibTeX
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
📧 Contact
For questions or feedback, please contact:

GitHub: @William19D
<p align="center">Made with ❤️ for computer vision and AI research</p>
