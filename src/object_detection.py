"""
Step 6: Human detection using YOLOv8 (pretrained).
Detects only the 'person' class and draws boxes.
"""
from ultralytics import YOLO
from PIL import Image
import numpy as np

# COCO class index for 'person' is 0
PERSON_CLASS = 0

_model = None  # lazy global so we load weights only once


def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # nano = fast, auto-downloads first time
    return _model


def detect_humans(image, conf=0.4):
    """
    image: PIL.Image or numpy array
    Returns: (annotated_image_array, num_people)
    """
    model = get_model()
    results = model(image, classes=[PERSON_CLASS], conf=conf, verbose=False)
    r = results[0]
    num_people = len(r.boxes)
    annotated = r.plot()  # numpy array (BGR) with boxes drawn
    # convert BGR -> RGB for correct display
    annotated = annotated[:, :, ::-1]
    return annotated, num_people