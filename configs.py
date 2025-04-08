import os

import torch

DATASET_TYPE = "knives-lite"
VIDEO_FILE = "video1.mp4"
OUTPUT_VIDEO_FILE = f"output_{VIDEO_FILE}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join("videos", VIDEO_FILE)
KNIVES_YOLO_PATH = os.path.join(BASE_DIR, "dataset", DATASET_TYPE)
DATA_PATH = os.path.join(KNIVES_YOLO_PATH, "data.yaml")
TRAIN_PATH = os.path.join(KNIVES_YOLO_PATH, "train", "images")
VAL_PATH = os.path.join(KNIVES_YOLO_PATH, "valid", "images")
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "runs", "train", "exp", "weights", "best.pt")
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
STREAMING = True
