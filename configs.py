import os

import torch


VIDEO_FILE = "video1.mp4"
DATASET_PATH = "datasets"
OUTPUT_VIDEO_FILE = f"output_{VIDEO_FILE}"
IMAGE_TRAINER_FOLDER = os.path.join(os.path.abspath(os.getcwd()), DATASET_PATH)
video_path = os.path.join("videos", VIDEO_FILE)
confidence_threshold = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128
batch_size = 1
