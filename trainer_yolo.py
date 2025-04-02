from yolov5 import train, val, detect, export
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "guns-knives-yolo", "data.yaml")

    train.run(
        imgsz=640,
        data=data_path,
        epochs=3,
        batch_size=16,
        weights="yolov5n.pt",
        cache=True,
    )
