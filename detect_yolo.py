from yolov5 import detect
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "videos")
    model_path = os.path.join(base_dir, "runs", "train", "exp", "weights", "best.pt")

    video1 = os.path.join(video_dir, "video1.mp4")
    video2 = os.path.join(video_dir, "video2.mp4")

    detect.run(
        weights=model_path,
        source=video1,
        imgsz=640,
        conf_thres=0.5,
        save_txt=True,
        save_conf=True,
        save_vid=True,
    )

    detect.run(
        weights=model_path,
        source=video2,
        imgsz=640,
        conf_thres=0.5,
        save_txt=True,
        save_conf=True,
        save_vid=True,
    )
