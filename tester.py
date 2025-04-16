import cv2
import numpy as np
import torch
import subprocess
import math
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from configs import DEVICE, IMG_SIZE, STREAMING, TRAINED_MODEL_PATH, VIDEO_PATH
import cvzone

knife_counter = 0
pending_knives = []      
confirmed_knives = []    
DIST_THRESHOLD = 10 
CONFIRM_FRAMES = 4       


def send_notification(message):
    subprocess.run(["notify-send", message])

def setup_model():
    device = select_device(DEVICE)
    model = DetectMultiBackend(TRAINED_MODEL_PATH, device=device)
    model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
    return model, device

def setup_output_writer(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

def preprocess_frame(frame, device):
    img = letterbox(frame, IMG_SIZE, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    return img.unsqueeze(0)

def draw_detections(frame, detections, classNames):
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h))
        name = classNames[int(cls)] if int(cls) < len(classNames) else f"Class {cls}"
        cvzone.putTextRect(frame, f"{name} {round(float(conf), 2)}", (max(0, x1), max(35, y1)), scale=1.5)

def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_near(center1, center2, threshold):
    dist = math.hypot(center1[0] - center2[0], center1[1] - center2[1])
    return dist < threshold

def update_pending_knife(center):
    found = False
    for knife in pending_knives:
        if is_near(center, knife["center"], DIST_THRESHOLD):
            knife["frames"] += 1
            found = True
            break
    if not found:
        pending_knives.append({"center": center, "frames": 1})

def promote_confirmed_knives(frame_id):
    global knife_counter
    new_confirmed = []

    for knife in pending_knives:
        if knife["frames"] >= CONFIRM_FRAMES:
            if not any(is_near(knife["center"], c, DIST_THRESHOLD) for c in confirmed_knives):
                confirmed_knives.append(knife["center"])
                new_confirmed.append(knife)
                knife_counter += 1
                send_notification(f"Objeto cortante detectado!!! Total identificado: {knife_counter}")

    for knife in new_confirmed:
        pending_knives.remove(knife)

def check_for_knives(detections, classNames, frame_id):
    for *xyxy, conf, cls in detections:
        class_index = int(cls)
        class_name = classNames[class_index].lower()

        if "knife" in class_name:
            center = get_center(xyxy)
            update_pending_knife(center)

    promote_confirmed_knives(frame_id)

def main():
    model, device = setup_model()
    classNames = model.names

    cap = cv2.VideoCapture(VIDEO_PATH)
    out = setup_output_writer(cap)
    frame_id = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            img = preprocess_frame(frame, device)

            with torch.no_grad():
                pred = model(img)

            pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.4)

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    check_for_knives(det, classNames, frame_id)
                    draw_detections(frame, det, classNames)

            if STREAMING:
                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            out.write(frame)

    except Exception as e:
        print(f"[Erro] Falha durante o processamento: {e}")

    finally:
        cap.release()
        out.release()
        send_notification(f"Total de objetos identificados #{knife_counter}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
