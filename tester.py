import cv2
import numpy as np
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from configs import DEVICE, IMG_SIZE, STREAMING, TRAINED_MODEL_PATH, VIDEO_PATH
import cvzone

device = select_device(DEVICE)

model = DetectMultiBackend(TRAINED_MODEL_PATH, device=device)
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
classNames = model.names

cap = cv2.VideoCapture(VIDEO_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = letterbox(frame, IMG_SIZE, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)

    pred = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.2)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(frame, (x1, y1, w, h))

                conf = round(float(conf), 2)

                cls = int(cls)
                name = classNames[cls] if cls < len(classNames) else f"Class {cls}"

                cvzone.putTextRect(
                    frame, f"{name} {conf}", (max(0, x1), max(35, y1)), scale=1.5
                )

    if STREAMING:
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
