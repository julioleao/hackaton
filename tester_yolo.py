import math
import cv2
import torch
from ultralytics import YOLO
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from configs import video_path
import cvzone

# Carrega o modelo treinado
model = YOLO('yolov5n.pt')


# Carregar vídeo ao vivo
cap = cv2.VideoCapture(video_path)  # Pode usar 0 para webcam
classNames = ["knife", "gun"]


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = classNames[int(cls)]

            cvzone.putTextRect(frame, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)



    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
