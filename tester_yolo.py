import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from configs import video_path

# Carrega o modelo treinado
device = select_device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend("best.pt", device=device)
model.eval()

# Carregar vídeo ao vivo
cap = cv2.VideoCapture(video_path)  # Pode usar 0 para webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para tensor PyTorch
    img = torch.from_numpy(frame).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # Formato [B, C, H, W]

    # Faz a inferência
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

    # Verifica se há objetos detectados
    if pred[0] is not None:
        print("⚠️ Objeto cortante detectado! Enviando alerta...")

    # Mostra o vídeo
    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
