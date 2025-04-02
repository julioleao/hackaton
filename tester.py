import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
from model import Autoencoder
from configs import *


def tester():
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("modelo_faca.pth", map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        exit()

    cv2.namedWindow("Detecção de Facas", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed = model(frame_tensor)

        mse_loss = nn.functional.mse_loss(reconstructed, frame_tensor)

        threshold = 0.001
        has_knife = mse_loss.item() < threshold
        print(mse_loss.item())

        label = "FACA DETECTADA!" if has_knife else "Nenhuma faca detectada"
        color = (0, 0, 255) if has_knife else (0, 255, 0)

        cv2.putText(
            frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )
        cv2.imshow("Detecção de Facas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
