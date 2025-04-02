import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from configs import *
from model import Autoencoder


def trainer():
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageFolder(IMAGE_TRAINER_FOLDER, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Classes detectadas: {dataset.class_to_idx}")

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print("Iniciando treinamento...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for img, _ in train_loader:
            img = img.to(device)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, img)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "modelo_faca.pth")
    print("Modelo salvo com sucesso!")
