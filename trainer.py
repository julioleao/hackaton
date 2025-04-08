import yaml
from yolov5 import train


from configs import DATA_PATH, DEVICE, IMG_SIZE, TRAIN_PATH, VAL_PATH


def create_data_yaml():
    data = {
        "train": TRAIN_PATH,
        "val": VAL_PATH,
        "nc": 1,
        "names": ["knife"],
    }

    # Criando o arquivo data.yaml
    with open(DATA_PATH, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    create_data_yaml()
    train.run(
        imgsz=IMG_SIZE,
        data=DATA_PATH,
        epochs=4,
        batch_size=4,
        weights="yolov5l.pt",
        cache=False,
        device=DEVICE,
        project="runs/train",
        name="exp",
        exist_ok=True,
    )
