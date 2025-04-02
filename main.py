import os

from tester import tester
from trainer import trainer

if not os.path.exists("modelo_faca.pth"):
    trainer()

if __name__ == "__main__":
    tester()

