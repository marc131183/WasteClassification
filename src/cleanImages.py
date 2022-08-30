# file for making the cleaning of images easier

import torch
from PIL import Image
from yolov5 import YoloV5Model


if __name__ == "__main__":
    yolo = YoloV5Model("yolov5s.pt")

    print(yolo.classifyImage("data/unlabelled/7051/1.png"))
