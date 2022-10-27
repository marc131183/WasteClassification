import torch
import os
import time
import numpy as np

from PIL import Image
from torchvision import transforms


class ModelManager:
    def __init__(self) -> None:
        self.class_names = ["7042", "7051", "7055", "7133", "other"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            "data/models/models_new/resnet18.pt", map_location=self.device
        )
        self.model.eval()
        self.data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def classifyImage(self, img_path: str, crop: bool = False) -> str:
        img = Image.open(img_path)

        if crop:
            y_cut_left: int = 50
            y_cut_right: int = 50
            x_cut_left: int = 200
            x_cut_right: int = 100
            img = Image.fromarray(
                np.array(img)[y_cut_left:-y_cut_right, x_cut_left:-x_cut_right]
            )

        img = self.data_transforms(img)[None, :]
        img = img.to(self.device)
        output = self.model(img)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        return self.class_names[output[0]]


if __name__ == "__main__":
    model = ModelManager()
    print("finished loading")

    # took 348.688 seconds/ 343.219 seconds/ 329.918 seconds for 3217 images
    correct = 0
    total = 0
    start_time = time.time()
    directory = "data/cleaned/"
    for folder in os.listdir(directory):
        temp_dir = directory + folder + "/"
        for path in os.listdir(temp_dir):
            label_pred = model.classifyImage(temp_dir + path, crop=False)
            correct += label_pred == (
                folder if folder in ["7133", "7055", "7051", "7042"] else "others"
            )
            total += 1
        print("finished folder {}".format(folder))
    print(correct / total)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(model.classifyImage("data/cleaned/7055/2.png", cropped=True))
    # print(model.classifyImage("data/cleaned/7055/3.png", cropped=True))
