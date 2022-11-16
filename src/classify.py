import torch
import os
import time
import numpy as np

from PIL import Image
from crossValidation import model_init_function, DATA_TRANSFORMS, CLASSIFIERS


class ModelManager:
    def __init__(self, path: str) -> None:
        self.class_names = ["7042", "7051", "7055", "7133", "others"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        temp = os.path.basename(path)[:-3].split("_")
        model_type = temp[0]
        classifier_type = int(temp[1][-1])
        # freeze percentage doesn't matter during testing

        self.model, _, _, _ = model_init_function(
            model_type,
            CLASSIFIERS[classifier_type][0],
            CLASSIFIERS[classifier_type][1],
            self.device,
        )
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.data_transforms = DATA_TRANSFORMS["test"]

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
        output = (torch.max(output, 1))[1].data.cpu().numpy()
        return self.class_names[output[0]]


if __name__ == "__main__":
    model = ModelManager(os.getcwd() + "/data/models/resnet50_ctype0_f06.pt")
    print("finished loading")

    # took 348.688 seconds/ 343.219 seconds/ 329.918 seconds for 3217 images
    correct = 0
    total = 0
    start_time = time.time()
    directory = os.getcwd() + "data/cleaned/"
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
