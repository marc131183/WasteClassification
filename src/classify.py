import torch
import os
import time
import numpy as np

from PIL import Image
from crossValidation import (
    model_init_function,
    DATA_TRANSFORMS,
    CLASSIFIERS,
    compute_confusion_matrix,
)


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
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        return self.class_names[output[0]]


if __name__ == "__main__":
    model = ModelManager(os.getcwd() + "/data/models/resnet50_ctype0_f06.pt")
    print("finished loading")

    y_true = []
    y_pred = []
    start_time = time.time()
    directory = os.getcwd() + "/data/classification/test/"
    for folder in os.listdir(directory):
        temp_dir = directory + folder + "/"
        for path in os.listdir(temp_dir):
            y_pred.append(
                model.class_names.index(
                    model.classifyImage(temp_dir + path, crop=False)
                )
            )
            y_true.append(
                model.class_names.index(
                    folder if folder in ["7133", "7055", "7051", "7042"] else "others"
                )
            )
        print("finished folder {}".format(folder))
    print("accuracy", sum(x == y for x, y in zip(y_true, y_pred)) / len(y_true))
    print("confusion matrix")
    print(compute_confusion_matrix(y_true, y_pred))
    print("--- %s seconds ---" % (time.time() - start_time))
