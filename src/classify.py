import torch

from PIL import Image
from torchvision import transforms

from IO import Camera


class ModelManager:
    def __init__(self) -> None:
        self.class_names = [
            "7011",
            "7023",
            "7042",
            "7051",
            "7055",
            "7123",
            "7133",
            "7134",
            "7151",
            "7152",
        ]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load("data/models/resnet18.pt")
        self.model.eval()
        self.data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def classifyImage(self, img_path: str, cropped: bool = False) -> str:
        img = Image.open(img_path)

        if not cropped:
            y_cut_left: int = 50
            y_cut_right: int = 50
            x_cut_left: int = 200
            x_cut_right: int = 100
            img = img[y_cut_left:-y_cut_right, x_cut_left:-x_cut_right]

        img = self.data_transform(img)[None, :]
        img = img.to(self.device)
        output = self.model(img)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        return self.class_names[output[0]]
