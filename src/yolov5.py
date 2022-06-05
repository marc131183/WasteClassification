from PIL import Image
import torch


class YoloV5Model:
    def __init__(self, weights_path: str) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
        # model = torch.hub.load(".", "custom", path=weights_path, source="local")

    def classifyImage(self, path: str) -> None:
        img = Image.open(path)
        results = self.model(img)
        return results.pandas().xyxy[0]
