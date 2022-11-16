import os
import time
from PIL import Image, ImageDraw, ImageFont

from IO import Camera, ImageViewer
from classify import ModelManager

if __name__ == "__main__":
    FOLDER = os.getcwd() + "/"
    COUNTDOWN_VALUE = 3  # number of seconds between shots

    save_path = FOLDER + "data/sample_image.png"

    cam = Camera(1280, 720)
    model = ModelManager(FOLDER + "data/models/resnet50_ctype0_f06.pt")
    imgV = ImageViewer()
    width, height = imgV.screen_width, imgV.screen_height

    remaining = COUNTDOWN_VALUE
    font = ImageFont.truetype("DejaVuSans.ttf", 50)

    while not imgV.stop:
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), str(remaining), font=font, fill=(0, 0, 0))
        imgV.setImage(img)
        remaining -= 1
        time.sleep(1)
        if not remaining:
            img = Image.new("RGB", (width, height), (255, 255, 255))
            imgV.setImage(img)
            remaining = COUNTDOWN_VALUE
            cam.savePicture(save_path)
            pred_label = model.classifyImage(save_path, crop=True)

            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), pred_label, font=font, fill=(0, 0, 0))
            imgV.setImage(img)
            for _ in range(20):
                imgV.setImage(img)
                time.sleep(0.1)
