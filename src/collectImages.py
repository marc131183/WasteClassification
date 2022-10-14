import os
import time
from PIL import Image, ImageDraw, ImageFont

from IO import Camera, ImageViewer

if __name__ == "__main__":
    FOLDER = os.getcwd() + "/"
    COUNTDOWN_VALUE = 5  # number of seconds between shots

    class_name = input("Enter class name: ")
    save_dir = FOLDER + "data/unlabelled/" + class_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        index = 0
    else:
        try:
            index = max([int(elem[:-4]) for elem in os.listdir(save_dir)])
        except (ValueError):
            index = 0

    cam = Camera(1280, 720)
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
            index += 1
            cam.savePicture(save_dir + "/{}.png".format(index))
