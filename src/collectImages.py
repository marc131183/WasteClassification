import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import RPi.GPIO as GPIO

from IO import Camera, ImageViewer

if __name__ == "__main__":
    FOLDER = "/home/saadjahangir/Code/WasteClassification/"
    COUNTDOWN_VALUE = 5  # number of seconds between shots

    class_name = input("Enter class name: ")
    save_dir = FOLDER + "data/unlabelled/" + class_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        index = 0
    else:
        index = int(max([elem for elem in os.listdir(save_dir)])[:-4])

    try:
        cam = Camera()
        imgV = ImageViewer()
        time.sleep(1)
        width, height = imgV.screen_width, imgV.screen_height

        remaining = COUNTDOWN_VALUE
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
        while True:
            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), str(remaining), font=font, fill=(0, 0, 0))
            img.show()
            imgV.setImage(img)
            remaining -= 1
            if not remaining:
                remaining = COUNTDOWN_VALUE
                index += 1
                cam.savePicture(save_dir + "/{}.png".format(index))
            time.sleep(1)
    finally:
        GPIO.cleanup()
