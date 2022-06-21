import os
import sys
import time
import keyboard
from PIL import Image, ImageDraw, ImageFont

import RPi.GPIO as GPIO

from IO import Camera, ImageViewer

if __name__ == "__main__":
    FOLDER = "/home/saadjahangir/Code/WasteClassification/"
    COUNTDOWN_VALUE = 4  # number of seconds between shots

    class_name = input("Enter class name: ")
    save_dir = FOLDER + "data/unlabeled/" + class_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        index = 0
    else:
        try:
            index = max([int(elem[:-4]) for elem in os.listdir(save_dir)])
        except (ValueError):
            index = 0
        
    try:
        cam = Camera(1280, 720)
        imgV = ImageViewer()
        time.sleep(
            1
        )  # give the image viewer thread cpu, so it can save the width/height of the screen
        width, height = imgV.screen_width, imgV.screen_height

        remaining = COUNTDOWN_VALUE
        font = ImageFont.truetype("DejaVuSans.ttf", 50)
        pause = False

        while not imgV.stop_thread:

            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # if remaining != 1:
            draw.text((0, 0), str(remaining), font=font, fill=(0, 0, 0))
            imgV.setImage(img)
            remaining -= 1
            time.sleep(1)
            if not remaining:
                # img = Image.new("RGB", (width, height), (255, 255, 255))
                # imgV.setImage(img)
                # time.sleep(
                #     0.1
                # )  # make sure the image viewer thread gets cpu time to change the image before taking the picture
                remaining = COUNTDOWN_VALUE
                index += 1
                cam.savePicture(save_dir + "/{}.png".format(index))
    finally:
        GPIO.cleanup()
        imgV.stop_thread = True
        os.system('xdg-open "%s"' % save_dir)
