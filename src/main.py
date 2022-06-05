#!/usr/bin/env python3

import time
import numpy as np
import RPi.GPIO as GPIO

from PIL import Image

from IO import *
from yolov5 import YoloV5Model

if __name__ == "__main__":
    TAKE_IMG_INTERVAL = 0  # seconds
    FOLDER = "/home/saadjahangir/Code/WasteClassification/"
    DRAW_ON_IMG = True

    try:
        mot = MotionSensor(18)
        cam = Camera()
        prj = Projector()
        prj.turn_on()
        model = YoloV5Model(FOLDER + "data/yolov5/yolov5s.pt")
        imgV = ImageViewer()
        print("Ready to operate")

        while True:
            value = mot.read()

            if value == GPIO.HIGH:
                print("Taking picture")
                img_path = FOLDER + "/data/pic.png"
                cam.savePicture(img_path)
                df = model.classifyImage(img_path)
                if DRAW_ON_IMG:
                    img = np.array(Image.open(img_path))
                else:
                    img = np.uint8(np.ones((400, 400)) * 255)
                for row in df.iterrows():
                    min_row, min_col, max_row, max_col = (
                        int(row[1]["ymin"]),
                        int(row[1]["xmin"]),
                        int(row[1]["ymax"]),
                        int(row[1]["xmax"]),
                    )
                    img[min_row:max_row, min_col - 2 : min_col + 3] = 0
                    img[min_row:max_row, max_col - 2 : max_col + 3] = 0
                    img[min_row - 2 : min_row + 3, min_col:max_col] = 0
                    img[max_row - 2 : max_row + 3, min_col:max_col] = 0
                img = Image.fromarray(img)
                img = img.resize(
                    (imgV.screen_width, imgV.screen_height), Image.ANTIALIAS
                )
                imgV.setImage(img)
                time.sleep(TAKE_IMG_INTERVAL)
    finally:
        GPIO.cleanup()
        prj.turn_off()
