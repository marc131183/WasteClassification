import cv2
import time
import os

import RPi.GPIO as GPIO


class MotionSensor:
    def __init__(self, input_pin, pin_mode=GPIO.BOARD) -> None:
        self.pin = input_pin

        GPIO.setmode(pin_mode)
        GPIO.setup(self.pin, GPIO.IN)

    def read(self):
        return GPIO.input(self.pin)


class Camera:
    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(0)

    def savePicture(self, path: str):
        worked, frame = self.cam.read()
        cv2.imwrite(path, frame)

    def showPicture(self, milliseconds: int = 3000):
        worked, frame = self.cam.read()

        cv2.imshow("Image", frame)
        key = cv2.waitKey(milliseconds)
        if key == 27:  # ESC pressed
            return


if __name__ == "__main__":
    TAKE_IMG_INTERVAL = 3  # seconds
    SAVE_FOLDER = "/home/saadjahangir/Code/WasteClassification/"

    try:
        mot = MotionSensor(18)
        cam = Camera()

        while True:
            value = mot.read()

            if value == GPIO.HIGH:
                print("Taking picture")
                cam.savePicture(SAVE_FOLDER + "/data/pic.png")
                time.sleep(TAKE_IMG_INTERVAL)
    finally:
        GPIO.cleanup()
