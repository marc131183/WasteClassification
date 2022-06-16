#!/usr/bin/env python3

import cv2
import time

cam = cv2.VideoCapture(0)


for i in range(100):
    worked, frame = cam.read()

    # cv2.imshow('img', frame) m

    cv2.imwrite("data/image{}.png".format(i), frame)
    time.sleep(1)

cam.release()
cv2.destroyAllWindows()
