import sys
import tkinter
import numpy as np
import time
import threading
from PIL import Image, ImageTk

IMG = Image.fromarray(np.uint8(np.ones((400, 400)) * 0))
IMG_CHANGED = False


def showPIL(pilImage):
    global IMG_CHANGED
    global IMG
    root = tkinter.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (w, h))
    root.focus_set()
    root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
    canvas = tkinter.Canvas(root, width=w, height=h)
    canvas.pack()
    canvas.configure(background="white")
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w / imgWidth, h / imgHeight)
        imgWidth = int(imgWidth * ratio)
        imgHeight = int(imgHeight * ratio)
        pilImage = pilImage.resize((imgWidth, imgHeight), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(pilImage)
    imagesprite = canvas.create_image(w / 2, h / 2, image=image)

    while True:
        root.update_idletasks()
        root.update()
        # is this safe in terms of threading? might need lock or smth?
        if IMG_CHANGED:
            canvas.delete(imagesprite)
            image = ImageTk.PhotoImage(IMG)
            imagesprite = canvas.create_image(w / 2, h / 2, image=image)
            IMG_CHANGED = False


if __name__ == "__main__":
    # https://realpython.com/intro-to-python-threading/
    whiteImage = Image.fromarray(np.uint8(np.ones((400, 400)) * 50))
    x = threading.Thread(target=showPIL, args=(whiteImage,), daemon=True)
    x.start()
    time.sleep(3)
    IMG_CHANGED = True
    time.sleep(3)
    print(IMG_CHANGED)
