import cv2
import socket
import tkinter
import threading
from PIL import Image, ImageTk
from git import Repo

import RPi.GPIO as GPIO


def pushChanges():
    repo = Repo(".")
    repo.git.add(update=True)
    repo.index.commit("automatic push from jetson nano")
    origin = repo.remote(name="origin")
    origin.push()


class MotionSensor:
    def __init__(self, input_pin, pin_mode=GPIO.BOARD) -> None:
        self.pin = input_pin

        GPIO.setmode(pin_mode)
        GPIO.setup(self.pin, GPIO.IN)

    def read(self):
        return GPIO.input(self.pin)


class Camera:
    def __init__(self, width, height) -> None:
        # does not work for certain resolutions
        # TODO: look into why that is
        self.width = width
        self.height = height

    def savePicture(self, path: str) -> None:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        worked, frame = cam.read()
        cam.release()
        cv2.imwrite(path, frame)

    def showPicture(self, milliseconds: int = 3000) -> None:
        cam = cv2.VideoCapture(0)
        worked, frame = cam.read()
        cam.release()

        cv2.imshow("Image", frame)
        key = cv2.waitKey(milliseconds)
        if key == 27:  # ESC pressed
            return


class Projector:
    def __init__(self) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(1)
        self.working = True
        try:
            self.s.connect(("192.168.0.10", 7142))
        except:
            print("Socket connection error")
            self.s.close()
            self.working = False

    def send_command(self, cmd):
        try:
            self.s.send(cmd)
            try:
                data = self.s.recv(100)
                print("Respose from projector:")
                print(data)
            except socket.timeout:
                pass
        except:
            print("projector communication error")
            self.s.close()
            self.working = False

    def turn_on(self):
        self.send_command(b"\x02\x00\x00\x00\x00\x02")

    def turn_off(self):
        self.send_command(b"\x02\x01\x00\x00\x00\x03")

    def info(self):
        self.send_command(b"\x03\x8A\x00\x00\x00\x8D")


def stop(e, imgV):
    e.widget.withdraw()
    e.widget.quit()
    imgV.stop_thread = True


class ImageViewer:
    def __init__(self) -> None:
        self.thread = None
        self.image_changed = False
        self.image = None

        self.stop_thread = False
        self.thread = threading.Thread(target=self.thread_mainloop, daemon=True)
        self.thread.start()

    def thread_mainloop(self):
        root = tkinter.Tk()
        self.screen_width, self.screen_height = (
            root.winfo_screenwidth(),
            root.winfo_screenheight(),
        )
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda e: stop(e, self))
        canvas = tkinter.Canvas(
            root, width=self.screen_width, height=self.screen_height
        )
        canvas.pack()
        canvas.configure(background="white")
        imagesprite = None

        while not self.stop_thread:
            root.update_idletasks()
            root.update()

            if self.image_changed:
                if imagesprite:
                    canvas.delete(imagesprite)
                image = ImageTk.PhotoImage(self.image)
                imagesprite = canvas.create_image(
                    self.screen_width / 2, self.screen_height / 2, image=image
                )
                self.image_changed = False

    def setImage(self, img: Image.Image):
        imgWidth, imgHeight = img.size

        if imgWidth > self.screen_width or imgHeight > self.screen_height:
            ratio = min(self.screen_width / imgWidth, self.screen_height / imgHeight)
            imgWidth = int(imgWidth * ratio)
            imgHeight = int(imgHeight * ratio)
            img = img.resize((imgWidth, imgHeight), Image.ANTIALIAS)

        self.image = img
        self.image_changed = True
