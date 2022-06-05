import sys
import tkinter
from PIL import Image, ImageTk

def showPIL(img):
	root = tkinter.Tk()
	w, h = root.winfo_screenwidth(), root.winfo_screenheight()
	root.overridedirect(1)
	root.geometry("%dx%d+0+0" % (w, h))
	root.focus_set()
	root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
	canvas = tkinter.Canvas(root, width = w, height = h)
	canvas.pack()
	canvas.configure(background = "black")
	imgWidth, imgHeight = img.size
	if imgWidth > w or imgHeight > h:
		ratio = min(w/imgWidth, h/imgHeight)
		imgWidth = int(imgWidth*ratio)
		imgHeight = int(imgHeight*ratio)
		img = img.resize((imgWidth,imgHeight), Image.ANTIALIAS)
	image = ImageTk.PhotoImage(img)
	imagesprite = canvas.create_image(w/2,h/2,image=image)
	root.mainloop()


if __name__ == "__main__":
	FOLDER = "/home/saadjahangir/Code/WasteClassification/"
	img = cv2.imread(FOLDER + "data/pic.png")
	

