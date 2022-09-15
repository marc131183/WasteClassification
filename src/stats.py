from importlib.resources import path
import os


def countImageFiles():
    path = "data/cleaned/"
    img_per_folder = [
        (elem, len(os.listdir(path + "/" + elem))) for elem in os.listdir(path)
    ]
    img_per_folder = sorted(img_per_folder, key=lambda x: x[1], reverse=True)
    total_images = sum([value for key, value in img_per_folder])
    print("Total number of collected images: {}".format(total_images))
    for key, value in img_per_folder:
        print("  ->{}: {}".format(key, value))


if __name__ == "__main__":
    countImageFiles()
