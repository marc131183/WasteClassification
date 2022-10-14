import numpy as np

from PIL import Image

if __name__ == "__main__":
    hspace = np.ones((620, 30, 3), dtype=np.uint8) * 255

    img1 = np.array(Image.open("data/cleaned/7055/96.png"))
    img2 = np.array(Image.open("data/cleaned/7051/7.png"))
    row1 = np.hstack([img1, hspace, img2])

    img3 = np.array(Image.open("data/cleaned/7133/62.png"))
    img4 = np.array(Image.open("data/cleaned/7042/258.png"))
    row2 = np.hstack([img3, hspace, img4])

    img5 = np.array(Image.open("data/cleaned/7023/34.png"))
    img6 = np.array(Image.open("data/cleaned/7134/35.png"))
    row3 = np.hstack([img5, hspace, img6])

    vspace = np.ones((30, row1.shape[1], 3), dtype=np.uint8) * 255

    final_img = np.vstack([row1, vspace, row2, vspace, row3])
    Image.fromarray(final_img).save("overview.png")
