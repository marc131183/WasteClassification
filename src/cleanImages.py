import os
import pygame
import numpy as np
from PIL import Image


def updateImageNames(path: str) -> None:
    """
    Input: path to a class folder
    Restores order of images (their names) to go from 1..n, in case there were some numbers missing
    """
    if path[-1] != "/":
        path += "/"
    names = sorted([int(elem[:-4]) for elem in os.listdir(path)])
    if len(names) != names[-1]:
        # first rename them into a dummy name (if we don't do this, then we might accidentally rename an image
        # into an already existing name and thus delete the latter image)
        for name in names:
            os.rename(path + str(name) + ".png", path + str(name) + "_temp.png")
        # rename dummy names into actual ones
        for i, name in enumerate(names):
            os.rename(path + str(name) + "_temp.png", path + str(i + 1) + ".png")


def deleteImages(path: str) -> None:
    """
    Input: path to a class folder
    creates an image viewer to make deletion of bad images easier
    controls:
    left/right arrow key: previous/next image
    space: change if image should be deleted
    esc/X: end program, deleting all marked images
    """
    if path[-1] != "/":
        path += "/"

    sorted_image_paths = [
        path + str(elem) + ".png"
        for elem in sorted([int(elem[:-4]) for elem in os.listdir(path)])
    ]
    keep_image = [True for i in range(len(sorted_image_paths))]
    current_img = 0

    pygame.init()
    pygame.display.set_caption("Image Viewer")
    white = (255, 255, 255)
    display_surface = pygame.display.set_mode((1500, 720))
    my_font = pygame.font.SysFont("Comic Sans MS", 45)
    image = pygame.image.load(sorted_image_paths[current_img])

    while True:
        display_surface.fill(white)
        display_surface.blit(image, (0, 0))
        text_surface = my_font.render(
            str(
                sorted_image_paths[current_img][
                    sorted_image_paths[current_img].rindex("/") :
                ]
            ),
            False,
            (0, 0, 0),
        )
        display_surface.blit(text_surface, (1280, 10))
        text_surface = my_font.render("Keep Image?", False, (0, 0, 0))
        display_surface.blit(text_surface, (1280, 100))
        text_surface = my_font.render(
            "Yes" if keep_image[current_img] else "No", False, (0, 0, 0)
        )
        display_surface.blit(text_surface, (1350, 200))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for i, elem in enumerate(keep_image):
                    if not elem:
                        os.remove(sorted_image_paths[i])
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_img = (current_img + 1) % len(sorted_image_paths)
                    image = pygame.image.load(sorted_image_paths[current_img])
                elif event.key == pygame.K_LEFT:
                    current_img = (current_img - 1) % len(sorted_image_paths)
                    image = pygame.image.load(sorted_image_paths[current_img])
                elif event.key == pygame.K_ESCAPE:
                    for i, elem in enumerate(keep_image):
                        if not elem:
                            os.remove(sorted_image_paths[i])
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_SPACE:
                    keep_image[current_img] = not keep_image[current_img]

            pygame.display.update()


def downsizeImages(
    folder: str,
    y_cut_left: int = 50,
    y_cut_right: int = 50,
    x_cut_left: int = 200,
    x_cut_right: int = 100,
):
    """downsizes all images in all subfolders of the given folder"""
    if folder[-1] != "/":
        folder += "/"
    for fold in os.listdir(folder):
        print(fold)
        total_path = folder + fold + "/"
        for image_path in os.listdir(total_path):
            img = np.array(Image.open(total_path + image_path))[
                y_cut_left:-y_cut_right, x_cut_left:-x_cut_right
            ]
            Image.fromarray(img).save(total_path + image_path)


if __name__ == "__main__":
    folder = "data/cleaned/7051"
    # deleteImages(folder)
    # updateImageNames(folder)

    # downsizeImages("data/cleaned")
