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


def mergeAllFolders(parent_path_into: str, parent_path_from: str) -> None:
    """
    Calls mergeFolders for all pairs of folders with the same name in the given directories
    """
    if parent_path_into[-1] != "/":
        parent_path_into += "/"
    if parent_path_from[-1] != "/":
        parent_path_from += "/"

    folders_into = set(os.listdir(parent_path_into))
    folders_from = set(os.listdir(parent_path_from))

    for sub_folder in folders_into.intersection(folders_from):
        mergeFolders(parent_path_into + sub_folder, parent_path_from + sub_folder)


def mergeFolders(folder_to_merge_into: str, folder_to_merge_from: str) -> None:
    """
    Merges the image files of two folders
    Images from folder_to_merge_into keep their names (1..n)
    Images from folder_to_merge_from get renamed to (n+1..m) and are moved to the other folder (keeps their order)
    deletes folder_to_merge_from in the end
    """
    if folder_to_merge_into[-1] != "/":
        folder_to_merge_into += "/"
    if folder_to_merge_from[-1] != "/":
        folder_to_merge_from += "/"

    highest_index = max([int(elem[:-4]) for elem in os.listdir(folder_to_merge_into)])
    names = sorted([int(elem[:-4]) for elem in os.listdir(folder_to_merge_from)])

    for i, name in enumerate(names):
        os.rename(
            folder_to_merge_from + str(name) + ".png",
            folder_to_merge_into + str(highest_index + i + 1) + ".png",
        )

    os.rmdir(folder_to_merge_from[:-1])


def deleteImages(path: str) -> None:
    """
    Input: path to a class folder
    creates an image viewer to make deletion of bad images easier
    calls updateImagesNames when finished
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
        text_surface = my_font.render(
            "#total {}".format(len(sorted_image_paths)), False, (0, 0, 0)
        )
        display_surface.blit(text_surface, (1280, 400))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for i, elem in enumerate(keep_image):
                    if not elem:
                        os.remove(sorted_image_paths[i])
                pygame.quit()
                updateImageNames(path)
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
                    updateImageNames(path)
                    quit()
                elif event.key == pygame.K_SPACE:
                    keep_image[current_img] = not keep_image[current_img]

            pygame.display.update()


def cutImages(
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


def splitIntoTrainTest():
    path = "data/cleaned/"
    dest_train = "data/classification/train/"
    dest_test = "data/classification/val/"
    split = 0.8  # 80% train, 20% test

    for folder in os.listdir(path):
        files = os.listdir(path + folder + "/")
        os.mkdir(dest_train + folder)
        for file in files[: int(len(files) * split)]:
            source_path = path + folder + "/" + file
            dest_path = dest_train + folder + "/" + file
            os.popen("cp {} {}".format(source_path, dest_path))

        os.mkdir(dest_test + folder)
        for file in files[int(len(files) * split) :]:
            source_path = path + folder + "/" + file
            dest_path = dest_test + folder + "/" + file
            os.popen("cp {} {}".format(source_path, dest_path))


if __name__ == "__main__":
    folder = "data/unlabelled/7042"
    # splitIntoTrainTest()
    # deleteImages(folder)
    # updateImageNames(folder)

    cutImages("data/unlabelled")
    # mergeFolders("data/unlabelled/7051", "data/unlabelled/7025")
    # mergeAllFolders("data/cleaned/", "data/unlabelled/")
