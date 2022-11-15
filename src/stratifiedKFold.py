import os
import shutil


def stratifiedKFold(y, number_of_folds):
    """it is assumed that y is sorted"""
    train_folds = [[] for i in range(number_of_folds)]
    test_folds = [[] for i in range(number_of_folds)]
    val_folds = [[] for i in range(number_of_folds)]

    first_indices = []
    last_elem_seen = None
    for i in range(len(y)):
        if y[i] != last_elem_seen:
            first_indices.append(i)
            last_elem_seen = y[i]
    first_indices.append(len(y))
    elements_per_class = [
        first_indices[i + 1] - first_indices[i] for i in range(len(first_indices) - 1)
    ]
    class_indices = [
        set([k + first_indices[i] for k in range(elements_per_class[i])])
        for i in range(len(elements_per_class))
    ]

    for i in range(number_of_folds):
        for j in range(len(elements_per_class)):
            start_test = first_indices[j] + elements_per_class[j] * i // number_of_folds
            end_test = (
                first_indices[j] + elements_per_class[j] * (i + 1) // number_of_folds
            )
            start_val = (
                first_indices[j] + elements_per_class[j] * (i + 1) // number_of_folds
            )
            end_val = (
                first_indices[j]
                + elements_per_class[j] * ((i + 2) % number_of_folds) // number_of_folds
            )

            if start_val < end_val:
                val_indices = set(range(start_val, end_val))
            else:
                val_indices = set(
                    range(start_val, first_indices[j] + elements_per_class[j])
                ).union(set(range(first_indices[j], end_val)))
            test_indices = set(range(start_test, end_test))
            train_indices = (
                class_indices[j].difference(test_indices).difference(val_indices)
            )

            train_folds[i].extend(train_indices)
            test_folds[i].extend(test_indices)
            val_folds[i].extend(val_indices)

    return [(x, y, z) for x, y, z in zip(train_folds, test_folds, val_folds)]


def createKFoldSplit(number_of_folds=5):
    dir = ""  # os.getcwd() + "/WasteClassification/"
    source = dir + "data/cleaned/"
    target = dir + "data/classification/"
    main_classes = os.listdir(source)

    paths = []
    labels = []
    for i, folder in enumerate(main_classes):
        temp_path = source + folder + "/"
        files = sorted(
            os.listdir(temp_path), key=lambda x: int(os.path.basename(x)[:-4])
        )
        paths.extend([temp_path + elem for elem in files])
        labels.extend([i] * len(files))

    main_classes = [
        elem
        if elem
        in [
            "7133",
            "7055",
            "7051",
            "7042",
        ]
        else "others"
        for elem in main_classes
    ]

    os.mkdir(target + "kFold_" + str(number_of_folds))
    for i, indices in enumerate(stratifiedKFold(labels, number_of_folds)):
        temp_path = target + "kFold_" + str(number_of_folds) + "/" + "fold_" + str(i)
        os.mkdir(temp_path)

        os.mkdir(temp_path + "/train")
        os.mkdir(temp_path + "/test")
        os.mkdir(temp_path + "/val")
        for elem in ["7133", "7055", "7051", "7042", "others"]:
            os.mkdir(temp_path + "/train/" + elem)
            os.mkdir(temp_path + "/test/" + elem)
            os.mkdir(temp_path + "/val/" + elem)

        for i, phase in zip(range(len(indices)), ["train", "test", "val"]):
            for j, index in enumerate(indices[i]):
                source_path = paths[index]
                dest_path = (
                    temp_path
                    + "/{}/".format(phase)
                    + main_classes[labels[index]]
                    + "/{}.png".format(j)
                )
                shutil.copy(source_path, dest_path)


def splitIntoTrainTest():
    base_path = os.getcwd() + "/WasteClassification/"
    path = base_path + "data/cleaned/"
    dest_train = base_path + "data/classification/train/"
    dest_test = base_path + "data/classification/test/"
    split = 0.9
    main_classes = ["7133", "7055", "7051", "7042"]
    i = 0
    j = 0
    os.mkdir(dest_train[:-1])
    os.mkdir(dest_test[:-1])
    os.mkdir(dest_train + "others")
    os.mkdir(dest_test + "others")

    for folder in os.listdir(path):
        files = os.listdir(path + folder + "/")
        files = sorted(files, key=lambda x: int(os.path.basename(x)[:-4]), reverse=True)
        if folder in main_classes:
            os.mkdir(dest_train + folder)
            for file in files[: int(len(files) * split)]:
                source_path = path + folder + "/" + file
                dest_path = dest_train + folder + "/" + file
                shutil.copy(source_path, dest_path)

            os.mkdir(dest_test + folder)
            for file in files[int(len(files) * split) :]:
                source_path = path + folder + "/" + file
                dest_path = dest_test + folder + "/" + file
                shutil.copy(source_path, dest_path)
        else:
            for file in files[: int(len(files) * split)]:
                source_path = path + folder + "/" + file
                dest_path = dest_train + "others/{}.png".format(i)
                shutil.copy(source_path, dest_path)
                i += 1

            for file in files[int(len(files) * split) :]:
                source_path = path + folder + "/" + file
                dest_path = dest_test + "others/{}.png".format(j)
                shutil.copy(source_path, dest_path)
                j += 1


if __name__ == "__main__":
    # createKFoldSplit(number_of_folds=10)
    splitIntoTrainTest()
