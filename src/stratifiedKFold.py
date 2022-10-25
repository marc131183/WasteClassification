import os
import shutil


def stratifiedKFold(y, number_of_folds):
    """it is assumed that y is sorted"""
    train_folds = [[] for i in range(number_of_folds)]
    test_folds = [[] for i in range(number_of_folds)]

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

            test_indices = set(range(start_test, end_test))
            train_indices = class_indices[j].difference(test_indices)

            train_folds[i].extend(train_indices)
            test_folds[i].extend(test_indices)

    return [(x, y) for x, y in zip(train_folds, test_folds)]


def createKFoldSplit(number_of_folds=5):
    dir = os.getcwd() + "/WasteClassification/"
    source = dir + "data/cleaned/"
    target = dir + "data/classification/"
    main_classes = os.listdir(source)  # ["7133", "7055", "7051", "7042"]

    paths = []
    labels = []
    for folder in os.listdir(source):
        temp_path = source + folder + "/"
        files = sorted(
            os.listdir(temp_path), key=lambda x: int(os.path.basename(x)[:-4])
        )
        paths.extend([temp_path + elem for elem in files])
        if folder in main_classes:
            labels.extend([main_classes.index(folder)] * len(files))
        else:
            labels.extend([len(main_classes)] * len(files))

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
    for i, (train_index, test_index) in enumerate(
        stratifiedKFold(labels, number_of_folds)
    ):
        temp_path = target + "kFold_" + str(number_of_folds) + "/" + "fold_" + str(i)
        os.mkdir(temp_path)

        os.mkdir(temp_path + "/train")
        os.mkdir(temp_path + "/test")
        for elem in ["7133", "7055", "7051", "7042", "others"]:
            os.mkdir(temp_path + "/train/" + str(elem))
            os.mkdir(temp_path + "/test/" + str(elem))

        for phase in ["train", "test"]:
            for j, index in enumerate(train_index if phase == "train" else test_index):
                source_path = paths[index]
                dest_path = (
                    temp_path
                    + "/{}/".format(phase)
                    + main_classes[labels[index]]
                    + "/{}.png".format(j)
                )
                shutil.copy(source_path, dest_path)


if __name__ == "__main__":
    createKFoldSplit(number_of_folds=10)
