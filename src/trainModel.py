import os
import torch

from torch import nn
from torchvision import datasets


from crossValidation import (
    train_model,
    model_init_function,
    DATA_TRANSFORMS,
    CLASSIFIERS,
    BATCH_SIZE,
    NUM_EPOCHS,
)


if __name__ == "__main__":
    # Model Parameters
    model_type = "resnet18"
    feature_percentage_frozen = 0
    classifier_type = 0

    save_dir = (
        os.getcwd()
        + "/WasteClassification/data/models/"
        + model_type
        + "_ctype{}".format(classifier_type)
        + "_f{}".format(str(feature_percentage_frozen).replace(".", ""))
        + ".pt"
    )

    data_dir = os.getcwd() + "/WasteClassification/data/classification/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = {
        x: datasets.ImageFolder(data_dir + x, DATA_TRANSFORMS[x])
        for x in ["train", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    data_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=10
        )
        for x in ["train", "test"]
    }

    print(
        "-" * 15,
        "Start training {} model, feature_percentage_frozen = {}".format(
            model_type, feature_percentage_frozen
        ),
        "-" * 15,
    )
    print("Classifier model type:", classifier_type)
    print("Classifier model structure:", CLASSIFIERS[classifier_type][0])
    model, e, t = train_model(
        *model_init_function(
            model_type,
            CLASSIFIERS[classifier_type][0],
            CLASSIFIERS[classifier_type][1],
            device,
            feature_percentage_frozen=feature_percentage_frozen,
        ),
        data_loaders,
        dataset_sizes,
        device,
        max_epochs=NUM_EPOCHS,
        patience=10,
    )

    print("time elapsed", t)
    print("epochs trained", e)

    torch.save(model.state_dict(), save_dir)
