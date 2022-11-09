import os
import torch

from torch import nn
from torchvision import datasets


from crossValidation import (
    train_model_optional_validation,
    model_init_function,
    DATA_TRANSFORMS,
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_CLASSES,
)


if __name__ == "__main__":
    model_type = "resnet50"
    feature_extractor = False
    save_dir = (
        os.getcwd()
        + "/WasteClassification/data/models/"
        + model_type
        + ("_feat" if feature_extractor else "")
        + ".pt"
    )

    data_dir = os.getcwd() + "/WasteClassification/data/classification/all"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = datasets.ImageFolder(data_dir, DATA_TRANSFORMS["train"])
    dataset_sizes = {x: len(image_datasets) for x in ["train"]}
    data_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=10
        )
        for x in ["train"]
    }

    print(
        "-" * 15,
        "Start training {}{} model".format(
            model_type, "_feat" if feature_extractor else ""
        ),
        "-" * 15,
    )
    model = train_model_optional_validation(
        *model_init_function(
            model_type, [], NUM_CLASSES, device, feature_extractor=feature_extractor
        ),
        data_loaders,
        dataset_sizes,
        device,
        num_epochs=NUM_EPOCHS
    )

    torch.save(model.state_dict(), save_dir)
