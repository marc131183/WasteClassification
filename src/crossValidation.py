import os
import torch
import numpy as np
import time
import copy

np.random.seed(42)
torch.manual_seed(42)

from torch import nn, optim
from torchvision import models, transforms, datasets


# define parameters
NUM_CLASSES = 5
NUM_EPOCHS = 25
NUMBER_OF_FOLDS = 10
BATCH_SIZE = 4

DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def compute_confusion_matrix(y_true, y_pred):
    num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
    confusion_matrix = np.zeros((num_classes, num_classes))

    for x, y in zip(y_true, y_pred):
        confusion_matrix[x, y] += 1

    return confusion_matrix


def evaluate_model(model, data_loaders, device):
    y_pred = []
    y_true = []

    # iterate over test data
    with torch.no_grad():
        for inputs, labels in data_loaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    return sum([x == y for x, y in zip(y_true, y_pred)]) / len(
        y_true
    ), compute_confusion_matrix(y_true, y_pred)


def crossValidateModel(init_function, train_function, device, number_of_folds):
    all_acc = []

    # base_dir = (
    #     os.getcwd()
    #     + "/WasteClassification/data/classification/kFold_{}/".format(number_of_folds)
    # )

    base_dir = "data/classification/kFold_{}/".format(number_of_folds)

    confusion_matrix = None
    conf_initialized = False

    for i in range(number_of_folds):
        print("-" * 15, "Started working on Fold {}".format(i + 1), "-" * 15)
        temp_dir = base_dir + "fold_{}".format(i)

        image_datasets = {
            x: datasets.ImageFolder(os.path.join(temp_dir, x), DATA_TRANSFORMS[x])
            for x in ["train", "test"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
        data_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=10
            )
            for x in ["train", "test"]
        }

        model = train_function(*init_function(), data_loaders, dataset_sizes)

        acc, conf = evaluate_model(model, data_loaders, device)

        all_acc.append(acc)
        if conf_initialized:
            confusion_matrix = confusion_matrix + conf
        else:
            confusion_matrix = conf
            conf_initialized = True

    return all_acc, confusion_matrix


def train_model_optional_validation(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loaders,
    dataset_sizes,
    device,
    early_stopping_subset_ratio=0.1,
    early_stopping_tolerance=5,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    phases = ["train"]
    if "test" in data_loaders:
        phases.append("test")

    # select subset of batches of training data to do early stopping on
    early_stopping_subset_indices = np.arange(len(data_loaders["train"]))
    np.random.shuffle(early_stopping_subset_indices)
    early_stopping_subset_indices = early_stopping_subset_indices[
        : int(len(early_stopping_subset_indices) * early_stopping_subset_ratio)
    ]
    is_part_of_indices = np.zeros((len(data_loaders["train"])), dtype=bool)
    is_part_of_indices[early_stopping_subset_indices] = True
    no_improvement_since = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and testing phase
        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            subset_running_loss = 0.0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == "train" and is_part_of_indices[i]:
                    subset_running_loss += loss.item() * inputs.size(0)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
                subset_running_loss = subset_running_loss / len(
                    early_stopping_subset_indices
                )
                print(f"Subset loss : {subset_running_loss:.4f}")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                no_improvement_since += 1

                if subset_running_loss < best_loss:
                    best_loss = subset_running_loss
                    no_improvement_since = 0
                    # deep copy the model
                    best_model_wts = copy.deepcopy(model.state_dict())

                if no_improvement_since > early_stopping_tolerance:
                    print("-" * 5, "Early stopping", "-" * 5)
                    time_elapsed = time.time() - since
                    print(
                        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
                    )
                    model.load_state_dict(best_model_wts)
                    return model

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loaders,
    dataset_sizes,
    device,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and testing phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def model_init_function(
    model_architecture,
    final_layers,
    final_layers_in,
    device,
    feature_extractor=False,
    learn_rate=0.001,
    momentum=0.9,
):
    available_architectures = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "alexnet": models.alexnet,
        "vgg": models.vgg11,
    }

    class CustomModel(nn.Module):
        def __init__(self) -> None:
            super(CustomModel, self).__init__()

            self.network = available_architectures[model_architecture](pretrained=True)
            if feature_extractor:
                for param in self.network.parameters():
                    param.requires_grad = False
            if "resnet" in model_architecture:
                self.network.fc = nn.Sequential(
                    nn.Linear(self.network.fc.in_features, final_layers_in),
                    *final_layers,
                )
            elif "alexnet" in model_architecture or "vgg" in model_architecture:
                self.network.classifier[-1] = nn.Sequential(
                    nn.Linear(self.network.classifier[-1].in_features, final_layers_in),
                    *final_layers,
                )

        def forward(self, x):
            x = self.network(x)
            return x

    model = CustomModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learn_rate,
        momentum=momentum,
    )

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, exp_lr_scheduler


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accuracy, confusion_matrix = crossValidateModel(
        lambda: model_init_function(
            "alexnet",
            [nn.ReLU(), nn.Linear(100, NUM_CLASSES)],
            100,
            device,
            feature_extractor=True,
        ),
        lambda a, b, c, d, e, f: train_model_optional_validation(
            a, b, c, d, e, f, device, num_epochs=NUM_EPOCHS
        ),
        device,
        NUMBER_OF_FOLDS,
    )
    print("-" * 30)
    print("accuracy", accuracy)
    print("mean accuracy", np.mean(accuracy))
    print("stddev accuracy", np.std(accuracy))
    print("confusion matrix")
    print(confusion_matrix)
