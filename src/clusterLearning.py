import os
import torch
import time
import copy

from torch import nn, optim
from torchvision import models, transforms, datasets


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

    return sum(y_true == y_pred) / len(y_true)


def crossValidateModel(init_function, train_function, device, number_of_folds):
    all_acc = []

    base_dir = "data/classification/kFold_{}/".format(number_of_folds)

    for i in range(number_of_folds):
        temp_dir = base_dir + "fold_{}".format(i)

        image_datasets = {
            x: datasets.ImageFolder(os.path.join(temp_dir, x), data_transforms[x])
            for x in ["train", "test"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
        data_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=6
            )
            for x in ["train", "test"]
        }

        model = train_function(*init_function(), data_loaders, dataset_sizes)

        all_acc.append(evaluate_model(model, data_loaders, device))

    return all_acc


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


def resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, exp_lr_scheduler


if __name__ == "__main__":
    # define parameters
    num_classes = 5
    num_epochs = 15
    number_of_folds = 10
    batch_size = 4

    data_transforms = {
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accuracy = crossValidateModel(
        resnet18,
        lambda a, b, c, d, e, f: train_model(
            a, b, c, d, e, f, device, num_epochs=num_epochs
        ),
        device,
        number_of_folds,
    )
    print("accuracy", accuracy)
    print("mean accuracy", sum(accuracy) / len(accuracy))
