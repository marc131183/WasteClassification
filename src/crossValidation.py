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
NUM_EPOCHS = 100
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
    "val": transforms.Compose(
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
        for inputs, labels in data_loaders["val"]:
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

    base_dir = (
        os.getcwd()
        + "/WasteClassification/data/classification/kFold_{}/".format(number_of_folds)
    )

    # base_dir = "data/classification/kFold_{}/".format(number_of_folds)

    confusion_matrix = None
    conf_initialized = False
    time_elapsed = []
    num_epochs = []

    for i in range(number_of_folds):
        print("-" * 15, "Started working on Fold {}".format(i + 1), "-" * 15)
        temp_dir = base_dir + "fold_{}".format(i)

        image_datasets = {
            x: datasets.ImageFolder(os.path.join(temp_dir, x), DATA_TRANSFORMS[x])
            for x in ["train", "test", "val"]
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test", "val"]}
        data_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=10
            )
            for x in ["train", "test", "val"]
        }

        model, e, t = train_function(*init_function(), data_loaders, dataset_sizes)

        time_elapsed.append(t)
        num_epochs.append(e)

        acc, conf = evaluate_model(model, data_loaders, device)

        all_acc.append(acc)
        if conf_initialized:
            confusion_matrix = confusion_matrix + conf
        else:
            confusion_matrix = conf
            conf_initialized = True

    return all_acc, confusion_matrix, time_elapsed, num_epochs


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loaders,
    dataset_sizes,
    device,
    max_epochs=25,
    patience=5,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0
    no_improvement_since = 0

    phases = [elem for elem in ["train", "test", "val"] if elem in data_loaders]

    plot_data = {phase + metric: [] for phase in phases for metric in ["loss", "acc"]}

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        print("-" * 10)

        # Each epoch has a training and testing phase
        for phase in phases:
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

            plot_data[phase + "acc"].append(epoch_acc)
            plot_data[phase + "loss"].append(epoch_loss)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "test":
                # deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    no_improvement_since = 0

                # early stopping when accuracy hasn't improved on test data for some time (patience epochs)
                no_improvement_since += 1
                if no_improvement_since >= patience:
                    time_elapsed = time.time() - since
                    print(
                        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, due to early stopping"
                    )
                    print(f"Best test acc: {best_acc:4f}")

                    for metric in ["acc", "loss"]:
                        for phase in phases:
                            print("{} {}:".format(phase, metric))
                            print(plot_data[phase + metric][: best_epoch + 1])

                    # load best model weights
                    model.load_state_dict(best_model_wts)

                    return model, best_epoch, time_elapsed

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s, due to maximum epochs reached"
    )
    print(f"Best test acc: {best_acc:4f}")

    for metric in ["acc", "loss"]:
        for phase in phases:
            print("{} {}:".format(phase, metric))
            print(plot_data[phase + metric][: best_epoch + 1])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch, time_elapsed


def get_trainable_layers_of_resnet(model):
    trainable_layers = []
    elem_stack = list(model.children())
    elem_stack.reverse()
    while len(elem_stack) > 0:
        cur_elem = elem_stack[-1]
        del elem_stack[-1]
        # check if elem has children, if yes then don't count it
        children_of_cur = list(cur_elem.children())
        if len(children_of_cur) > 0:
            elem_stack.extend(children_of_cur)
        else:
            # elem has no children, count it as layer and check if is trainable
            if len(list(cur_elem.parameters())) > 0:
                trainable_layers.append(cur_elem)

    return trainable_layers


def model_init_function(
    model_architecture,
    final_layers,
    final_layers_in,
    device,
    feature_percentage_frozen=0.0,
    learn_rate=0.001,
    momentum=0.9,
):
    available_architectures = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "alexnet": models.alexnet,
        "vgg": models.vgg11,
    }

    model = available_architectures[model_architecture](pretrained=True)

    if "resnet" in model_architecture:
        trainable_layers = get_trainable_layers_of_resnet(model)

        for i in range(int(0.9 * len(trainable_layers))):
            for param in trainable_layers[i].parameters():
                param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, final_layers_in), *final_layers
        )
    elif "alexnet" in model_architecture or "vgg" in model_architecture:
        # get the indices of layers which have trainable parameters
        trainable_layers = [
            i
            for i, elem in enumerate(model.features)
            if len(list(elem.parameters())) != 0
        ]
        # freeze the first (feature_percentage_frozen * 100)% trainable layers
        for i in range(int(feature_percentage_frozen * len(trainable_layers))):
            for param in model.features[trainable_layers[i]].parameters():
                param.requires_grad = False

        if model_architecture == "vgg":
            in_features = 25088
        elif model_architecture == "alexnet":
            in_features = 9216

        # change the classifier
        model.classifier = nn.Sequential(
            nn.Linear(in_features, final_layers_in), *final_layers
        )

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
    # Model Parameters
    model_type = "resnet50"
    feature_percentage_frozen = 0
    classifier_type = 1
    if classifier_type == 0:
        model_final_struc = []
        model_final_in = NUM_CLASSES
    elif classifier_type == 1:
        model_final_struc = [
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, NUM_CLASSES),
        ]
        model_final_in = 512
    elif classifier_type == 2:
        model_final_struc = [nn.ReLU(), nn.Dropout(), nn.Linear(128, NUM_CLASSES)]
        model_final_in = 128

    print(
        "-" * 15,
        "Started crossvalidation on {} model, feature_percentage_frozen = {}".format(
            model_type, feature_percentage_frozen
        ),
        "-" * 15,
    )
    print("Classifier model type:", classifier_type)
    print("Classifier model structure:", model_final_struc)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    accuracy, confusion_matrix, time_elapsed, num_epochs = crossValidateModel(
        lambda: model_init_function(
            model_type,
            model_final_struc,
            model_final_in,
            device,
            feature_percentage_frozen=feature_percentage_frozen,
        ),
        lambda a, b, c, d, e, f: train_model(
            a,
            b,
            c,
            d,
            e,
            f,
            device,
            max_epochs=NUM_EPOCHS,
            patience=10,
        ),
        device,
        NUMBER_OF_FOLDS,
    )
    print("-" * 30)
    print("accuracy", accuracy)
    print("mean accuracy", np.mean(accuracy))
    print("stddev accuracy", np.std(accuracy))
    print("time elapsed", time_elapsed)
    print("average time elapsed", np.mean(time_elapsed))
    print("epochs trained", num_epochs)
    print("average epochs trained", np.mean(num_epochs))
    print("confusion matrix")
    print(confusion_matrix)
