from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch
import torch.nn as nn


def kfold(
    model, train_ds, train_func, test_func, optimizer, device, epochs, batch_size, max_beta, k=4
):
    result = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_ds):
        # Create Subsets
        train_subset = Subset(train_ds, train_idx)
        validate_subset = Subset(train_ds, val_idx)

        # Create DataLoader
        train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Train and Test
        _ = train_func(model, train_dl, optimizer, device, epochs, max_beta)
        result += test_func(model, validate_subset, device)
    return result / k


def f1score(model, data, device):
    x, y = next(iter(DataLoader(data, batch_size=len(data))))
    y = y.argmax(1)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_hat = model(x).argmax(1)
    return f1_score(y.cpu(), y_hat.cpu(), average="macro")


def accuracy(model, data, device):
    x, y = next(iter(DataLoader(data, batch_size=len(data))))
    y = y.argmax(1)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_hat = model(x).argmax(1)
    return accuracy_score(y.cpu(), y_hat.cpu())


def prfs(model, data, device):
    x, y = next(iter(DataLoader(data, batch_size=len(data))))
    y = y.argmax(1)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_hat = model(x).argmax(1)
    return precision_recall_fscore_support(y.cpu(), y_hat.cpu(), average="macro")


def cm(model, data, device):
    x, y = next(iter(DataLoader(data, batch_size=len(data))))
    y = y.argmax(1)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_hat = model(x).argmax(1)
    return confusion_matrix(y.cpu(), y_hat.cpu())


def test_cnn(model, test_dl, device):
    model.eval()
    test_loss = 0
    dataset_length = len(test_dl.dataset)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in test_dl:
            x, y = batch[0].to(device), batch[1].to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            test_loss += loss.item()
    return test_loss / dataset_length


def aggregated_metrics(model, data, device, window=5):
    predictions, labels = [], []

    unique_labels = set(label.argmax(0).item() for _, label in data)
    for label in unique_labels:
        subset = [x[0] for x in data if x[1].argmax(0).item() == label]
        for i in range(0, len(subset), window):
            batch = torch.stack(subset[i : i + window]).to(device)
            predictions.append(torch.mode(model(batch).argmax(1))[0].item())
            labels.append(label)
    acc = accuracy_score(labels, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    return acc, precision, recall, fscore
