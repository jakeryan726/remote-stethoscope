import torch
import torch.nn as nn
from models import DiagnosisNetwork
from torch.utils.data import DataLoader
import pickle
import optuna
from training_utils import f1score


class Objective:
    def __init__(self, study, train_ds, validate_ds, num_classes, model_filename, training_losses_filename, study_filename):
        self.study = study
        self.num_classes = num_classes
        self.train_ds = train_ds
        self.validate_ds = validate_ds
        self.model_filename = model_filename
        self.training_losses_filename = training_losses_filename
        self.study_filename = study_filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)


    def define_model(self, trial):
        model = DiagnosisNetwork(self.num_classes).to(self.device)
        lr = trial.suggest_float("lr", low=.00001, high=.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        return model, optimizer


    def __call__(self, trial):
        model, optimizer = self.define_model(trial)
        epochs = trial.suggest_int("epochs", low=500, high=2000)
        batch_size = trial.suggest_int("batch_size", low=1, high=32)
        train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)

        training_losses = train_cnn(model, train_dl, optimizer, self.device, epochs)
        fscore = f1score(model, self.validate_ds, self.device)
        
        if trial.number == 0 or fscore > self.study.best_value:
            torch.save(model, self.model_filename)
            torch.save(training_losses, self.training_losses_filename)

        # Saving study every trial
        with open(self.study_filename, 'wb') as f:
            pickle.dump(self.study, f)
        
        return fscore


def train_cnn(model, train_dl, optimizer, device, epochs):
    model.train()
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in train_dl:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
    return losses


if __name__ == "__main__":
    torch.cuda.empty_cache()
    model_filename = "cnn_training_results/cnn_model.pt"
    training_losses_filename = "cnn_training_results/cnn_training_losses.pt"
    study_filename = "cnn_training_results/cnn_study.pkl"
    train_ds = torch.load("processed data/bispectrum_train_generated_ds.pt")
    validate_ds = torch.load("processed data/bispectrum_validate_ds.pt")
    num_classes = 9

    study = optuna.create_study(direction='maximize')
    #study = pickle.load(open(study_filename, 'rb'))
    study.optimize(Objective(study=study, train_ds=train_ds, validate_ds=validate_ds, num_classes=num_classes, model_filename=model_filename, training_losses_filename=training_losses_filename, study_filename=study_filename), n_trials=1)