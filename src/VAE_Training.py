import torch
from models import VAE, vae_loss
from training_utils import kfold
from torch.utils.data import DataLoader
import pickle
import optuna


class Objective:
    def __init__(
        self,
        study,
        max_beta,
        train_ds,
        model_filename,
        training_losses_filename,
        study_filename,
    ):
        self.study = study
        self.max_beta = max_beta
        self.train_ds = train_ds
        self.model_filename = model_filename
        self.training_losses_filename = training_losses_filename
        self.study_filename = study_filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def define_model(self, trial):
        latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256, 512])
        model = VAE(latent_dim).to(self.device)
        lr = trial.suggest_float("lr", low=0.0000001, high=0.0001)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def __call__(self, trial):
        model, optimizer = self.define_model(trial)
        epochs = trial.suggest_int("epochs", low=400, high=2000)
        batch_size = trial.suggest_int("batch_size", low=1, high=32)

        loss = kfold(
            model,
            self.train_ds,
            train_vae,
            test_vae,
            optimizer,
            self.device,
            epochs,
            batch_size,
        )

        if trial.number == 0 or loss < self.study.best_value:
            train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
            model, optimizer = self.define_model(trial)
            training_losses = train_vae(
                model, train_dl, optimizer, self.device, epochs, self.max_beta
            )

            torch.save(model, self.model_filename)
            torch.save(training_losses, self.training_losses_filename)

        # Saving study every trial
        with open(self.study_filename, "wb") as f:
            pickle.dump(self.study, f)

        return loss


def train_vae(model, train_dl, optimizer, device, epochs, max_beta):
    model.train()
    losses = []

    for epoch in range(epochs):
        for batch in train_dl:
            x = batch[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(x)
            if torch.isinf(recon_batch).any().item():
                print("Infinity in recon")
            if torch.isnan(recon_batch).any().item():
                print("nan in recon")

            beta = beta_scheduler(epoch, epochs, max_beta)
            loss = vae_loss(recon_batch, x, mu, log_var, beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            losses.append(loss.item())
            optimizer.step()
    return losses


def test_vae(model, test_dl, device, beta=0.005):
    model.eval()
    test_loss = 0
    dataset_length = len(test_dl.dataset)
    with torch.no_grad():
        for batch in test_dl:
            x = batch[0].to(device)

            recon_batch, mu, log_var = model(x)
            loss = vae_loss(recon_batch, x, mu, log_var, beta)
            test_loss += loss.item()
    return test_loss / dataset_length


def beta_scheduler(epoch, max_epoch, max_beta):
    return min((max_beta * epoch / max_epoch), max_beta)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    model_filename = "training_results/VAE_model.pt"
    training_losses_filename = "training_results/training_losses.pt"
    study_filename = "training_results/study.pkl"
    train_ds = torch.load("processed data/bispectrum_train_ds.pt")
    max_beta = 0.005

    study = optuna.create_study(direction="minimize")
    # study = pickle.load(open(study_filename, 'rb'))
    study.optimize(
        Objective(
            study=study,
            max_beta=max_beta,
            train_ds=train_ds,
            model_filename=model_filename,
            training_losses_filename=training_losses_filename,
            study_filename=study_filename,
        ),
        n_trials=10,
    )
