import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, log_var, beta):
    """
    batch_size = x.size(0)

    MSE = F.mse_loss(recon_x, x, reduction="mean")

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size / torch.prod(torch.tensor(mu.shape[1:]))

    """
    # Reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE + beta * KLD


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32)
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64)
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128)
        )
        self.flatten = nn.Flatten()
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(128 * 32 * 32, latent_dim), nn.Tanh()  # Constrain mu to [-1, 1]
        )
        self.encoder_fc2 = nn.Sequential(
            nn.Linear(128 * 32 * 32, latent_dim),
            nn.Hardtanh(-2, 2),  # Constrain log_var to reasonable range
        )

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 128 * 32 * 32)
        self.decoder_conv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder_conv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder_conv3 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Apply in __init__
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def encode(self, x):
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = F.relu(self.encoder_conv3(x))
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        mu = self.encoder_fc1(x)
        log_var = self.encoder_fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decoder_fc1(z))
        z = z.view(
            z.size(0), 128, 32, 32
        )  # Reshape to (batch_size, channels, height, width)
        z = F.relu(self.decoder_conv1(z))
        z = F.relu(self.decoder_conv2(z))
        z = torch.sigmoid(self.decoder_conv3(z))  # Output a range [0, 1]
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class DiagnosisNetwork(nn.Module):
    def __init__(self, output_size):
        super(DiagnosisNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.4),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(2048, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.4),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.network(x)
