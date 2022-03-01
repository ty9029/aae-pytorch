import torch
import torch.nn as nn
import math


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTranspose2d, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.convt(x)
        out = self.relu(out)

        return out


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out


def reparameterize(mu, sigma):
    std = torch.exp(0.5 * sigma)
    esp = torch.randn(*mu.size()).to("cuda")
    z = mu + std * esp
    return z


class Encoder(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim):
        super(Encoder, self).__init__()
        down_size = 4
        num_layer = int(math.log2(image_size) - math.log2(down_size))

        self.conv = [Conv2d(image_channels, 16, 3, 1, 1)]
        for i in range(num_layer):
            in_channels = min(16 * 2 ** i, 512)
            out_channels = min(16 * 2 ** (i + 1), 512)
            self.conv.append(Conv2d(in_channels, out_channels, 3, 2, 1))
        self.conv = nn.Sequential(*self.conv)

        num_channels = min(16 * 2 ** num_layer, 512)
        self.fc_mean = nn.Linear(down_size ** 2 * num_channels, latent_dim)
        self.fc_logvar = nn.Linear(down_size ** 2 * num_channels, latent_dim)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.size(0), -1)

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        z = reparameterize(mean, logvar)

        return z


class Decoder(nn.Module):
    def __init__(self, image_size, image_channels, latent_dim):
        super(Decoder, self).__init__()
        self.down_size = 4
        num_layer = int(math.log2(image_size) - math.log2(self.down_size))

        self.base_channels = min(16 * 2 ** num_layer, 512)
        self.fc = nn.Linear(latent_dim, self.down_size ** 2 * self.base_channels)

        self.conv = []
        for i in reversed(range(num_layer)):
            in_channels = min(16 * 2 ** (i + 1), 512)
            out_channels = min(16 * 2 ** i, 512)
            self.conv.append(ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
        self.conv.append(nn.Conv2d(16, image_channels, 3, 1, 1))
        self.conv = nn.Sequential(*self.conv)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = out.view(x.size(0), self.base_channels, self.down_size, self.down_size)
        out = self.sigmoid(self.conv(out))

        return out


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)

        return out


def one_hot(x, num_classes):
    onehot = torch.zeros(len(x), num_classes, device="cuda")
    onehot = onehot.scatter_(1, x.view(-1, 1), 1)
    return onehot


class Discriminator(nn.Module):
    def __init__(self, latent_dim, label_size=None, use_label=False):
        super(Discriminator, self).__init__()
        self.use_label = use_label
        self.label_size = label_size
        latent_dim = latent_dim + label_size if use_label else latent_dim

        self.linear = nn.Sequential(
            Linear(latent_dim, 256),
            Linear(256, 128),
            Linear(128, 64),
            nn.Linear(64, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        if self.use_label:
            y = one_hot(y, self.label_size)
            x = torch.cat([x, y], dim=1)

        out = self.sigmoid(self.linear(x))

        return out
