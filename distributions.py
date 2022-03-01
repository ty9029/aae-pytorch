import numpy as np
from torch.utils.data import Dataset


class GaussianMixture(Dataset):
    def __init__(self, num_data, num_labels):
        self.num_data = num_data
        self.num_labels = num_labels
        self.x_var = 0.5
        self.y_var = 0.1
        self.mean = 2.0

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        label = idx % self.num_labels

        r = 2.0 * np.pi * label / self.num_labels

        x = np.random.normal(self.mean, self.x_var)
        y = np.random.normal(0, self.y_var)
        x_tmp = x * np.cos(r) - y * np.sin(r)
        y_tmp = x * np.sin(r) + y * np.cos(r)

        z = np.array([x_tmp, y_tmp], dtype="float32")

        return z, label


class SwissRoll(Dataset):
    def __init__(self, num_data, num_labels):
        self.num_data = num_data
        self.num_labels = num_labels

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        label = idx % self.num_labels
        phi = np.sqrt((np.random.uniform(0.0, 1.0) + label) / self.num_labels)
        noise = np.random.rand()

        x = (5.0 * phi + 0.2 * noise) * np.sin(15.0 * phi)
        y = (5.0 * phi + 0.2 * noise) * np.cos(15.0 * phi)

        z = np.array([x, y], dtype="float32")

        return z, label


class Normal(Dataset):
    def __init__(self, num_data, num_labels):
        self.num_data = num_data
        self.num_labels = num_labels

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        z = np.array(np.random.randn(2), dtype="float32")

        if np.sqrt(z[0] ** 2 + z[1] ** 2) < 1.0:
            label = self.num_labels - 1
        else:
            label = int((np.angle(z[0] + z[1] * 1j, deg=True) + 180.0) // (360.0 / (self.num_labels - 1)))

        return z, label


def get_distribution(name, num_data, num_labels):
    if name == "gaussian":
        return GaussianMixture(num_data, num_labels)
    elif name == "swissroll":
        return SwissRoll(num_data, num_labels)
    elif name == "normal":
        return Normal(num_data, num_labels)
