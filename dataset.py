import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset(data_name, data_root, image_size, train):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    if data_name == "mnist":
        dataset = datasets.MNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "fushion-mnist":
        dataset = datasets.FashionMNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "kmnist":
        dataset = datasets.KMNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "emnist":
        dataset = datasets.EMNIST(root=data_root, split="byclass", train=train, transform=transform, download=True)

    else:
        dataset = None

    return dataset
