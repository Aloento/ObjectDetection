import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def prepare():
    train_dataset = torchvision.datasets.CIFAR10(
        root='CIFAR',
        train=True,
        download=False,
        transform=transform_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root='CIFAR',
        train=False,
        download=False,
        transform=transform_test
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = prepare()

    for i in tqdm(train):
        pass

    for i in tqdm(val):
        pass
