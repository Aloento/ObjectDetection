from torch.utils.data import DataLoader

from VOCDataset import VOCDataset


def prepare():
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=8,
        drop_last=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    prepare()
