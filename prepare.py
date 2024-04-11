import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from VOCDataset import VOCDataset


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]

    images = torch.stack(images)

    return images, bboxes


def prepare():
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = prepare()

    for i in tqdm(train):
        pass

    for i in tqdm(val):
        pass
