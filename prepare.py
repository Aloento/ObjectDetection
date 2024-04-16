import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from VOCDataset import VOCDataset


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    bboxes = []

    for image, bbox_s in batch:
        images.append(image)
        bboxes.append(bbox_s[0])

    images = torch.stack(images)
    bboxes = torch.stack(bboxes)

    return images, bboxes


def prepare():
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        # batch_size=2,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=100,
        # batch_size=2,
        num_workers=2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = prepare()

    for i in tqdm(train):
        pass

    for i in tqdm(val):
        pass
