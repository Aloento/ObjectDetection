import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from VOCDataset import VOCDataset


def collate_fn(batch) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images = []
    bboxes = []

    for image, bbox_s in batch:
        images.append(image)
        bboxes.append(bbox_s)

    images = torch.stack(images)

    return images, bboxes


def prepare():
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(
        train_dataset,
        # batch_size=200,
        batch_size=2,
        shuffle=True,
        # num_workers=8,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(
        val_dataset,
        # batch_size=100,
        batch_size=1,
        shuffle=False,
        # num_workers=8,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = prepare()

    for i in tqdm(train):
        pass

    for i in tqdm(val):
        pass
