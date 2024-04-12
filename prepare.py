import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from VOCDataset import VOCDataset, catalogs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    images = []
    bboxes = []
    labels = []

    for image, bbox_s in batch:
        images.append(image.to(device))

        unique_labels = bbox_s[:, 4].unique(sorted=True).long()
        hot_labels = torch.zeros((1, len(catalogs))).scatter_(1, unique_labels.unsqueeze(0), 1.)
        labels.append(hot_labels.squeeze(0).to(device))

        bbox_s = bbox_s[:, :4]
        bboxes.append(bbox_s)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, bboxes, labels


def prepare():
    train_dataset = VOCDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=40,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = VOCDataset("val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=40,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = prepare()

    for i in tqdm(train):
        pass

    for i in tqdm(val):
        pass
