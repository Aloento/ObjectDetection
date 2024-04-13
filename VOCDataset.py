import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from VOCImage import VOCItem

catalogs = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(Dataset):
    sets_path = "VOC/ImageSets/Main/"

    def __init__(
            self,
            image_set: str = "train",  # train, val, trainval
    ):
        self.dataset: list[VOCItem] = []
        set_path = VOCDataset.sets_path + image_set + ".txt"

        with open(set_path) as f:
            for line in f:
                self.dataset.append(VOCItem(line.strip()))

        self.transform = A.Compose([
            A.RandomScale(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(416, 416),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        bboxes: list[list[int]] = []

        for obj in item.annotation.objects:
            bbox = obj.bndbox.get()
            bbox += [catalogs.index(obj.name)]
            bboxes.append(bbox)

        image = item.get_image()
        transformed = self.transform(image=image, bboxes=bboxes)

        t_image = transformed["image"]  # type: torch.FloatTensor
        t_bboxes = torch.tensor(transformed["bboxes"])

        return t_image, t_bboxes
