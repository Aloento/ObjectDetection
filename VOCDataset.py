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
                i = VOCItem(line.strip())

                if len(i.annotation.objects) == 1:
                    self.dataset.append(i)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="coco"))

        if image_set == "train":
            self.transform = A.Compose([
                A.RandomScale(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format="coco"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        bboxes: list[list[int]] = []

        for obj in item.annotation.objects:
            bbox = obj.bndbox.get()
            # convert form pascal_voc (x_min, y_min, x_max, y_max)
            # to coco (x_min, y_min, width, height)
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]

            bbox += [catalogs.index(obj.name)]
            bboxes.append(bbox)

        image = item.get_image()
        transformed = self.transform(image=image, bboxes=bboxes)

        t_image = transformed["image"]  # type: torch.FloatTensor
        t_bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)

        return t_image, t_bboxes
