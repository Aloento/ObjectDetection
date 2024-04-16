import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from VOCImage import VOCItem

catalogs = [
    'aeroplane',  # 0
    'bicycle',  # 1
    'bird',  # 2
    'boat',  # 3
    'bottle',  # 4
    'bus',  # 5
    'car',  # 6
    'cat',  # 7
    'chair',  # 8
    'cow',  # 9
    'diningtable',  # 10
    'dog',  # 11
    'horse',  # 12
    'motorbike',  # 13
    'person',  # 14
    'pottedplant',  # 15
    'sheep',  # 16
    'sofa',  # 17
    'train',  # 18
    'tvmonitor'  # 19
]


class VOCDataset(Dataset):
    def __init__(
            self,
            image_set: str = "train",  # train, val
    ):
        self.dataset: list[VOCItem] = []
        set_paths = [("2007", f"VOC/2007/ImageSets/Main/{image_set}.txt")]

        if image_set == "train":
            set_paths.append(("2012", "VOC/2012/ImageSets/Main/trainval.txt"))

        for year, set_path in set_paths:
            with open(set_path) as f:
                for line in f:
                    i = VOCItem(year, line.strip())

                    if len(i.annotation.objects) == 1:
                        self.dataset.append(i)

        self.transform = A.Compose([
            A.Resize(224, 224),
            # A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="coco"))

        if image_set == "train":
            self.transform = A.Compose([
                # A.RandomScale(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(224, 224),
                # A.Normalize(),
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
