import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from torch.nn.functional import one_hot

from VOCImage import VOCItem


class VOCDataset(Dataset):
    catalogs = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    sets_path = "VOC/ImageSets/Main/"

    def __init__(
            self,
            image_set: str = "train",  # train, val, trainval
    ):
        self.dataset: list[VOCItem] = []

        for catalog in tqdm(VOCDataset.catalogs, desc=f"Loading VOC {image_set} Dataset"):  # type: str
            set_path = VOCDataset.sets_path + catalog + "_" + image_set + ".txt"

            with open(set_path) as f:
                for line in f:
                    image, exist = line.split()
                    self.dataset.append(VOCItem(catalog, image, exist))

        self.transform = A.Compose([
            A.Resize(640, 640),
            A.RandomScale(scale_limit=0.5, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        bboxes: list[list[int]] = []

        for obj in item.annotation.objects:
            bbox = obj.bndbox.get()
            bbox += [VOCDataset.catalogs.index(obj.name)]
            bboxes.append(bbox)

        one_hot_target = one_hot(
            torch.tensor(VOCDataset.catalogs.index(item.category)),
            num_classes=len(VOCDataset.catalogs)
        )  # type: torch.LongTensor

        image = item.get_image()
        transformed = self.transform(image=image, bboxes=bboxes)

        t_image = transformed["image"]  # type: torch.FloatTensor
        t_bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float16)

        return {
            "image": t_image,
            "bboxes": t_bboxes,
            "target": one_hot_target,
            "exist": item.exist
        }
