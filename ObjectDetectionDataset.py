import os.path
import random

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm


class ObjectDetectionDataset(Dataset):
    def __init__(
            self,
            statues_dict: dict[str, tuple[int, Image.Image]],
            bgs_dict: dict[str, tuple[int, Image.Image]],
            dataset_type: str = "train",
            no_generate: bool = False,
            image_size: int = 640
    ):
        self.statues_dict = statues_dict
        self.bgs_dict = bgs_dict
        self.dataset_type = dataset_type
        self.dataset_path = f'Dataset/{dataset_type}'
        self.data: list[tuple[str, list[float]]] = []
        self.image_size = image_size

        self.transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="coco"))

        if dataset_type == "train":
            self.num_samples = 5000
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(224, 224),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco"))
        elif dataset_type == "val":
            self.num_samples = 1000
        elif dataset_type == "test":
            self.num_samples = 200
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if not os.path.exists(self.dataset_path) or len(os.listdir(self.dataset_path)) != self.num_samples * 2:
            print(f"Generating {self.num_samples} samples for the {dataset_type} dataset")
            if no_generate:
                raise ValueError(f"Dataset {dataset_type} does not exist")
            self.generate_data()
        else:
            print(f"Loading {self.num_samples} samples for the {dataset_type} dataset")
            self.load_data()

    def generate_data(self):
        if not os.path.exists(self.dataset_path):
            print(f"Creating directory {self.dataset_path}")
            os.makedirs(self.dataset_path)
        else:
            print(f"Cleaning directory {self.dataset_path}")
            for file in os.listdir(self.dataset_path):
                file_path = os.path.join(self.dataset_path, file)
                os.remove(file_path)

        progress_bar = tqdm(range(self.num_samples), desc=f"Generating {self.dataset_type}")
        for i in progress_bar:
            _, bg_img = random.choice(list(self.bgs_dict.values()))  # type: int, Image.Image

            if bg_img.width > self.image_size and bg_img.height > self.image_size:
                start_x = random.randint(0, bg_img.width - self.image_size)
                start_y = random.randint(0, bg_img.height - self.image_size)
                bg_img = bg_img.crop((start_x, start_y, start_x + self.image_size, start_y + self.image_size))
            else:
                bg_img = bg_img.resize((self.image_size, self.image_size))

            statue_id, statue_img = random.choice(list(self.statues_dict.values()))  # type: int, Image.Image

            statue_height = random.randint(150, 200)
            ratio = statue_height / statue_img.height
            statue_width = round(statue_img.width * ratio)
            statue_img = statue_img.resize((statue_width, statue_height))

            bg_width, bg_height = bg_img.size
            assert bg_width == bg_height == self.image_size, \
                f"Background image should be {self.image_size} x {self.image_size}"
            max_x, max_y = bg_width - statue_width, bg_height - statue_height

            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)

            combined_img = bg_img.copy()
            combined_img.paste(statue_img, (rand_x, rand_y), statue_img)

            combined_img_path = os.path.join(self.dataset_path, f"{i:05d}.webp")
            combined_img.save(combined_img_path)

            # [x_min, y_min, width, height, class_id]
            bbox = [
                rand_x,
                rand_y,
                statue_width,
                statue_height,
                statue_id
            ]
            bbox_str = " ".join(map(str, bbox))

            with open(os.path.join(self.dataset_path, f"{i:05d}.txt"), "w") as f:
                f.write(bbox_str)

            self.data.append((combined_img_path, bbox))

    def load_data(self):
        for i in range(self.num_samples):
            img_path = os.path.join(self.dataset_path, f"{i:05d}.webp")
            with open(os.path.join(self.dataset_path, f"{i:05d}.txt"), "r") as f:
                bbox_str = f.read().strip()

            bbox = list(map(float, bbox_str.split()))
            self.data.append((img_path, bbox))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, bbox = self.data[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0

        transformed = self.transform(image=img, bboxes=[bbox])
        img = transformed["image"]

        bbox = transformed["bboxes"][0]
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return img, bbox
