import os.path
import random
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):
    def __init__(
            self,
            statues_dict: dict[str, Image.Image],
            bgs_dict: dict[str, Image.Image],
            dataset_type: str = "train",
            transform: Optional[Callable] = None,
            no_generate: bool = False
    ):
        self.statues_dict = statues_dict
        self.bgs_dict = bgs_dict
        self.dataset_type = dataset_type
        self.transform = transform
        self.dataset_path = f'Dataset/{dataset_type}'
        self.data: list[tuple[str, list[float]]] = []

        if dataset_type == "train":
            self.num_samples = 5000
        elif dataset_type == "val":
            self.num_samples = 1000
        elif dataset_type == "test":
            self.num_samples = 200
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        if not os.path.exists(self.dataset_path) or len(os.listdir(self.dataset_path)) != self.num_samples:
            print(f"Generating {self.num_samples} samples for the {dataset_type} dataset")
            if no_generate:
                raise ValueError(f"Dataset {dataset_type} does not exist")
            self.generate_data()
        else:
            print(f"Loading {self.num_samples} samples for the {dataset_type} dataset")
            self.load_data()

    def generate_data(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        else:
            for file in os.listdir(self.dataset_path):
                file_path = os.path.join(self.dataset_path, file)
                os.remove(file_path)

        for i in range(self.num_samples):
            _, bg_img = random.choice(list(self.bgs_dict.items()))  # type: str, Image.Image
            _, statue_img = random.choice(list(self.statues_dict.items()))  # type: str, Image.Image

            bg_width, bg_height = bg_img.size
            statue_width, statue_height = statue_img.size

            max_x, max_y = bg_width - statue_width, bg_height - statue_height

            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)

            combined_img = bg_img.copy()
            combined_img.paste(statue_img, (rand_x, rand_y), statue_img)

            combined_img_path = os.path.join(self.dataset_path, f"{i:05d}.jpg")
            combined_img.save(combined_img_path)

            bbox = [
                (rand_x + statue_width / 2) / bg_width,
                (rand_y + statue_height / 2) / bg_height,
                statue_width / bg_width,
                statue_height / bg_height
            ]
            bbox_str = " ".join(map(str, bbox))

            with open(os.path.join(self.dataset_path, f"{i:05d}.txt"), "w") as f:
                f.write(bbox_str)

            self.data.append((combined_img_path, bbox))

    def load_data(self):
        for i in range(self.num_samples):
            img_path = os.path.join(self.dataset_path, f"{i:05d}.jpg")
            with open(os.path.join(self.dataset_path, f"{i:05d}.txt"), "r") as f:
                bbox_str = f.read().strip()

            bbox = list(map(float, bbox_str.split()))
            self.data.append((img_path, bbox))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path, bbox = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, bbox
