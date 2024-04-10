import os

import mmh3
from PIL import Image
from torch.utils.data import DataLoader

from ObjectDetectionDataset import ObjectDetectionDataset


def load_images(directory: str) -> dict[str, tuple[int, Image.Image]]:
    print("Loading images from", directory)
    images_dict: dict[str, tuple[int, Image.Image]] = {}
    filenames = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            filename, _ = os.path.splitext(file)
            filenames.append(filename)

    filenames.sort()
    name_to_id = {name: i for i, name in enumerate(filenames)}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            filename, _ = os.path.splitext(file)

            try:
                img = Image.open(file_path)
                class_id = name_to_id[filename]
                images_dict[filename] = (class_id, img)
            except Exception as e:
                print(f"Error loading image {file}: {e}")

    return images_dict


def download_backgrounds(directory: str):
    if os.path.exists(directory):
        if os.listdir(directory):
            print(f"Directory {directory} already exists and is not empty")
            return
    else:
        os.makedirs(directory)

    print(f"Downloading backgrounds to {directory}")

    from icrawler.builtin import BingImageCrawler

    crawler = BingImageCrawler(
        parser_threads=2,
        downloader_threads=4,
        storage={"root_dir": directory}
    )

    search_queries = ['budapest', 'budapest parliament', 'buda castle', 'budapest st stephen basilica',
                      'budapest heroes square']

    for query in search_queries:
        crawler.crawl(keyword=query, max_num=10, file_idx_offset="auto")


def load_dicts():
    print("Preparing the data...")

    statues_dict = load_images("Statues")
    print(f"Loaded {len(statues_dict)} images", statues_dict.keys())

    download_backgrounds("Backgrounds")
    bgs_dict = load_images("Backgrounds")
    print(f"Loaded {len(bgs_dict)} backgrounds")

    return statues_dict, bgs_dict


def load_datasets(statues_dict: dict[str, tuple[int, Image.Image]], bgs_dict: dict[str, tuple[int, Image.Image]]):
    train_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="train")
    train_loader = DataLoader(train_dataset, batch_size=5000 // 100, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="val")
    val_loader = DataLoader(val_dataset, batch_size=1000 // 100, shuffle=False, num_workers=4)

    test_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="test")
    test_loader = DataLoader(test_dataset, batch_size=200 // 50, shuffle=False, num_workers=2)

    print("Data prepared")
    return train_loader, val_loader, test_loader


def prepare():
    statues, bgs = load_dicts()
    return load_datasets(statues, bgs)


if __name__ == "__main__":
    prepare()
