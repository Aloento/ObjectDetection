import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ObjectDetectionDataset import ObjectDetectionDataset


def load_images(directory: str) -> dict[str, Image.Image]:
    print("Loading images from", directory)
    images_dict: dict[str, Image.Image] = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            filename, _ = os.path.splitext(file)

            try:
                img = Image.open(file_path)
                images_dict[filename] = img
            except Exception as e:
                print(f"Error loading image {file}: {e}")

    return images_dict


def download_backgrounds(directory: str):
    if os.path.exists(directory) and os.listdir(directory):
        print(f"Directory {directory} already exists and is not empty")
        return

    print(f"Downloading backgrounds to {directory}")
    os.makedirs(directory)

    from icrawler.builtin import GoogleImageCrawler

    crawler = GoogleImageCrawler(
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


def load_datasets(statues_dict: dict[str, Image.Image], bgs_dict: dict[str, Image.Image]):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = ObjectDetectionDataset(statues_dict, bgs_dict, dataset_type="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print("Data prepared")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    statues, bgs = load_dicts()
    load_datasets(statues, bgs)
