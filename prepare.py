import os
from PIL import Image


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

    search_queries = ['budapest', 'budapest parliament', 'buda castle', 'budapest st stephen basilica', 'budapest heroes square']

    for query in search_queries:
        crawler.crawl(keyword=query, max_num=10, file_idx_offset="auto")


def prepare():
    print("Preparing the data...")

    statues_dict = load_images("Statues")
    print(f"Loaded {len(statues_dict)} images", statues_dict.keys())

    download_backgrounds("Backgrounds")
    bgs_dict = load_images("Backgrounds")
    print(f"Loaded {len(bgs_dict)} backgrounds")

    return statues_dict, bgs_dict


if __name__ == "__main__":
    prepare()
