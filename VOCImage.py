import numpy as np
from PIL import Image

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from torchvision.datasets import VOCDetection


class BoundingBox:
    def __init__(self, xmin: str, ymin: str, xmax: str, ymax: str, **kwargs):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def get(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class Object:
    def __init__(self, name: str, pose: str, truncated: str, difficult: str, bndbox: dict, **kwargs):
        self.name = name
        self.pose = pose
        self.truncated = truncated == '1'
        self.difficult = difficult == '1'
        self.bndbox = BoundingBox(**bndbox)


class Size:
    def __init__(self, width: str, height: str, depth: str, **kwargs):
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)


class VOCAnnotation:
    def __init__(self, raw: dict):
        raw = raw['annotation']
        self.folder = raw.get('folder')
        self.filename = raw.get('filename')
        self.size = Size(**raw['size'])
        self.segmented = raw.get('segmented') == '1'
        self.objects = [Object(**obj) for obj in raw.get('object', [])]


class VOCItem:
    base_path = "VOC/"
    anno_path = base_path + "Annotations/"
    images_path = base_path + "JPEGImages/"

    annotation_dict: dict[str, VOCAnnotation] = {}

    def __init__(self, image_id: str):
        self.image_id = image_id

        self.image_path = f"{VOCItem.images_path}{image_id}.jpg"
        self.annotation_path = f"{VOCItem.anno_path}{image_id}.xml"

        self.annotation = self.parse_annotation()

    def parse_annotation(self):
        if self.annotation_path in VOCItem.annotation_dict:
            return VOCItem.annotation_dict[self.annotation_path]

        parser = VOCDetection.parse_voc_xml
        root = ET_parse(self.annotation_path).getroot()
        res = parser(root)
        anno = VOCAnnotation(res)

        VOCItem.annotation_dict[self.annotation_path] = anno
        return anno

    def get_image(self):
        img = Image.open(self.image_path).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0
        return img
