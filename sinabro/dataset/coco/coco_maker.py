from typing import List
from dataclasses import dataclass, field
import json

from .tools import make_license, make_info


@dataclass
class COCOAnnotation:
    category_id: int
    segmentation: List
    area: float
    bbox: List[float]
    iscrowd: int = 0
    attributes = {"occluded": False}

    def to_dict(self):
        return self.__dict__


@dataclass
class COCOImage:
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: int = 0
    annotations: List[COCOAnnotation] = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ["annotations"]}


class COCOMaker:
    def __init__(self, category_map: dict):
        self._data: dict[str, COCOImage] = {}
        self._category_map = category_map

    def __len__(self):
        return len(self._data)

    def add(self, img_name: str, coco_image: COCOImage):
        self._data[img_name] = coco_image

    def pop(self, img_name):
        return self._data.pop(img_name)

    def get_coco_image(self, img_name):
        return self._data.get(img_name)

    def to_coco(self, filename, licenses=None, info=None):
        img_id = 1
        anno_id = 1
        imgs = []
        annos = []
        if licenses is None:
            licenses = [make_license()]
        if info is None:
            info = make_info()

        for coco_img in self._data.values():
            imgs.append({"id": img_id, **coco_img.to_dict()})
            for coco_anno in coco_img.annotations:
                annos.append({"id": anno_id, "image_id": img_id, **coco_anno.to_dict()})
                anno_id += 1
            img_id += 1

        with open(filename, "w") as f:
            json.dump(
                {
                    "licenses": licenses,
                    "info": info,
                    "categories": list(self._category_map.values()),
                    "images": imgs,
                    "annotations": annos,
                },
                f,
            )
