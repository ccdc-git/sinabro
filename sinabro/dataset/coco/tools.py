from typing import List
import funcy
import json

from sklearn.model_selection import train_test_split


def make_license(name: str = "", _id: int = 0, url: str = ""):
    return {"name": name, "id": _id, "url": url}


def make_info(
    contributor: str = "",
    date_created: str = "",
    description: str = "",
    url: str = "",
    version: str = "",
    year: str = "",
):
    return {
        "contributor": contributor,
        "date_created": date_created,
        "description": description,
        "url": url,
        "version": version,
        "year": year,
    }


def make_category(
    _id: int,
    name: str,
    supercategory: str = "",
):
    return {
        "id": _id,
        "name": name,
        "supercategory": supercategory,
    }


def make_image(
    _id: int,
    width: int,
    height: int,
    file_name: str,
    license: int = 0,
    flickr_url: str = "",
    coco_url: str = "",
    date_captured: int = 0,
):
    return {
        "id": _id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": license,
        "flickr_url": flickr_url,
        "coco_url": coco_url,
        "date_captured": date_captured,
    }


def make_annotation(
    _id: int,
    image_id: int,
    category_id: int,
    segmentation: List,
    area: float,
    bbox: List[float],
    iscrowd: int = 0,
    attributes={"occluded": False},
):
    return {
        "id": _id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": iscrowd,
        "attributes": attributes,
    }


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
        )


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def split_coco_data(
    annotations: str, train: str, test: str, s: float, having: bool = False
):
    with open(annotations, "rt", encoding="UTF-8") as ann_file:
        coco = json.load(ann_file)
    info = coco["info"]
    licenses = coco["licenses"]
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]
    # number_of_images = len(images)

    images_with_annotations = funcy.lmap(lambda a: int(a["image_id"]), annotations)

    if having:
        images = funcy.lremove(lambda i: i["id"] not in images_with_annotations, images)

    x, y = train_test_split(images, train_size=s, random_state=42)

    save_coco(train, info, licenses, x, filter_annotations(annotations, x), categories)
    save_coco(test, info, licenses, y, filter_annotations(annotations, y), categories)

    print("Saved {} entries in {} and {} in {}".format(len(x), train, len(y), test))
    return x, y, filter_annotations(annotations, x), filter_annotations(annotations, y)
