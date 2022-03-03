from typing import *
import os
import random

from pycocotools.coco import COCO


def save_anns(img, coco: COCO, dst_path):
    anns = coco.getAnnIds(imgIds=img["id"])
    anns = coco.loadAnns(anns)
    lines = []
    w, h = (img["width"], img["height"])
    for ann in anns:
        catId = ann["category_id"]
        bbox = ann["bbox"]
        c_x = (bbox[0] + bbox[2] / 2) / w
        c_y = (bbox[1] + bbox[3] / 2) / h
        w_x = bbox[2] / w
        w_y = bbox[3] / h

        lines.append(f"{catId} {c_x:.05} {c_y:.05} {w_x:.05} {w_y:.05}")
    with open(dst_path, "w") as f:
        f.write("\n".join(lines))


def make_yolo_txt(coco: COCO, dataset_dir, txt_path):
    lines = []
    for img in coco.imgs.values():
        file_name = img["file_name"]
        dst_path = os.path.join(dataset_dir, file_name)
        dst_path = os.path.abspath(dst_path)
        lines.append(dst_path)
        lines.append(dst_path.replace(".jpg", ".txt"))

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def split_yolo_txt(filename, train_file="train.txt", test_file="test.txt", ratio=0.8):
    with open(filename, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    split_idx = int(ratio * len(lines))
    train = lines[:split_idx]
    test = lines[split_idx:]
    with open(train_file, "w") as f:
        f.writelines(train)
    with open(test_file, "w") as f:
        f.writelines(test)
