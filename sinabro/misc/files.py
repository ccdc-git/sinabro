import os

img_exts = [".jpg", ".png", ".jpeg"]


def is_img(filename: str):
    ext = os.path.splitext(filename)[1]
    return ext.lower() in img_exts
