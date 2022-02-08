from typing import *

def make_license(
        name: str = '',
        _id: int = 0,
        url: str = ''):
    return {'name': name, 'id': _id, 'url': url}

def make_info(
    contributor: str = '',
    date_created: str = '',
    description: str = '',
    url: str = '',
    version: str = '',
    year: str = '',
):
    return {
        'contributor': contributor,
        'date_created': date_created,
        'description': description,
        'url': url,
        'version': version,
        'year': year,
    }

def make_category(
    _id: int,
    name: str,
    supercategory: str = '',
):
    return {
        'id': _id,
        'name': name,
        'supercategory': supercategory,
    }

def make_image(
    _id: int,
    width: int,
    height: int,
    file_name: str,
    license: int = 0,
    flickr_url: str = '',
    coco_url: str = '',
    date_captured: int = 0,
):
    return { 
        'id': _id,
        'width': width,
        'height': height,
        'file_name': file_name,
        'license': license,
        'flickr_url': flickr_url,
        'coco_url': coco_url,
        'date_captured': date_captured,
    }

def make_annotation(
    _id: int,
    image_id: int,
    category_id: int,
    segmentation: List,
    area: float,
    bbox: List[float],
    iscrowd: int = 0,
    attributes = {'occluded': False},
):
    return { 
        'id': _id,
        'image_id': image_id,
        'category_id': category_id,
        'segmentation': segmentation,
        'area': area,
        'bbox': bbox,
        'iscrowd': iscrowd,
        'attributes': attributes,
    }