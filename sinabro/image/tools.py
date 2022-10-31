import math

import cv2
import numpy as np
import imutils


def rotate_dot(dot, degree, height=0, width=0):
    x, y = dot
    degree = math.radians(degree)
    sin_theta = math.sin(degree)
    cos_theta = math.cos(degree)
    x_offset = max(height * sin_theta, 0) - min(width * cos_theta, 0)
    y_offset = -min(height * cos_theta, 0) - min(width * sin_theta, 0)
    return (
        int(x * cos_theta - y * sin_theta + x_offset),
        int(x * sin_theta + y * cos_theta + y_offset),
    )


def rotate_dots(dots, degree, height=0, width=0):
    return np.array([rotate_dot(dot, degree, height, width) for dot in dots])


def find_bbox(segmentation):
    xmin, ymin = segmentation[0]
    xmax, ymax = segmentation[0]
    for dot in segmentation:
        x, y = dot
        if xmin > x:
            xmin = x
        elif xmax < x:
            xmax = x
        if ymin > y:
            ymin = y
        elif ymax < y:
            ymax = y
    return (xmin, ymin, xmax, ymax)


def put_into_image(background, foreground, segmentation, c_x, c_y, r_size, degree):
    # rotate foreground image and segmentation with {rotate} degree
    assert c_x < background.shape[1]
    assert c_y < background.shape[0]
    segmentation = np.array(
        [
            rotate_dot(dot, degree, foreground.shape[0], foreground.shape[1])
            for dot in segmentation
        ]
    )
    foreground = imutils.rotate_bound(foreground, degree)

    h, w, c = foreground.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.fillPoly(mask, [segmentation], 255)

    # crop bbox from foreground
    back_h, back_w, _ = background.shape
    xmin, ymin, xmax, ymax = find_bbox(segmentation)
    img_w = xmax - xmin
    img_h = ymax - ymin

    size_fact = max(back_h / img_h, back_w / img_w) * r_size

    left = int((c_x - img_w * size_fact / 2))
    right = int((c_x + img_w * size_fact / 2))
    top = int((c_y - img_h * size_fact / 2))
    bottom = int((c_y + img_h * size_fact / 2))

    if left < 0:
        xmin = int(xmin - left / size_fact)
        left = 0

    if top < 0:
        ymin = int(ymin - top / size_fact)
        top = 0

    if right > background.shape[1]:
        xmax = int(xmax - (right - back_w) / size_fact)
        right = background.shape[1]

    if bottom > background.shape[0]:
        ymax = int(ymax - (bottom - back_h) / size_fact)
        bottom = background.shape[0]

    cropped_img = foreground[ymin:ymax, xmin:xmax]
    cropped_mask = mask[ymin:ymax, xmin:xmax]

    dst = draw_img(background, cropped_img, left, top, right, bottom, mask=cropped_mask)
    return dst, (left, top, right, bottom)


def draw_img(
    dst: np.ndarray, src: np.ndarray, left, top, right, bottom, mask: np.ndarray = None
):
    if mask is None:
        if src.shape[2] == 4:
            _, mask = cv2.threshold(src[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        else:
            mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
    _, _, src_c = src.shape
    width = right - left
    height = bottom - top

    dst = np.copy(dst)
    resized_src = cv2.resize(src, dsize=(width, height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_AREA)
    inv_mask = cv2.bitwise_not(resized_mask)

    if src_c == 4:
        resized_src = cv2.cvtColor(resized_src, cv2.COLOR_BGRA2BGR)

    roi = dst[top:bottom, left:right]
    masked_fg = cv2.bitwise_and(resized_src, resized_src, mask=resized_mask)
    masked_bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
    added = masked_fg + masked_bg
    dst[top:bottom, left:right] = added

    return dst
