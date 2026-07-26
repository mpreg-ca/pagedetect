import numpy as np
from PIL import Image


def merge_pages_rtl(img_right, img_left):
    if type(img_right) is Image:
        img_right_np = np.asarray(img_right)
        img_left_np = np.asarray(img_left)
    else:
        img_right_np = img_right
        img_left_np = img_left

    merged = np.hstack([img_left_np[:, -32:], img_right_np[:, :32]])
    merged = Image.fromarray(merged, mode="L")
    return merged


def extract_edges(img, edge_width=64):
    if type(img) is Image:
        img_np = np.asarray(img)
    else:
        img_np = img
    left_edge = Image.fromarray(img_np[:, :edge_width], mode="L")
    right_edge = Image.fromarray(img_np[:, -edge_width:], mode="L")
    return left_edge, right_edge
