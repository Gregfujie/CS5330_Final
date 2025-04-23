from tkinter import Image
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image

# labels from the pretrained model config
# https://huggingface.co/facebook/mask2former-swin-small-cityscapes-panoptic/blob/main/config.json
cityscapes_labels = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "unlabeled",
}

cityscapes_palette = [
    (128, 64,128),  # 0 road
    (244, 35,232),  # 1 sidewalk
    ( 70, 70, 70),  # 2 building
    (102,102,156),  # 3 wall
    (190,153,153),  # 4 fence
    (153,153,153),  # 5 pole
    (250,170, 30),  # 6 traffic light
    (220,220,  0),  # 7 traffic sign
    (107,142, 35),  # 8 vegetation
    (152,251,152),  # 9 terrain
    ( 70,130,180),  # 10 sky
    (220, 20, 60),  # 11 person
    (255,  0,  0),  # 12 rider
    (  0,  0,142),  # 13 car
    (  0,  0, 70),  # 14 truck
    (  0, 60,100),  # 15 bus
    (  0, 80,100),  # 16 train
    (  0,  0,230),  # 17 motorcycle
    (119, 11, 32),  # 18 bicycle
    (  0,  0,  0),  # 19 unlabeled (black)
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segment_model():
    """
    Loads the segmentation model and class names.
    """
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")

    return processor, model, cityscapes_labels, cityscapes_palette

def segmentation(image: Image.Image, processor, model, device):
    """
    Segments the input image using the given segmentation model.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return processed['segmentation'].cpu().numpy(), processed['segments_info']


def refine_mask(mask: np.ndarray, min_area: int = 3000, kernel_size: int = 5) -> None:
    """
    Refines a binary mask in-place using morphological operations and component filtering.

    Parameters:
        mask (np.ndarray): A 2D binary NumPy array with values 0 or 1.
        min_area (int): Minimum area to consider for components (useful in multiple patch mode).
        kernel_size (int): Size of the structuring element.
    """
    assert mask.ndim == 2
    assert set(np.unique(mask)).issubset({0, 1})

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # has a background label by default
    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return

    elif num_labels == 2:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if np.sum(cleaned) < min_area:
            mask.fill(0)
            return
    else:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        new_num_labels, new_labels = cv2.connectedComponents(closed)
        if new_num_labels <= 1:
            mask.fill(0)
            return

        components = [(i, np.sum(new_labels == i)) for i in range(1, new_num_labels)]

        if not components:
            mask.fill(0)
            return

        largest_label = max(components, key=lambda x: x[1])[0]
        cleaned = (new_labels == largest_label).astype(np.uint8)

    mask[:] = cleaned[:]


def find_adjacent_labels(seg_map, target_label) -> set:
    """
    Finds labels adjacent to the target label in the segmentation map.
    Args:
        seg_map (np.ndarray): Segmentation map.
        target_label (int): Target label to find adjacent labels for.
    Returns:
        set: Set of adjacent labels.
    """

    target_mask = (seg_map == target_label).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(target_mask, kernel, iterations=1)

    border = dilated - target_mask

    adjacent_labels = set(np.unique(seg_map[border == 1]))
    adjacent_labels.discard(target_label)

    return adjacent_labels