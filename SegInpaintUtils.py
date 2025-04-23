from PIL import Image
from typing import Dict, List, Tuple, Set
import numpy as np
from SegmentUtils import cityscapes_labels

# look at SegmentUtils to see labels
TRANSIENT_LABELS = {11, 12}
STATIC_LABELS = {0, 1, 2, 3, 8, 9, 10}
CONTEXTUAL_LABELS = {13, 14, 15, 16, 17, 18}
NON_DECIDED_LABELS = {4, 5, 6, 7}

def aggregate_scene_stats(per_image_stats: List[Dict[int, int]]) -> Dict[int, Tuple[int, int]]:
    """
    Aggregates stats for one scene across all views.

    Args:
        per_image_stats: List of dicts per image, each mapping label_id -> area_in_pixels

    Returns:
        Dict of label_id -> (total_area_across_views, num_views_present)
    """
    agg_stats: Dict[int, Tuple[int, int]] = {}

    for img_stats in per_image_stats:
        for label_id, area in img_stats.items():
            total_area, count = agg_stats.get(label_id, (0, 0))
            agg_stats[label_id] = (total_area + area, count + 1)

    return agg_stats

def decide_labels_scene_level(
    scene_stats: Dict[int, Tuple[int, int]],
    num_views: int,
    area_threshold: float = 0.01,
    freq_threshold: float = 0.5
) -> Tuple[Set[int], Set[int]]:
    """
    Decide which labels to keep or remove based on scene-level statistics.
    Args:
        scene_stats: Dictionary of label_id -> (total_area, num_views_present)
        num_views: Total number of views in the scene
        area_threshold: Minimum area threshold for a label to be considered
        freq_threshold: Minimum frequency threshold for a label to be considered
    Returns:
        Tuple of sets: (labels_to_keep, labels_to_remove)
    """

    remove, keep = set(), set()

    for label_id, (total_area, view_count) in scene_stats.items():
        avg_area = total_area / view_count
        rel_freq = view_count / num_views

        if label_id in TRANSIENT_LABELS:
            remove.add(label_id)  
        elif label_id in STATIC_LABELS:
            keep.add(label_id)
        elif label_id in CONTEXTUAL_LABELS:
            if rel_freq > freq_threshold and avg_area > area_threshold:
                keep.add(label_id)
                if label_id == 14:
                    keep.add(13)
            else:
                remove.add(label_id)
    overlap = keep & remove
    for label in overlap:
        if label in remove:
            remove.remove(label)
    return keep, remove

def generate_prompt_from_labels(keep_labels: List[int], remove_labels: List[int]) -> str:
    """
    Generate a prompt and negative prompt based on the labels to keep and remove.
    Args:
        keep_labels: List of labels to keep
        remove_labels: List of labels to remove
    Returns:
        Tuple of strings: (prompt, negative_prompt)
    """
    
    keep_names = [cityscapes_labels[i] for i in keep_labels if i in cityscapes_labels]
    remove_names = [cityscapes_labels[i] for i in remove_labels if i in cityscapes_labels] 

    keep_content = ""
    for i, name in enumerate(keep_names):
        if i == len(keep_names) - 1:
            keep_content += f"{name}"
            break
        keep_content += f"{name}, " 

    prompt = f"An outdoor clean scene that features {keep_content}. Preserve the architectural integrity and surrounding textures. Ensure seamless blending with adjacent areas, maintaining natural transitions and consistent color coherence. Avoid new elements or altering the original composition." 
    negative_prompt = ""
    for i, name in enumerate(remove_names):
        if i == len(remove_names) - 1:
            negative_prompt += f"{name}, (people:1.3)"
            break
        negative_prompt += f"{name}"
    return prompt, negative_prompt


def square_padding(image, mask=None):
    image_np = np.array(image)
    height, weight = image_np.shape[:2]
    canvas_size = max(height, weight)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image_np.dtype)
    padded_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8) if mask is not None else None

    top, left = (canvas_size - height) // 2, (canvas_size - weight) // 2
    canvas[top:top+height, left:left+weight, :] = image_np

    if mask is not None:
        mask_np = np.array(mask)
        padded_mask[top:top+height, left:left+weight] = mask_np

    return Image.fromarray(canvas), padded_mask if mask is not None else None

def padded2normal(raw_image, padded_image):
    raw_np = np.array(raw_image)
    padded_np = np.array(padded_image)

    raw_height, raw_width = raw_np.shape[:2]

    padded_size = max(raw_height, raw_width)
    padded_current_height, padded_current_width = padded_np.shape[:2]
    scale_factor = padded_size / float(padded_current_height)  

    target_height = int(scale_factor * padded_current_height)
    target_width = int(scale_factor * padded_current_width)

    resized_image = Image.fromarray(padded_np).resize((target_width, target_height), Image.Resampling.LANCZOS)

    start_x = (target_width - raw_width) // 2
    start_y = (target_height - raw_height) // 2

    cropped_image = resized_image.crop((start_x, start_y, start_x + raw_width, start_y + raw_height))

    return cropped_image
