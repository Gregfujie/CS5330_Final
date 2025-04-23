from Inpaint import get_all_mask_regions
from SegmentUtils import find_adjacent_labels, load_segment_model, refine_mask, segmentation, device
import numpy as np

def segment_count(image, segment_model):
    """
        Segments the input image and collect label statistics.
    """
    image_processor, model, _, _ = segment_model

    segmentation_map, segments_info = segmentation(image, image_processor, model, device=device)

    counter = {}

    for seg in segments_info:
        if seg['score'] <= 0.91:
            continue

        label_id = seg['label_id']
        mask = (segmentation_map == seg['id']).astype(np.uint8)
        if label_id not in counter:
            counter[label_id] = mask.sum()
        else:
            counter[label_id] += mask.sum()
    return counter

def segment_img(image, segment_model, labels2remove):
    """
    Segments the input image using the given segmentation model.
    """
    image_processor, model, _, _ = segment_model
    image_np = np.array(image)

    segmentation_map, segments_info = segmentation(image, image_processor, model, device=device)
    transient_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    mask_list = []
    bbox_list = []
    adjactent_labels_list = []
    regional = True

    for seg in segments_info:
        if seg['score'] <= 0.91:
            continue
        label_id = seg['label_id']
        if label_id in labels2remove:
            mask = (segmentation_map == seg['id']).astype(np.uint8) # 01 Map
            refine_mask(mask, 2000)
            transient_mask = np.maximum(transient_mask, mask)

            if regional:
                bboxes, _ = get_all_mask_regions(mask)
                if len(bboxes) == 0:
                    continue
                bbox = bboxes[0]
                if bbox[2] - bbox[0] == image_np.shape[1] and bbox[3] - bbox[1] == image_np.shape[0]:
                    regional = False 
                
                adjactent_labels = find_adjacent_labels(segmentation_map, seg['id'])
                adjactent_label_ids = []
                for label in adjactent_labels:
                    adjactent_label_ids.append(segments_info[label - 1]['label_id'])
                
                adjactent_labels_list.append(adjactent_label_ids)   
                mask_list.append(mask)
                bbox_list.append(bbox)

    result_image = np.zeros_like(image_np)
    result_image[transient_mask == 0] = image_np[transient_mask == 0]  # Keep only non-transient areas  

    if not regional:
        bbox_list = [(0, 0, image_np.shape[1], image_np.shape[0])]
        mask_list = [transient_mask]
        adjactent_labels_list = [[]] 

    return result_image, transient_mask, mask_list, bbox_list, adjactent_labels_list

if __name__ == "__main__":
    from PIL import Image
    image = Image.open("./OursTest/0029.jpg").convert("RGB")

    from matplotlib import pyplot as plt
    processor, model, cityscapes_labels, cityscapes_palette = load_segment_model()
    model.to(device)
    result, _ = segment_img(image, (processor, model, cityscapes_labels, cityscapes_palette))

    plt.imshow(result)
    plt.axis('off')
    plt.show()