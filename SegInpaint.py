import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from Inpaint import get_all_mask_regions, inpaint_img, load_difussion_model
from SegInpaintUtils import square_padding, padded2normal, aggregate_scene_stats, decide_labels_scene_level, generate_prompt_from_labels
from Segment import segment_count, segment_img
from SegmentUtils import load_segment_model,  device

def segment_inpaint(input_folder, seg_folder, mask_folder, inpainted_folder, segment_model, difussion_model):

    counter_list = []
    count = 0

    # summarize the scene
    for file in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file)
        if os.path.isfile(input_path):
            image = Image.open(input_path).convert("RGB")
            counter = segment_count(image, segment_model)
            counter_list.append(counter)
            count += 1
    
    agg_stat = aggregate_scene_stats(counter_list)
    keep, remove = decide_labels_scene_level(agg_stat, count)
    prompt, negative_prompt = generate_prompt_from_labels(list(keep), list(remove))

    print(f"Negative Prompt: {negative_prompt}")

    # segment and inpaint
    for file in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file)
        if os.path.isfile(input_path):
            
            image = Image.open(input_path).convert("RGB")

            # Segment
            segmented_image, mask, mask_list, bbox_list, adjactent_labels_list = segment_img(image, segment_model, remove)

            mask_path = os.path.join(mask_folder, f"{os.path.splitext(file)[0]}.npy")
            np.save(mask_path, mask)

            # Save segmented imgs
            Image.fromarray(segmented_image).save(os.path.join(seg_folder, file))

            inpainted_image = segmented_image.copy()
            if mask.sum() == 0: 
                inpaint_path = os.path.join(inpainted_folder, file)
                result = Image.fromarray(inpainted_image)
                result.save(inpaint_path)
                continue

            # inpaint based on different conditions
            for i in range(len(mask_list)):
                bbox, label_mask, adj_labels = bbox_list[i], mask_list[i], adjactent_labels_list[i]
                if bbox[2] - bbox[0] == segmented_image.shape[1] or bbox[3] - bbox[1] == segmented_image.shape[0]:
                    pre_inpaint, padded_mask = square_padding(segmented_image, mask)
                    inpainted = inpaint_img(pre_inpaint, padded_mask, difussion_model, prompt, negative_prompt)
                    if inpainted is not None:
                        inpainted = padded2normal(image, inpainted)
                        inpainted = np.array(inpainted)
                        inpainted_image[label_mask == 1] = inpainted[label_mask == 1]
                else:
                    pre_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    pre_inpaint = segmented_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    temp_k = []
                    for adj_label in adj_labels:
                        if adj_label in keep:
                            temp_k.append(adj_label)
                    regional_prompt, _ = generate_prompt_from_labels(temp_k, list(remove))
                    print(input_path)
                    print(regional_prompt)
                    inpainted = inpaint_img(pre_inpaint, pre_mask, difussion_model, regional_prompt, negative_prompt)
                    if inpainted is not None:
                        inpainted = inpainted.resize((bbox[3] - bbox[1], bbox[2] - bbox[0]), Image.Resampling.LANCZOS)
                        inpainted = np.array(inpainted)
                        resized_inpainted = np.zeros_like(inpainted_image)
                        resized_inpainted[bbox[1]:bbox[3], bbox[0]:bbox[2]] = inpainted
                        inpainted_image[label_mask == 1] = resized_inpainted[label_mask == 1]

            inpaint_path = os.path.join(inpainted_folder, file)
            result = Image.fromarray(inpainted_image)
            result.save(inpaint_path)


if __name__ == "__main__":
    import warnings 
    warnings.filterwarnings("ignore", message='`label_ids_to_fuse` unset. No instance will be fused.')

    image_root = '/mnt/d/dataset/1080p'
    input_folder = image_root + '/images'
    segmented_output_folder = image_root + '/segmented_imgs'
    mask_output_folder = image_root + '/img_masks'
    inpaint_img_folder = image_root + '/inpainted_imgs'
    
    if not os.path.exists(input_folder):
        print(f"folder {input_folder} does not exist.")
        import sys
        sys.exit()
    
    if not os.path.exists(segmented_output_folder):
        os.makedirs(segmented_output_folder)

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    
    if not os.path.exists(inpaint_img_folder):
        os.makedirs(inpaint_img_folder)

    
    segment_model = load_segment_model()
    segment_model[1].to(device)
    print('Segmentation model Loaded')

    difussion_model = load_difussion_model()
    print('Difussion model Loaded')
    
    print("Start processing images")
    segment_inpaint(input_folder, segmented_output_folder, mask_output_folder, inpaint_img_folder, segment_model,difussion_model)
    print(f"Finished.")