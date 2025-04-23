from diffusers import StableDiffusionInpaintPipeline
import torch
import numpy as np
from PIL import Image
import cv2

def load_difussion_model():
    """
    Load the Stable Diffusion inpainting model.
    """
    model = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    model = model.to("cuda")
    return model

def inpaint_img(image, mask, difussion_model, prompt, negative_prompt=None):
    """
    Inpaint the image using the given mask and prompt.
    """
    mask = (mask * 255).astype(np.uint8)

    image_pil = image if isinstance(image, Image.Image) else Image.fromarray(image.astype(np.uint8))
    mask_pil = Image.fromarray(mask)

    inpainted = difussion_model(prompt=prompt, negative_prompt=negative_prompt, image=image_pil, mask_image=mask_pil, guidance_scale = 8.0).images[0]
    return inpainted


def get_all_mask_regions(mask):
    """
    Get all mask regions from the binary mask.
    Args:
        mask (np.ndarray): A binary mask with values 0 or 1.
    Returns:
        List of tuples: Each tuple contains (left, top, right, bottom) coordinates of the bounding box.
        List of np.ndarray: Each element is a mask for the corresponding region.
    """
    num_labels, labels = cv2.connectedComponents(mask)
    regions = []
    masks = []

    for label in range(1, num_labels):  
        region_mask = (labels == label).astype(np.uint8)
        coords = cv2.findNonZero(region_mask)
        x, y, w, h = cv2.boundingRect(coords)

        cx = x + w // 2
        cy = y + h // 2

        size = 512
        if w * 1.2 > size or h * 1.2 > size:
            size = 768
        if w * 1.2 > size or h * 1.2 > size:
            return [(0, 0, mask.shape[1], mask.shape[0])], [region_mask]

        half = size // 2
        top = max(cy - half, 0)
        left = max(cx - half, 0)
        bottom = min(top + size, mask.shape[0])
        right = min(left + size, mask.shape[1])

        top = bottom - size if bottom - size >= 0 else 0
        left = right - size if right - size >= 0 else 0

        regions.append((left, top, right, bottom))
        masks.append(region_mask)

    return regions, masks