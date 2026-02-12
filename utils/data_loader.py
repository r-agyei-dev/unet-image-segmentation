# Imports
import numpy as np
import tifffile as tiff
from patchify import patchify


# Load TIFF images
def load_tiff_data(image_path, mask_path):
    large_image_stack = tiff.imread(image_path)
    large_mask_stack = tiff.imread(mask_path)
    return large_image_stack, large_mask_stack


# Patchify and preprocess images and masks
def create_patches(large_image_stack, large_mask_stack, patch_size=256):

    all_img_patches = []

    # Process images
    for img in range(large_image_stack.shape[0]):
        large_img = large_image_stack[img]
        patches_img = patchify(large_img, (patch_size, patch_size), step=patch_size)

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :]
                single_patch_img = single_patch_img.astype("float32") / 255.0
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)
    images = np.expand_dims(images, -1)


    # Process masks
    all_mask_patches = []

    for img in range(large_mask_stack.shape[0]):
        large_mask = large_mask_stack[img]
        patches_mask = patchify(large_mask, (patch_size, patch_size), step=patch_size)

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = single_patch_mask / 255
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)
    masks = np.expand_dims(masks, -1)

    return images, masks

