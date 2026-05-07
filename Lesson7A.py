import os
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_opening, binary_closing


# --------------------------------------------------
# STEP 1 — HISTOGRAM FUNCTION
# --------------------------------------------------

def compute_histogram(image_path):

    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    return image_array.flatten()


# --------------------------------------------------
# STEP 2 — SMART THRESHOLDING (ROBUST)
# --------------------------------------------------

def smart_threshold(image_path):

    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # Otsu threshold
    threshold_value = threshold_otsu(image_array)
    binary_mask = image_array > threshold_value

    # Inversion check
    ratio = np.sum(binary_mask) / binary_mask.size

    if ratio > 0.5:
        binary_mask = ~binary_mask

    # Region validation
    labeled = label(binary_mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        t1, t2 = threshold_multiotsu(image_array)
        binary_mask = (image_array > t1) & (image_array < t2)
        return binary_mask, threshold_value

    largest_region = max(regions, key=lambda r: r.area)

    area = largest_region.area
    cy, cx = largest_region.centroid

    valid = (
        area >= 140000 and
        250 <= cx <= 750 and
        250 <= cy <= 700
    )

    # fallback if invalid
    if not valid:
        t1, t2 = threshold_multiotsu(image_array)
        binary_mask = (image_array > t1) & (image_array < t2)

    return binary_mask, threshold_value


# --------------------------------------------------
# STEP 3 — CLEAN MASK (MORPHOLOGY)
# --------------------------------------------------

def clean_mask(binary_mask):

    disk_size = binary_mask.shape[1] // 51
    if disk_size < 1:
        disk_size = 1

    selem = disk(disk_size)

    opened = binary_opening(binary_mask, selem)
    cleaned = binary_closing(opened, selem)

    return cleaned, disk_size


# --------------------------------------------------
# STEP 4 — PROCESS FULL PIPELINE
# --------------------------------------------------

def process_pipeline(input_dir, output_hist, output_masks):

    os.makedirs(output_hist, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)

    for image_name in os.listdir(input_dir):

        if not image_name.endswith(".png"):
            continue

        image_path = os.path.join(input_dir, image_name)

        # -------------------------
        # HISTOGRAM
        # -------------------------
        hist_data = compute_histogram(image_path)

        hist_path = os.path.join(
            output_hist,
            f"hist_{image_name}"
        )

        plt.hist(hist_data, bins=256, range=(0, 255))
        plt.title("Pixel Intensity Histogram")
        plt.savefig(hist_path)
        plt.clf()

        # -------------------------
        # THRESHOLDING
        # -------------------------
        binary_mask, thr = smart_threshold(image_path)

        # -------------------------
        # CLEANING
        # -------------------------
        cleaned_mask, disk_size = clean_mask(binary_mask)

        # -------------------------
        # SAVE MASK
        # -------------------------
        save_path = os.path.join(
            output_masks,
            f"mask_{image_name}"
        )

        plt.imsave(save_path, cleaned_mask, cmap='gray')

        print(f"\nProcessed: {image_name}")
        print(f"Threshold: {thr}")
        print(f"Disk size: {disk_size}")
        print(f"Saved mask: {save_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    input_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/NormalizedAndStandardized_images"

    output_hist = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Histogram_Images"

    output_masks = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Final_Clean_Masks"

    process_pipeline(input_dir, output_hist, output_masks)