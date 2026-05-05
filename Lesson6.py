#Histogram and Ostu Thresholding
import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import regionprops, label


def histogram(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    flatten_image = image.flatten()
    return flatten_image

def otsu_thresholdings(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    body_threshold = image[image > 0] # Exclude background pixels (0) from threshold calculation
    value = threshold_otsu(body_threshold)
    binary_mask = image < value
    return binary_mask, value

def otsu_thresholdings_with_background(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    value = threshold_otsu(image)
    binary_mask = image < value  
    return binary_mask, value
'''
def otsu_thresholdings_normalized(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    value = threshold_otsu(image)

    binary_mask = image < value  
    white_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size

    print(f"White pixel ratio: {white_pixels / total_pixels:.3f}")
    area = image.shape[0] * image.shape[1] % 15
    centroid = image.shape[0] // 2, image.shape[1] // 2

    if white_pixels / total_pixels > 0.5:
        binary_mask = ~binary_mask

    elif area / centroid == 0:
        value = threshold_multiotsu(image)
    else:
        return binary_mask, value

def smart_threshold(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    print(f"Image size: {image.shape}")


    # Step 1 — Otsu
    value = threshold_otsu(image)
    binary_mask = image < value

    # Step 2 — Inversion check
    white_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size
    print(f"White pixel ratio: {white_pixels / total_pixels:.3f}")


    if white_pixels / total_pixels > 0.5:
        binary_mask = ~binary_mask


    # Step 3 — regionprops validation
    else:
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        region = max(regions, key=lambda r: r.area)

        area = region.area
        centroid_y, centroid_x = region.centroid
        
        
        mask_is_valid = (157286 <= area <= 419430) and \
                        (300 <= centroid_x <= 700) and \
                        (300 <= centroid_y <= 700)

        mask_is_valid = (area >= 140000) and \
                (250 <= centroid_x <= 750) and \
                (250 <= centroid_y <= 700)


        print(f"Area: {area}")
        print(f"Centroid X: {centroid_x:.1f}, Centroid Y: {centroid_y:.1f}")
        print(f"Mask is valid: {mask_is_valid}")

        # Step 4 — Multi-Otsu fallback
        if not mask_is_valid:
            thresholds = threshold_multiotsu(image)
            t1, t2 = thresholds
            binary_mask = (image > t1) & (image < t2)

    return binary_mask, value

'''
def smart_threshold(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)

    print(f"\nProcessing: {image_path}")
    print(f"Image size: {image.shape}")

    # Step 1 — Otsu Initial segmentation
    value = threshold_otsu(image)
    binary_mask = image < value

    # Step 2 — Inversion Measures how much area is white
    white_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size
    ratio = white_pixels / total_pixels
    print(f"White pixel ratio: {ratio:.3f}")

    #If white > 50%. mask is likely inverted
    if ratio > 0.5:
        binary_mask = ~binary_mask
        print("Inversion applied")

    # Step 3 — Validation ALWAYS 
    labeled_mask = label(binary_mask) #Assign IDs to connected components
    regions = regionprops(labeled_mask) #Extract properties

    if len(regions) == 0:  #No object found → threshold failed
        print("No regions found → Multi-Otsu fallback")
        #Switch to multi-level segmentation
        thresholds = threshold_multiotsu(image)
        print(f"Multi-Otsu thresholds: {thresholds}")
        t1, t2 = thresholds
        #Select middle intensity region
        return (image > t1) & (image < t2), value
    
    #Picks biggest object WHY:Body/lung region is largest meaningful structure

    region = max(regions, key=lambda r: r.area)

    #Properties used for validation

    area = region.area
    centroid_y, centroid_x = region.centroid

    #WHY: X-ray anatomy is usually: centered | large region. So you enforce: minimum area,centroid inside expected region

    print(f"Area: {area}")
    print(f"Centroid X: {centroid_x:.1f}, Centroid Y: {centroid_y:.1f}")

    mask_is_valid = (area >= 140000) and \
                    (250 <= centroid_x <= 750) and \
                    (250 <= centroid_y <= 700)

    print(f"Mask valid: {mask_is_valid}")

    # Step 4 — Fallback If mask looks wrong → fix it
    if not mask_is_valid:
        print("Applying Multi-Otsu fallback")
        thresholds = threshold_multiotsu(image)
        t1, t2 = thresholds
        # Use middle intensity class WHY:Often corresponds to soft tissue
        binary_mask = (image > t1) & (image < t2)

    return binary_mask, value


if __name__ == '__main__':

    input_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS"
    input_dir2 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/NormalizedAndStandardized_images"

    image_paths = glob.glob(input_dir2)
    #image_paths = glob.glob(input_dir2 + "/*.png")

    output_dir1 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Histogram_Images"
    output_dir2 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Thresholded_Images"
    output_dir3 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Thresholded_Images_with_background"
    output_dir4 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Thresholded_Images_with_normalized"





    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1) 
    for images in os.listdir(input_dir):
        if images.endswith(".png"):
            image_path = os.path.join(input_dir, images)
            result = histogram(image_path)
            save_path = os.path.join(output_dir1, f"histogram_{images}")
            plt.hist(result, bins=256, range=(0, 255), color='blue', alpha=0.7)
            plt.title("Histogram of the Image")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.savefig(save_path)
            plt.clf()  # Clear the figure for the next image
            print(f"Histogram saved as: {save_path}")

'''
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2) 
    for images in os.listdir(input_dir):
        if images.endswith(".png"):
            image_path = os.path.join(input_dir, images)
            binary_mask , threshold_value = otsu_thresholdings(image_path)
            save_path = os.path.join(output_dir2, f"thresholded_{images}")
            plt.imsave(save_path, binary_mask, cmap='gray')
            print(f"Thresholded image saved as: {save_path}")
            print(f"Without background threshold: {threshold_value}")



    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3) 
    for images in os.listdir(input_dir):
        if images.endswith(".png"):
            image_path = os.path.join(input_dir, images)
            binary_mask , threshold_value = otsu_thresholdings_with_background(image_path)
            save_path = os.path.join(output_dir3, f"thresholded_withbackground{images}")
            plt.imsave(save_path, binary_mask, cmap='gray')
            print(f"Thresholded image saved as: {save_path}")
            print(f"With background threshold:{threshold_value}")

'''       
if not os.path.exists(output_dir4):
        os.makedirs(output_dir4) 
for images in os.listdir(input_dir2):
    if images.startswith("normalized_image") and images.endswith(".png"):
        image_path = os.path.join(input_dir2, images)
        binary_mask , threshold_value = smart_threshold(image_path)
        save_path = os.path.join(output_dir4, f"thresholded_withnormalized{images}")
        plt.imsave(save_path, binary_mask, cmap='gray')
        print(f"Thresholded image saved as: {save_path}")
        print(f"With normalized threshold:{threshold_value}")
