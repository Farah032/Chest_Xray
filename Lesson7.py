#binary mask noise removal using morphological operations

import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, opening, closing
from skimage.measure import regionprops, label
import cv2



def binary_masks(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    #disk_size = image_array.shape[0] // 20  # Adjust the disk size based on image dimensions

    disk_size = image_array.shape[1] // 51
    structuring_element = disk(disk_size)
    opened_image = opening(image_array,structuring_element)
    closed_image = closing(opened_image, structuring_element)
    return  closed_image, disk_size

def morphology_operations_task(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    kernel = np.ones((5,5), np.uint8)

    eroded_image = cv2.erode(image_array, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)



    return dilated_image, kernel.shape

'''
     # Dilation
def morphology_operations_dilation(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    kernel = np.ones((5,5), np.uint8)

    dilated = cv2.dilate(image_array, kernel, iterations=1)

    return dilated, kernel.shape

'''



if __name__ == '__main__':

    #input_dir = "/Users/farahjabeen/Desktop/thesis/Code/X_ray"
    input_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Thresholded_Images_with_normalized"
    #input_dir2 = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/binary_masks"


    output_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/binary_masks"
    morphology_operations = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/morphology_operations"

    #output_dir_morphology_erosion = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/morphology_operations_erosion"
    #output_dir_morphology_dilation = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/morphology_operations_dilation"
    #output_dir_morphology_erosion_binarymask = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/morphology_operations_erosion_binarymask"
    #output_dir_morphology_dilation_binarymask = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/morphology_operations_dilation_binarymask"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    for images in os.listdir(input_dir):
        if images.endswith(".png"):
            image_path = os.path.join(input_dir, images)
            binary_mask , dsize = binary_masks(image_path)
            save_path = os.path.join(output_dir, f"binarymasks_{images}_{dsize}.png")
            plt.imsave(save_path, binary_mask, cmap='gray')
            print(f"Binary image saved as: {save_path} and disk_size:{dsize}" )

    if not os.path.exists(morphology_operations):
        os.makedirs(morphology_operations) 
    for images in os.listdir(output_dir):
        if images.endswith(".png"):
            image_path = os.path.join(output_dir, images)
            dilated_image , kernel = morphology_operations_task(image_path)
            save_path = os.path.join(morphology_operations, f"morphology_{images}_{kernel}.png")
            plt.imsave(save_path, dilated_image, cmap='gray')
            print(f"Morphology image saved as: {save_path} and disk_size:{kernel}" )

'''
    if not os.path.exists(output_dir_morphology_dilation_binarymask):
        os.makedirs(output_dir_morphology_dilation_binarymask)
    for images in os.listdir(output_dir_morphology_erosion_binarymask):
        if images.endswith(".png"):
            image_path = os.path.join(output_dir_morphology_erosion_binarymask, images)
            binary_mask , kernel = morphology_operations_dilation(image_path)
            save_path = os.path.join(output_dir_morphology_dilation_binarymask, f"morphology_{images}_{kernel}.png")
            plt.imsave(save_path, binary_mask, cmap='gray')
            print(f"Morphology image saved as: {save_path} and disk_size:{kernel}" )

'''
