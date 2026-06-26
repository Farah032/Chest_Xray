#Data Augmentation(flip,zoom,rotate)

from PIL import Image, ImageEnhance
import numpy as np
import glob
import os

def augment_image(image_path,augmentation_type):
    
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    
    if augmentation_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        augment_image = enhancer.enhance(1.5)   # Increase brightness by a factor of 1.5
                   
    elif augmentation_type == 'tilt':
        augment_image = image.rotate(15)
              
    elif augmentation_type == 'zoom':
        augment_image = image.crop((20,20,1020,1020))
        augment_image = augment_image.resize((224,224))
    
    else:
        print("Invalid augmentation type. Please choose from 'brightness', 'tilt', or 'zoom'.")
                
    return augment_image


if __name__ =='__main__':
    
    input_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS"
    output_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/Augmented_Images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    print("Choose augmentation type: 'brightness', 'tilt', or 'zoom'")
    augmentation_type = input("Enter augmentation type: ")

    for images in os.listdir(input_dir):
        if images.endswith(".png"):
            image_path = os.path.join(input_dir, images)
            augmented_image = augment_image(image_path, augmentation_type)
            save_path = os.path.join(output_dir, f"augmented_{augmentation_type}_{images}")
            augmented_image.save(save_path)
            print(f"Augmented image saved as: {save_path}")
     

