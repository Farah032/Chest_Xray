import PIL.Image as Image
import numpy as np
import glob

#resizing and standardizing the images in the dataset

def preprocess(image_path):
    result = []
    glob_path = f'{image_path}/*'
    for images in glob.iglob(glob_path):
        #print(f"Processing image: {images}")
        if images.endswith(".png"):
              image = Image.open(images)
              image_array = image.convert('L')  # Convert to grayscale
              image_array = image_array.resize((224,224))
              image_array = np.array(image_array)
              standardize_image = (image_array - image_array.mean())/image_array.std()
              result.append((standardize_image, images))
    return result




if __name__ =='__main__':
    image_path = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS"
    result = preprocess(image_path)
    print(result)
