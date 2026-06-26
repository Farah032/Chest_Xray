import numpy as np
import PIL.Image as Image
import os


#preprocessing the images in the dataset using normalization and standardization techniques
def preprocess_image(image_path, method):
     image = Image.open(image_path)
     image_array = np.array(image)

     if method == 'normalize':
          minarr = np.min(image_array)
          maxarr = np.max(image_array)
          normalize_image = (image_array - minarr)/ (maxarr - minarr)
          return normalize_image
     
     elif method == 'standardize':
          meanarr = np.mean(image_array)
          stdarr = np.std(image_array)
          standardize_image = (image_array - meanarr)/stdarr
          return standardize_image
     
     else:
            raise ValueError("Invalid method. Choose 'normalize' or 'standardize'.")
'''
def normalize_image(image_path, method):
    image = Image.open(image_path)
    image_array = np.array(image)
    minarr = image_array.min()
    maxarr = image_array.max()
    return (image_array-minarr)/(maxarr-minarr)


def standardize_image(image_path, method):
    image = Image.open(image_path)
    image_array = np.array(image)
    meanarr = image_array.mean()
    stdarr = image_array.std()
    return (image_array-meanarr)/stdarr
    '''




if __name__ == "__main__":
     input_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS"
     output_dir = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS/NormalizedAndStandardized_images"

print("Choose a method: normalize or standardize")
method = input().strip().lower()   




if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
for images in os.listdir(input_dir):
      if images.endswith(".png"):
          image_path = os.path.join(input_dir, images)
          preprocess_images = preprocess_image(image_path, method)
          result = preprocess_image(image_path, method)
          
          if method == 'normalize':
               print(result)
               if result.min() ==0 and result.max() == 1:
                    save_path = os.path.join(output_dir, f"normalized_{images}")

                    images = Image.fromarray((result * 255).astype(np.uint8)).save(save_path)
                    print("Normalization successful!")
               else:
                    print("Normalization failed.")

          elif method == 'standardize':
               print(result)
               if np.isclose(result.mean(),0) and np.isclose(result.std(),1):
                    save_path = os.path.join(output_dir, f"standardize{images}")

                    images = Image.fromarray(((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)).save(save_path)
                    print("Standardization successful!")
               else:
                    print("Standardization failed.")
                         
          else:
               print("Invalid method. Please choose 'normalize' or 'standardize'.")   
        