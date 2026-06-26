import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import glob

#preprocessing the images in the dataset using normalization and standardization techniques

'''
# Load the image
image = Image.open(image_path)
# Convert the image to a NumPy array
image_array = np.array(image)
#print(image_array.shape)
#print(image_array)

print(image_array.max())
print(image_array.min())
# Display the image

plt.imshow(image_array, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.axis('off')  # Hide axes
plt.show()'''




def image(image_path):
    # Load the image
    i = 1

    for images in glob.iglob(f'{image_path}/*'):
      print(f"Image number {i}:")
      if (images.endswith(".png")):
        image = Image.open(images)
        image_array = np.array(image)

  
        print("Min:", image_array.min())
        print("Max:", image_array.max())
        i+=1
    return image_array

if __name__ == "__main__": 
    image_path = "/Users/farahjabeen/Desktop/XRAY_PROJECT/XRAY_ACTIONS"
    image = image(image_path)
    #print(image)

