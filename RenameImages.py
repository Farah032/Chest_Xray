import os
import shutil

source_folder = "/Users/farahjabeen/Desktop/thesis/Code/chest_xray/Images"
target_folder = "/Users/farahjabeen/Desktop/thesis/xray_images/"

os.makedirs(target_folder, exist_ok=True)

files = os.listdir(source_folder)
files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
files.sort()

for i, file in enumerate(files, start=1):
    src_path = os.path.join(source_folder, file)
    dst_path = os.path.join(target_folder, f"image_{i}.png")
    
    shutil.copy(src_path, dst_path)

print("Images copied and renamed successfully.")