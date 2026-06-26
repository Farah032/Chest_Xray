import torch
import numpy as np
from PIL import Image


print("torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())


list1 = [1,2,3,4,5]
print("list:", list1)
t1 = torch.tensor(list1)
print("t1:", t1)

print('---'*5)
print("t1 value:",t1)
print("t1 shape:", t1.shape)
print("t1 dtype:", t1.dtype)
print("t1 device:",t1.device)
print("cpu to MPS",t1.to("mps"))
print("t1 numpy:", t1.numpy())

print('---'*5)
t2  = t1.to(torch.float32)
print("value t1 int64 to t1 float32:", t2)

print('----'*5)
np_array = np.array([[1.0,2.0],[3.0,4.0]])
t4 = torch.from_numpy(np_array)
print("tensor to numpy array:", t4)
back_to_numpy = t4.numpy()
print("back to numpy:", back_to_numpy)

print('----'*5)
np_array[0,0] = 999
print(t4)

print('----'*5)
t5 = torch.arange(12, dtype=torch.float32)
print("t5:", t5)
t5_2d = t5.reshape(3,4)
print("t5_2d:", t5_2d)
t5_3d = t5.reshape(2,2,3)
print("t5_3d:", t5_3d)

print('----'*5)
image_tensor = torch.zeros(1024,1024)
image_ready = image_tensor.unsqueeze(0).unsqueeze(0)
print("image_ready shape:", image_ready.shape)


print('----'*5)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("using device:", device)
t6 = torch.tensor([1.0,2.0,3.0])
t6_gpu = t6.to(device)
print("t6 on gpu:", t6_gpu.device)
#print("t6 gpu to numpy:", t6_gpu.numpy())

print('----'*5)
t6_gpu_cpu = t6_gpu.cpu()
print("t6: gpu to cpu",t6_gpu_cpu)
t6_cpu_numpy = t6_gpu_cpu.numpy()
print("t6: cpu to numpy", t6_cpu_numpy)



image = "/Users/farahjabeen/Desktop/XRAY_PROJECT/data/NIH/images_001/images/00000001_000.png"

img = Image.open(image)
print("image size:", img.size)
img_np = np.array(img)
print("convert to numpy:", img_np)
img_tensor  = torch.tensor(img_np, dtype=torch.float32)
print("convert to tensor type float32:", img_tensor)
image_normalized = img_tensor / 255.0
print("normalized image:", image_normalized)
img_ready = image_normalized.unsqueeze(0).unsqueeze(0)
print("ready for model input:", img_ready.shape)