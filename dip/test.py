# import numpy as np
# img = np.random.randn(3,512,512)
# print(img)
# print(*img)
# print(img.shape)
# print(*img.shape)
import cv2
import numpy as np
import os
import sys

dir = os.getcwd()
sys.path.append(dir)

from dip.denoise import read_image_np, cv2_to_torch, calculate_psnr


import torch
import torch.nn.functional as F

def calculate_psnr_tensor(image1, image2):
    """
        image1: tensorFloat
        image2: tensorFloat
        return: tensorFloat
    """
    mse = F.mse_loss(image1, image2)
    # 计算PSNR
    # psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)
    psnr = 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))
    return psnr


def psnr(image1, image2):
    # 转换图像数据类型为浮点型
    print(type(image1))
    image1 = image1.float()
    image2 = image2.float()
    image1 = image1.type(torch.cuda.FloatTensor)
    image2 = image2.type(torch.cuda.FloatTensor)
    print(type(image1))

    # 计算均方差
    mse = F.mse_loss(image1, image2)

    # 计算PSNR
    psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)
    return psnr

# 在CUDA上计算PSNR
def psnr_cuda(image1, image2):
    # 将图像数据移到GPU上
    image1 = image1.cuda()
    image2 = image2.cuda()

    # 在CUDA上计算PSNR
    # psnr_value = calculate_psnr_tensor(image1, image2)
    psnr_value = psnr(image1,image2)

    return psnr_value

# 创建两个随机图像（示例）
# image1 = torch.rand(3, 256, 256)  # 图像1
# image2 = torch.rand(3, 256, 256)  # 图像2

data_file1 = "./dip/testset/CSet9/kodim01.png"
data_file2 = "./dip/testset/Cset9/kodim02.png"
image1 = read_image_np(data_file1)
image2 = read_image_np(data_file2)

image1 = np.clip(image1, 0, 255)
image2 = np.clip(image2, 0, 255)
dtype = torch.cuda.FloatTensor
image1_torch = torch.from_numpy(image1).type(dtype)
image2_torch = torch.from_numpy(image2).type(dtype)
# image1_torch =  cv2_to_torch(image1)
# image2_torch = cv2_to_torch(image2)

psnr_value = calculate_psnr(image1, image2)
print("psnr",psnr_value)



# 计算PSNR
psnr_value = psnr_cuda(image1_torch, image2_torch)
print("PSNR:", psnr_value.item())


# def read_image_np(fname):
#     img_np = cv2.imread(fname, -1) # h,w,r  #灰度图，则为2通道，彩色图，则为3通道
#     if len(img_np.shape) == 2:
#         # 灰度图
#         print("[*] read GRAY image.")
#         img_np = img_np[np.newaxis,:] # 增加一维，保持三维 r(1),g,b, 灰度图
#     else:
#         print("[*] read COLOR image.")
#         img_np = img_np.transpose([2, 0, 1]) # r,g,b
#     img_np.astype(np.float64)
#     return img_np