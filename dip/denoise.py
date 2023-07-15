import torch
import torch.functional as F
import torch.nn as nn
import torch.distributed as dist

import cv2
import math

import numpy as np
import pandas as pd
import argparse

import json

import os
import sys

from torch.multiprocessing  import Process
import torch.multiprocessing as mp
import threading
import copy

import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# print(torch.cuda.is_available())


import models
import additional_utils

dir = os.getcwd()
sys.path.append(dir)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read_image
def read_image_np(fname):
    img_np = cv2.imread(fname, -1) # h,w,r  #灰度图，则为2通道，彩色图，则为3通道
    # print(img_np)
    if len(img_np.shape) == 2:
        # 灰度图
        print("[*] read GRAY image.")
        img_np = img_np[np.newaxis,:] # 增加一维，保持三维 r(1),g,b, 灰度图
    else:
        print("[*] read COLOR image.")
        img_np = img_np.transpose([2, 0, 1]) # r,g,b
    img_np.astype(np.float64)
    return img_np

# np2torch
def cv2_to_torch(cv2_img, dtype=None):
    img_torch = torch.from_numpy(cv2_img)[None, :]
    if dtype == None:
        out = img_torch.float() / 255.0
    else:
        out = img_torch.type(dtype) / 255.0
    return out
def torch2torch255(out_torch):
    # out_torch = out_torch.clamp(out_torch, 0, 1)
    out_torch = torch.clamp(out_torch, 0, 1)
    out_torch = torch.squeeze(out_torch * 255).type(torch.uint8)
    return out_torch

# 随机随机噪声
def get_noise(noise_shape,noise_type='u',var=1./10):
    # net_input 的shape
    shape = [noise_shape[0],noise_shape[1],noise_shape[2],noise_shape[3]]
    # print("shape",shape)
    net_input = torch.zeros(shape)
    # fill noise_val
    if noise_type == 'u':
        net_input.uniform_()
    elif noise_type == 'n':
        net_input.normal_()
    net_input *= var
    # print(net_input)
    return net_input

# calclue_psnr (np)
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# calculate_psnr (tensor)
def calculate_psnr_tensor(image1, image2):
    """
        image1: tensorFloat
        image2: tensorFloat
        return: tensorFloat
    """
    mse = torch.nn.functional.mse_loss(image1, image2)
    # mse = F.mse_loss(image1, image2)
    # 计算PSNR
    # psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)
    psnr = 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))
    return psnr

# # dip_sure/ eSURE_uniform loss /ealry_stop


# 计算差异
def divergence(net_input, out):
    dtype = torch.cuda.FloatTensor
    b_prime = torch.randn_like(net_input).type(dtype) # shape一样的均值为0，方差为1的正态分布
    nh_y = torch.sum(b_prime * out, dim =[1, 2, 3])
    vector = torch.ones(1).to(out)
    divergence = b_prime * \
                     torch.autograd.grad(nh_y, net_input, grad_outputs=vector, retain_graph=True, create_graph=True)[0]

    return divergence

# 斯坦无偏估计
def SURE(output, target, divergence, sigma):
    batch, c, h, w = output.shape
    divergence = divergence * sigma
    mse = (output - target) ** 2
    esure = mse + 2*divergence - sigma
    esure = torch.sum(esure)
    # esure = esure if self.reduction == "sum" else esure / (h * w * c)
    esure = esure 
    esure = esure / (h * w * c)
    return esure


def SURE_loss(net_input_ep, output, target, args):
    # 计算divergence
    divergence_val = divergence(net_input_ep, output)
    # sigma_z = args.sigma * args.sigma_z
    # 计算vary
    eps_y = torch.ones([1], device=args.device).reshape([-1,1,1,1]) * args.sigma /255.0
    vary = (eps_y) ** 2
    total_loss = SURE(output, target, divergence_val, vary ) 
    return total_loss


def load_image_pair(fname, task, args):
    if task == "denoising":
        img_np  = read_image_np(fname)
        # noise_np   degrage_image
        if "mean" in fname:
            noisy_np = read_image_np(fname.replace("mean", "real"))
        else:
            noisy_np = img_np + np.random.randn(*img_np.shape) * args.sigma # 加噪声
    print("[!] clean image domain : [%.2f, %.2f]" %(img_np.min(), img_np.max()))
    print("[!] noisy image domain : [%.2f, %.2f]" %(noisy_np.min(), noisy_np.max()))
    return img_np, noisy_np

def get_net_input_ep(net_input_saved, args, uniform_sigma=True):
    sigma_z = args.sigma * args.sigma_z
    if sigma_z > 0 or uniform_sigma:
        if uniform_sigma:
            eSigma = np.random.uniform(0, sigma_z) / 255.0
        else:
            eSigma = sigma_z / 255.0
        net_input_ep = net_input_saved + torch.randn_like(net_input_saved).type(args.dtype) * eSigma # net_input 
    net_input_ep = net_input_ep.requires_grad_() # 自动计算梯度 输入数据
    return net_input_ep





def image_restorazation(file, args, indi): # args
    torch.manual_seed(0) # GPU 随机种子
    np.random.seed(0)
    

    # args = argparse.ArgumentParser()

    # # cudnn 设置
    # if torch.cuda.is_available():
    #     # 指定显卡
    #     device = torch.device("cuda:0") # 之后可设置并行
    #     args.device = device
    #     # 寻找适合卷积实现的方法，网络加速
    #     torch.backends.cudnn.enabled = True
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    #     args.dtype = torch.cuda.FloatTensor # 数据类型： FloatTensor
    # else:
    #     device = torch.device("cpu")
    #     args.device = device
    #     args.dtype = torch.FloatTensor

    
    # args.dip_type = "eSURE_uniform"
    # args.net_type = "unet_indi" # s2s, unet_indi, s2s_test
    # args.device = torch.device("cuda:0")
    # args.gray = False
    # args.sigma = 50
    # args.dtype = torch.cuda.FloatTensor
    # args.task_type = "denoising"
    # args.desc = "_unet_indi_sigma50"
    # args.exp_tag = "single_image"

    # # net_set
    # args.optim = "RAdam"
    # args.lr = 0.1
    # args.force_steplr = False
    # args.beta1 = 0.9
    # args.beta2 = 0.999
    # args.sigma_z = 0.5
    # args.epoch = 100

    # # # dip
    # args.running_avg_ratio = 0.99
    t1 = time.time()
    stat = {}
    stat["net_indi"] = indi.id
    task_type = args.task_type
    sigma = args.sigma # noisy_degree
    dtype = args.dtype  

    device = args.device

    # sigma_z = args.sigma_z # net_input+ eSigma (Stochastic temporal ensembling)

    # Step 1. prepare clean & degradation(nosiy) pair
    img_np, noisy_np = load_image_pair(file, task_type, args)

    # if task_type == "denoising":

    #     # image2np  clean_image
    #     img_np  = read_image_np(file)
    #     # noise_np   degrage_image
    #     if "mean" in file:
    #         noisy_np = read_image_np(file.replace("mean", "real"))
    #     else:
    #         noisy_np = img_np + np.random.randn(*img_np.shape) * sigma # 加噪声
    # print("[!] clean image domain : [%.2f, %.2f]" %(img_np.min(), img_np.max()))
    # print("[!] noisy image domain : [%.2f, %.2f]" %(noisy_np.min(), noisy_np.max()))

    # # 真实噪声
    # if args.GT_noise:
    #     args.sigma = (img_np.astype(np.float) - noisy_np.astype(np.float)).std()
    
    # np2torch
    img_torch = cv2_to_torch(img_np, dtype)
    img_torch255 = torch.from_numpy(img_np).type(dtype) # c,w,h，用来在torch.cuda中计算psnr
    noisy_torch = cv2_to_torch(noisy_np, dtype) # batch,c,w,h
    noisy_clip_np = np.clip(noisy_np, 0, 255) # 方便计算psnr 
    noisy_clip_tensor =  torch.from_numpy(noisy_clip_np).type(dtype) # c, w, h 用来在cuda中计算psnr

    
    # net_input，模型输入数据
    if args.dip_type in ["dip_sure", "eSURE_uniform"]:
        net_input = noisy_torch.detach() # 50噪声的图像
    else:
        # print("shape:",*noise_torch.shape)
        net_input = get_noise(noisy_torch.shape).type(dtype).detach() # 随机产生噪声

    # model
    net = models.get_net_indi(args, indi)
    # to cuda
    net.to(device=device)
    net_input.to(device=device)
    net.train()

    # loss  
    # optim / 充分占用显卡利用率，单卡并行

    if args.optim == "adam":
        print("[*] optim_type : Adam")
        optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "adamw":
        print("[*] optim_type : AdamW (wd : 1e-2)")
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "RAdam":

        # optimizer = torch.optim.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))
        optimizer = additional_utils.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))

    # scheduler  更新学习率
    if args.force_steplr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=.9, step_size=300 )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000,3000], gamma=0.5)
    
    # 记录指标
    psnr_noisy_last = 0
    psnr_gt_running = 0

    running_avg = None 
    running_avg_ratio = args.running_avg_ratio

    stat["max_psnr"] = 0 
    stat["max_ssim"] = 0
    stat["NUM_Backtracking"] = 0
    stat["max_avg_psnr"] = 0

    image_name = file.split("/")[-1][:-4]

    args.save_dir = "./dip_denoising/result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    np_save_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(np_save_dir, exist_ok=True)



    

    # uniform_sigma = False
    # # 在net_input 中加uniform 噪声
    # if args.dip_type == "eSURE_uniform":
    #     uniform_sigma = True

    for ep in range(args.epoch):

        # print("第%d个epoch的学习率：%f\n" % (ep, optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()

        net_input_saved = net_input.detach().clone()

        net_input_ep = get_net_input_ep(net_input_saved, args, uniform_sigma=True)
        # set_sigma_z

        # sigma_z = sigma * args.sigma_z
        # eps_y = torch.ones([1], device=device).reshape([-1,1,1,1]) * sigma /255.0
        # vary = (eps_y) ** 2

        # net_input
        # eSURE_uniform
        # eSigma  sigma * sigma_z  50* 0.5


        # if sigma_z > 0 or uniform_sigma:
        #     if uniform_sigma:
        #         eSigma = np.random.uniform(0, sigma_z) / 255.0
        #     else:
        #         eSigma = sigma_z / 255.0
        #     net_input_ep = net_input_saved + torch.randn_like(net_input).type(dtype) * eSigma # net_input 
        # net_input_ep = net_input_ep.requires_grad_() # 自动计算梯度 输入数据
        
        out = net(net_input_ep)

        # 计算divergence
        # divergence_val = divergence(net_input_ep, out)
        # set_sigma

        # total_loss = SURE(out, noisy_torch,divergence_val, vary ) # 斯坦无偏估计计算loss
        total_loss = SURE_loss(net_input_ep, out, noisy_torch, args)

        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(out, img_torch).item()
            diff_loss = total_loss.item() - mse_loss
            # out = out2torch255(out)
            out = torch2torch255(out)
            psnr_noisy = calculate_psnr_tensor(noisy_clip_tensor, out) 
            psnr_gt = calculate_psnr_tensor(img_torch255, out)
        
        if total_loss < 0:
            print('\nLoss is less than 0')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            break
        if(psnr_noisy - psnr_noisy_last < -5)  and (ep > 5):
            print('\nFalling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            stat["NUM_Backtracking"] += 1
            if stat["NUM_Backtracking"] > 10:
                break
        
        
       

        else:
            out_saved = out.detach()
            if running_avg is None:
                running_avg = out_saved
            else:
                running_avg = running_avg * running_avg_ratio + out_saved *(1 -running_avg_ratio) #每个epoch的out进行求avg

            psnr_gt_running =   calculate_psnr_tensor(img_torch255, running_avg)
            

            if (stat["max_psnr"] <= psnr_gt.item()):
                stat["max_step"] = ep
                stat["max_psnr"] = psnr_gt.item()
                stat["max_psnr_avg"] = psnr_gt_running.item()
            
            if(stat["max_avg_psnr"] <= psnr_gt_running.item()):
                stat["max_avg_step"] = ep
                stat["max_avg_psnr"] = psnr_gt_running.item()

            # if (ep>0)and(stat["max_avg_psnr"] - psnr_gt_running.item() > 0.1):
            #     break
            if(ep ==200 or ep==10) and (psnr_gt_running.item() < psnr_gt.item()):
                running_avg = None
            
            print('net:%s Iteration %05d lr: %f total loss: %f / MSE: %f / diff: %f   PSNR_noisy: %f   psnr_gt: %f PSNR_gt_sm: %f' % (
                    indi.id, ep, optimizer.param_groups[0]["lr"],total_loss.item(), mse_loss, diff_loss, psnr_noisy.item(), psnr_gt.item(), psnr_gt_running.item()), end='\r')
            
            last_net = [x.detach().cpu() for x in net.parameters()]

            psnr_noisy_last = psnr_noisy
        total_loss.backward() 
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
    
    stat["final_ep"] = ep
    stat["final_psnr"] = psnr_gt.item()
    stat["final_psnr_avg"] = psnr_gt_running.item()
    stat["consume_time"] = time.time()-t1
    print("%s psnr clean_out : %.2f,  noise_out : %.2f, max %.2f " % (
        image_name, psnr_gt_running, psnr_noisy, stat["max_psnr"]), " " * 100)
    # print(stat)
    torch.cuda.empty_cache()
    return stat
    


from torch.nn.parallel import DistributedDataParallel as DDP



def image_restorazation_DDP(file, args, indi): # args
    
    # dist.init_process_group(backend="nccl", world_size=2, rank=rank)
    # dist.init_process_group(backend='nccl')
    # local_rank = os.getenv('LOCAL_RANK', -1)
    # print("local_rank:",local_rank)

    # if local_rank != -1:
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     dist.init_process_group(backend="nccl")
    
    

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local_rank",local_rank)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print("device",device)

    stat = {}
    task_type = args.task_type
    sigma = args.sigma # noisy_degree
    dtype = args.dtype  

    # device = args.device

    # Step 1. prepare clean & degradation(nosiy) pair
    img_np, noisy_np = load_image_pair(file, task_type, args)


    # # 真实噪声
    # if args.GT_noise:
    #     args.sigma = (img_np.astype(np.float) - noisy_np.astype(np.float)).std()
    
    # np2torch
    img_torch = cv2_to_torch(img_np, dtype)
    img_torch255 = torch.from_numpy(img_np).type(dtype) # c,w,h，用来在torch.cuda中计算psnr
    noisy_torch = cv2_to_torch(noisy_np, dtype) # batch,c,w,h
    noisy_clip_np = np.clip(noisy_np, 0, 255) # 方便计算psnr 
    noisy_clip_tensor =  torch.from_numpy(noisy_clip_np).type(dtype) # c, w, h 用来在cuda中计算psnr

    
    # net_input，模型输入数据
    if args.dip_type in ["dip_sure", "eSURE_uniform"]:
        net_input = noisy_torch.detach() # 50噪声的图像
    else:
        # print("shape:",*noise_torch.shape)
        net_input = get_noise(noisy_torch.shape).type(dtype).detach() # 随机产生噪声

    # model
    net = models.get_net_indi(args, indi)
    # torch.cuda.set_device(device)
    net.to(device=device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    # to cuda
    # net.to(device=device)
    # net = net.to(device)
    net_input.to(device=device)
    net.train()

    # loss  
    # optim / 充分占用显卡利用率，单卡并行

    if args.optim == "adam":
        print("[*] optim_type : Adam")
        optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "adamw":
        print("[*] optim_type : AdamW (wd : 1e-2)")
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "RAdam":

        # optimizer = torch.optim.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))
        optimizer = additional_utils.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))

    # scheduler  更新学习率
    if args.force_steplr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=.9, step_size=300 )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000,3000], gamma=0.5)
    
    # 记录指标
    psnr_noisy_last = 0
    psnr_gt_running = 0

    running_avg = None 
    running_avg_ratio = args.running_avg_ratio

    image_name = file.split("/")[-1][:-4]

    args.save_dir = "./dip_denoising/result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    np_save_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(np_save_dir, exist_ok=True)



    stat["max_psnr"] = 0 
    stat["max_ssim"] = 0
    stat["NUM_Backtracking"] = 0


    for ep in range(args.epoch):

        # print("第%d个epoch的学习率：%f\n" % (ep, optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()

        net_input_saved = net_input.detach().clone()

        net_input_ep = get_net_input_ep(net_input_saved, args, uniform_sigma=True)
       
        
        out = net(net_input_ep)

        total_loss = SURE_loss(net_input_ep, out, noisy_torch, args)

        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(out, img_torch).item()
            diff_loss = total_loss.item() - mse_loss
            # out = out2torch255(out)
            out = torch2torch255(out)
            psnr_noisy = calculate_psnr_tensor(noisy_clip_tensor, out) 
            psnr_gt = calculate_psnr_tensor(img_torch255, out)
        
        if total_loss < 0:
            print('\nLoss is less than 0')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            break
        if(psnr_noisy - psnr_noisy_last < -5)  and (ep > 5):
            print('\nFalling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            stat["NUM_Backtracking"] += 1
            if stat["NUM_Backtracking"] > 10:
                break

        else:
            out_saved = out.detach()
            if running_avg is None:
                running_avg = out_saved
            else:
                running_avg = running_avg * running_avg_ratio + out_saved *(1 -running_avg_ratio) #每个epoch的out进行求avg

            psnr_gt_running =   calculate_psnr_tensor(img_torch255, running_avg)

            if (stat["max_psnr"] <= psnr_gt):
                stat["max_step"] = ep
                stat["max_psnr"] = psnr_gt.item()
            
            if(ep ==200 or ep==10) and (psnr_gt_running.item() < psnr_gt.item()):
                running_avg = None
            
            print('Iteration %05d lr: %f total loss: %f / MSE: %f / diff: %f   PSNR_noisy: %f   psnr_gt: %f PSNR_gt_sm: %f' % (
                    ep, optimizer.param_groups[0]["lr"],total_loss.item(), mse_loss, diff_loss, psnr_noisy.item(), psnr_gt.item(), psnr_gt_running.item()), end='\r')
            
            last_net = [x.detach().cpu() for x in net.parameters()]

            psnr_noisy_last = psnr_noisy
        total_loss.backward() 
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
    
    stat["final_ep"] = ep
    stat["final_psnr"] = psnr_gt
    stat["final_psnr_avg"] = psnr_gt_running
    print("%s psnr clean_out : %.2f,  noise_out : %.2f, max %.2f " % (
        image_name, psnr_gt_running, psnr_noisy, stat["max_psnr"]), " " * 100)
    print(stat)
    torch.cuda.empty_cache()


def net_train(net, optimizer, scheduler, data_params, args ):

    stat = {}
    stat["max_psnr"] = 0 
    stat["max_ssim"] = 0
    stat["NUM_Backtracking"] = 0
    stat["max_psnr_avg"] = 0
    stat["max_ep"] = 0
    stat["final_psnr"] = 0
    stat["final_psnr_avg"] = 0
    stat["final_ep"] = 0
    stat["consume_time"] = 0

    image_name = data_params["image_name"]
    net_input = data_params["net_input"]
    noisy_torch = data_params["noisy_torch"]
    img_torch = data_params["img_torch"]
    noisy_clip_tensor = data_params["noisy_clip_tensor"]
    img_torch255 = data_params["img_torch255"]

    psnr_noisy_last = 0
    psnr_gt_running = 0

    running_avg = None
    running_avg_ratio = args.running_avg_ratio

    net.train()
    
    epoch_start_time =time.time()
    for ep in range(args.epoch):
        
        # print("第%d个epoch的学习率：%f\n" % (ep, optimizer.param_groups[0]['lr']))
        optimizer.zero_grad()

        net_input_saved = net_input.detach().clone()

        net_input_ep = get_net_input_ep(net_input_saved, args, uniform_sigma=True)
        # set_sigma_z
        out = net(net_input_ep)

        total_loss = SURE_loss(net_input_ep, out, noisy_torch, args)

        with torch.no_grad():
            mse_loss = torch.nn.functional.mse_loss(out, img_torch).item()
            diff_loss = total_loss.item() - mse_loss
            # out = out2torch255(out)
            out = torch2torch255(out)
            psnr_noisy = calculate_psnr_tensor(noisy_clip_tensor, out) 
            psnr_gt = calculate_psnr_tensor(img_torch255, out)
        
        if total_loss < 0:
            print('\nLoss is less than 0')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            break
        if(psnr_noisy - psnr_noisy_last < -5)  and (ep > 5):
            print('\nFalling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
            stat["NUM_Backtracking"] += 1
            if stat["NUM_Backtracking"] > 10:
                break

        else:
            out_saved = out.detach()
            if running_avg is None:
                running_avg = out_saved
            else:
                running_avg = running_avg * running_avg_ratio + out_saved *(1 -running_avg_ratio) #每个epoch的out进行求avg

            psnr_gt_running =   calculate_psnr_tensor(img_torch255, running_avg)

            if (stat["max_psnr"] <= psnr_gt):
                stat["max_step"] = ep
                stat["max_psnr"] = psnr_gt.item()
            
            if(ep ==200 or ep==10) and (psnr_gt_running.item() < psnr_gt.item()):
                running_avg = None
            
            print('Iteration %05d lr: %f total loss: %f / MSE: %f / diff: %f   PSNR_noisy: %f   psnr_gt: %f PSNR_gt_sm: %f' % (
                    ep, optimizer.param_groups[0]["lr"],total_loss.item(), mse_loss, diff_loss, psnr_noisy.item(), psnr_gt.item(), psnr_gt_running.item()), end='\r')
            
            last_net = [x.detach().cpu() for x in net.parameters()]

            psnr_noisy_last = psnr_noisy
        total_loss.backward() 
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
    
    epoch_end_time =time.time()
    
    stat["final_ep"] = ep
    stat["final_psnr"] = psnr_gt
    stat["final_psnr_avg"] = psnr_gt_running
    stat["consume_time"] = epoch_end_time-epoch_start_time
    print("%s psnr clean_out : %.2f,  noise_out : %.2f, max %.2f \n" % (
        image_name, psnr_gt_running, psnr_noisy, stat["max_psnr"]), " " * 100)
    
    print(stat)
    # print()
    torch.cuda.empty_cache()
    return stat


def image_restorazation_alternate(file,args,net_indis):

    # stat = {}
    task_type = args.task_type
    sigma = args.sigma # noisy_degree
    dtype = args.dtype  

    device = args.device

    net_len = len(net_indis) #  len of net_indi 

    # Step 1. prepare clean & degradation(nosiy) pair
    img_np, noisy_np = load_image_pair(file, task_type, args)

    # # 真实噪声
    # if args.GT_noise:
    #     args.sigma = (img_np.astype(np.float) - noisy_np.astype(np.float)).std()
    
    # np2torch
    img_torch = cv2_to_torch(img_np, dtype)
    img_torch255 = torch.from_numpy(img_np).type(dtype) # c,w,h，用来在torch.cuda中计算psnr
    noisy_torch = cv2_to_torch(noisy_np, dtype) # batch,c,w,h
    noisy_clip_np = np.clip(noisy_np, 0, 255) # 方便计算psnr 
    noisy_clip_tensor =  torch.from_numpy(noisy_clip_np).type(dtype) # c, w, h 用来在cuda中计算psnr

    
    # net_input，模型输入数据
    if args.dip_type in ["dip_sure", "eSURE_uniform"]:
        net_input = noisy_torch.clone() # 50噪声的图像
    else:
        # print("shape:",*noise_torch.shape)
        net_input = get_noise(noisy_torch.shape).type(dtype).detach() # 随机产生噪声

    # net_indis
    # optim / 充分占用显卡利用率，单卡并行
    optimizers = [None for i in range(net_len)]
    schedulers = [None for i in range(net_len)]
    nets = [None for i in range(net_len)]

    # net optim schedule
    stat_time = time.time()
    for net_index in range(net_len):
        
        # net DistributeDataParallel 封装网络
        nets[net_index] = models.get_net_indi(args, net_indis[net_index]).to(device=device)
        # net DistributeDataParallel 封装网络
        # net = nn.parallel.DistributedDataParallel(net, device_ids=[0])
        
        # optim
        if args.optim == "adam":
            print("[*] optim_type : Adam")
            optimizers[net_index] = torch.optim.Adam(params=nets[net_index].parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optim == "adamw":
            print("[*] optim_type : AdamW (wd : 1e-2)")
            optimizers[net_index] = torch.optim.AdamW(params=nets[net_index].parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optim == "RAdam":
            # optimizer = torch.optim.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))
            optimizers[net_index] = additional_utils.RAdam(params=nets[net_index].parameters(),lr=args.lr,betas=(args.beta1, args.beta2))

    
        # scheduler  更新学习率
        if args.force_steplr:
            schedulers[net_index] = torch.optim.lr_scheduler.StepLR(optimizers[net_index], gamma=.9, step_size=300 )
        else:
            schedulers[net_index] = torch.optim.lr_scheduler.MultiStepLR(optimizers[net_index], milestones=[2000,3000], gamma=0.5)
        
        # 

    image_name = file.split("/")[-1][:-4]

    args.save_dir = "./dip_denoising/result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    np_save_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(np_save_dir, exist_ok=True)

    
    # net_input
    net_input.to(device=device)

    net_input_list = [net_input.clone() for _ in range(net_len)]

    running_avg_list = [None for _ in range(net_len)]
    psnr_gt_list = [None for _ in range(net_len)]
    total_loss_list = [None for _ in range(net_len)]
    psnr_noisy_list = [None for _ in range(net_len)]
    mse_loss_list = [None for _ in range(net_len)]
    diff_loss_list = [None for _ in range(net_len)]

    net_flag = [True for _ in range(net_len)]

    stat = {}
    stat["max_psnr"] = 0 
    stat["max_ssim"] = 0
    stat["NUM_Backtracking"] = 0
    stat["max_psnr_avg"] = 0
    stat["max_ep"] = 0
    stat["final_psnr"] = 0
    stat["final_psnr_avg"] = 0
    stat["final_ep"] = 0
    stat["consume_time"] = 0

    stat_list = [stat for _ in range(net_len)]
    psnr_noisy_last_list = [0 for _ in range(net_len)]
    

    for ep in range(args.epoch):

        flag = net_flag[0]
        for net_index in range(1,net_len):
            flag = flag or net_flag[net_index]
        
        if flag == False:
            break

        for net_index in range(net_len):

            if net_flag[net_index] == False:
                continue
            # print("第%d个epoch的学习率：%f\n" % (ep, optimizer.param_groups[0]['lr']))
            optimizers[net_index].zero_grad()

            net_input_saved = net_input_list[net_index].detach().clone()

            net_input_ep = get_net_input_ep(net_input_saved, args, uniform_sigma=True)
            
            out = nets[net_index](net_input_ep)

            # 计算divergence
            # divergence_val = divergence(net_input_ep, out)
            # set_sigma

            # total_loss = SURE(out, noisy_torch,divergence_val, vary ) # 斯坦无偏估计计算loss
            total_loss_list[net_index] = SURE_loss(net_input_ep, out, noisy_torch, args)

            with torch.no_grad():
                mse_loss = torch.nn.functional.mse_loss(out, img_torch).item()
                diff_loss = total_loss_list[net_index].item() - mse_loss
                # out = out2torch255(out)
                out = torch2torch255(out)
                psnr_noisy = calculate_psnr_tensor(noisy_clip_tensor, out) 
                psnr_gt = calculate_psnr_tensor(img_torch255, out)
            
            if total_loss_list[net_index] < 0:
                print('\nLoss is less than 0')
                for new_param, net_param in zip(last_net, nets[net_index].parameters()):
                    net_param.data.copy_(new_param.cuda())
                net_flag[net_index] = False
                stat_list[net_index]["final_ep"] = ep
                stat_list[net_index]["final_psnr"] = psnr_gt
                stat_list[net_index]["final_psnr_avg"] = psnr_gt_running
                # stat["final_ep"] = ep
                # stat["final_psnr"] = psnr_gt
                # stat["final_psnr_avg"] = psnr_gt_running
                print("%s psnr clean_out : %.2f,  noise_out : %.2f, max %.2f " % (
                    image_name, psnr_gt_running, psnr_noisy, stat_list[net_index]["max_psnr"]), " " * 100)
                print(stat_list)

                # break
            if(psnr_noisy - psnr_noisy_last_list[net_index] < -5)  and (ep > 5):
                print('\nFalling back to previous checkpoint.')
                for new_param, net_param in zip(last_net, nets[net_index].parameters()):
                    net_param.data.copy_(new_param.cuda())
                stat_list[net_index]["NUM_Backtracking"] += 1
                if stat_list[net_index]["NUM_Backtracking"] > 10:
                    net_flag[net_index] = False

            else:
                out_saved = out.detach()
                if running_avg_list[net_index] is None:
                    running_avg_list[net_index] = out_saved
                else:
                    running_avg_list[net_index] = running_avg_list[net_index] * args.running_avg_ratio + out_saved *(1 -args.running_avg_ratio) #每个epoch的out进行求avg

                psnr_gt_running =   calculate_psnr_tensor(img_torch255, running_avg_list[net_index])

                if (stat_list[net_index]["max_psnr"] <= psnr_gt):
                    stat_list[net_index]["max_step"] = ep
                    stat_list[net_index]["max_psnr"] = psnr_gt.item()
                
                if(ep ==200 or ep==10) and (psnr_gt_running.item() < psnr_gt.item()):
                    running_avg_list[net_index] = None
                
                print('Iteration %05d net_indi_id:%s lr: %f total loss: %f / MSE: %f / diff: %f   PSNR_noisy: %f   psnr_gt: %f PSNR_gt_sm: %f' % (
                        ep, net_indis[net_index].id , optimizers[net_index].param_groups[0]["lr"],total_loss_list[net_index].item(), mse_loss, diff_loss, psnr_noisy.item(), psnr_gt.item(), psnr_gt_running.item()), end='\r')
                
                last_net = [x.detach().cpu() for x in nets[net_index].parameters()]

                psnr_noisy_last_list[net_index] = psnr_noisy
            total_loss_list[net_index].backward() 
            optimizers[net_index].step()
            schedulers[net_index].step()
            torch.cuda.empty_cache()
            if ep == args.epoch-1:
                stat_list[net_index]["final_ep"] = ep
                stat_list[net_index]["final_psnr"] = psnr_gt
                stat_list[net_index]["final_psnr_avg"] = psnr_gt_running
                print("%s net_indis:%s psnr clean_out : %.2f,  noise_out : %.2f, max %.2f " % (image_name,net_indis[net_index].id, psnr_gt_running, psnr_noisy, stat_list[net_index]["max_psnr"]), " " * 100)
                
        
    
    for net_index in range(net_len):

       
        print(stat_list[net_index])
    # torch.cuda.empty_cache()
   
    all_consume_time = time.time()-stat_time
    print("all time:", all_consume_time)
    torch.cuda.empty_cache()




class DifferNetThread(threading.Thread):
    def __init__(self, target, args=()):
        super(DifferNetThread, self).__init__()
        self.target = target
        self.args = args
    
    def run(self):
        self.result = self.target(*self.args)
    



def image_restorazation_parallel(file, args, net_indis):
    
    # dist.init_process_group(backend='nccl',rank=0)
    # stat = {}
    task_type = args.task_type
    sigma = args.sigma # noisy_degree
    dtype = args.dtype  

    device = args.device

    net_len = len(net_indis) #  len of net_indi 

    # Step 1. prepare clean & degradation(nosiy) pair
    img_np, noisy_np = load_image_pair(file, task_type, args)

    # # 真实噪声
    # if args.GT_noise:
    #     args.sigma = (img_np.astype(np.float) - noisy_np.astype(np.float)).std()
    
    # np2torch
    img_torch = cv2_to_torch(img_np, dtype)
    img_torch255 = torch.from_numpy(img_np).type(dtype) # c,w,h，用来在torch.cuda中计算psnr
    noisy_torch = cv2_to_torch(noisy_np, dtype) # batch,c,w,h
    noisy_clip_np = np.clip(noisy_np, 0, 255) # 方便计算psnr 
    noisy_clip_tensor =  torch.from_numpy(noisy_clip_np).type(dtype) # c, w, h 用来在cuda中计算psnr

    
    # net_input，模型输入数据
    if args.dip_type in ["dip_sure", "eSURE_uniform"]:
        net_input = noisy_torch.clone() # 50噪声的图像
    else:
        # print("shape:",*noise_torch.shape)
        net_input = get_noise(noisy_torch.shape).type(dtype).detach() # 随机产生噪声

    # net_indis
    # optim / 充分占用显卡利用率，单卡并行
    optimizers = [None for i in range(net_len)]
    schedulers = [None for i in range(net_len)]
    nets = [None for i in range(net_len)]

    # net optim schedule
    stat_time = time.time()
    for net_index in range(net_len):
        
        # net DistributeDataParallel 封装网络
        nets[net_index] = models.get_net_indi(args, net_indis[net_index]).to(device=device)
        # net DistributeDataParallel 封装网络
        # net = nn.parallel.DistributedDataParallel(net, device_ids=[0])
        
        # optim
        if args.optim == "adam":
            print("[*] optim_type : Adam")
            optimizers[net_index] = torch.optim.Adam(params=nets[net_index].parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optim == "adamw":
            print("[*] optim_type : AdamW (wd : 1e-2)")
            optimizers[net_index] = torch.optim.AdamW(params=nets[net_index].parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optim == "RAdam":
            # optimizer = torch.optim.RAdam(params=net.parameters(),lr=args.lr,betas=(args.beta1, args.beta2))
            optimizers[net_index] = additional_utils.RAdam(params=nets[net_index].parameters(),lr=args.lr,betas=(args.beta1, args.beta2))

    
        # scheduler  更新学习率
        if args.force_steplr:
            schedulers[net_index] = torch.optim.lr_scheduler.StepLR(optimizers[net_index], gamma=.9, step_size=300 )
        else:
            schedulers[net_index] = torch.optim.lr_scheduler.MultiStepLR(optimizers[net_index], milestones=[2000,3000], gamma=0.5)
        
        # 

    image_name = file.split("/")[-1][:-4]

    args.save_dir = "./dip_denoising/result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    np_save_dir = os.path.join(args.save_dir, image_name)
    os.makedirs(np_save_dir, exist_ok=True)

    # net_input
    net_input.to(device=device)
  

    # data_params ={}
    # data_params["image_name"] = image_name
    # data_params["net_input"] = net_input
    # data_params["noisy_torch"] = noisy_torch
    # data_params["img_torch"] = img_torch
    # data_params["noisy_clip_tensor"] = noisy_clip_tensor
    # data_params["img_torch255"] = img_torch255
    net_input_list = [net_input.clone() for _ in range(net_len)]
    noisy_torch_list = [noisy_torch.clone() for _ in range(net_len)]
    img_torch_list = [noisy_torch.clone() for _ in range(net_len)]
    noisy_clip_tensor_list = [noisy_clip_tensor.clone() for _ in range(net_len)]
    img_torch255_list = [img_torch255.clone() for _ in range(net_len)]
    
    data_params_list = [None for _ in range(net_len)]
    for net_index in range(net_len):
        data_params_list[net_index] = {}
        data_params_list[net_index]["image_name"] = image_name
        data_params_list[net_index]["net_input"] = net_input_list[net_index]
        data_params_list[net_index]["noisy_torch"] = noisy_torch_list[net_index]
        data_params_list[net_index]["img_torch"] = img_torch_list[net_index]
        data_params_list[net_index]["noisy_clip_tensor"] = noisy_clip_tensor_list[net_index]
        data_params_list[net_index]["img_torch255"] = img_torch255_list[net_index]
    
    args_list =[copy.deepcopy(args) for _ in range(net_len)]
    net_stats = [None for _ in range(net_len)]
    threads= [None for _ in range(net_len)]
    # processes = [None for _ in range(net_len)]


    # 创建线程
    for net_index in range(net_len):
        threads[net_index] = DifferNetThread(target=net_train, args=(nets[net_index], optimizers[net_index], schedulers[net_index], data_params_list[net_index],args))
        # threads[net_index] = threading.Thread(target=net_train, args=(nets[net_index], optimizers[net_index], schedulers[net_index], data_params_list[net_index],args))
        
        # processes[net_index] = Process(target=net_train, args=(nets[net_index], optimizers[net_index], schedulers[net_index], data_params_list[net_index],net_stats[net_index]))
    # 启动线程
    for net_index in range(net_len):
        threads[net_index].start()
        
        # processes[net_index].start()

    # 等待线程结束
    for net_index in range(net_len):
        threads[net_index].join()
        net_stats[net_index] = threads[net_index].result
        # processes[net_index].join()

    print(net_stats)
    all_consume_time = time.time()-stat_time
    print("all time:", all_consume_time)
    torch.cuda.empty_cache()

    # dist.destroy_process_group()



def denoise(denoise_params, indis):
    # 1. 加载图像
    # 2. 加载模型，进行降噪
    print("[*] reproduce mode On")
    # torch.manual_seed(0) # GPU 随机种子
    # np.random.seed(0)
    args = argparse.ArgumentParser()

    # cudnn 设置
    if torch.cuda.is_available():
        # 指定显卡
        device = torch.device("cuda:0") # 之后可设置并行
        args.device = device
        # 寻找适合卷积实现的方法，网络加速
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        args.dtype = torch.cuda.FloatTensor # 数据类型： FloatTensor
    else:
        device = torch.device("cpu")
        args.device = device
        args.dtype = torch.FloatTensor



    
    args.dip_type = "eSURE_uniform"
    args.net_type = "unet_indi_scale" # s2s, unet_indi, s2s_test, s2s_test_MIMO,s2s_test_first,unet_indi_scale, unet_indi_connect
    args.device = torch.device("cuda:0")
    args.gray = False
    args.sigma = 50
    args.dtype = torch.cuda.FloatTensor
    args.task_type = "denoising"
    # args.desc = "_unet_indi_sigma50"
    args.desc = "_"+args.net_type+"_sigam"+str(args.sigma)
    args.exp_tag = "single_image"

    # net_set
    args.optim = "RAdam"
    args.lr = 0.1
    args.force_steplr = False
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.sigma_z = 0.5
    args.epoch = 0

    # # dip
    args.running_avg_ratio = 0.99

    # data
    args.eval_data = "single_image" # single_image


    # 结果保存文件
    save_dir = "./dip/result/%s/%s/%s" % (args.task_type, args.exp_tag, args.dip_type + args.desc)
    os.makedirs(save_dir, exist_ok =True)

    # epoch
    if args.task_type == "denoising":
        args.epoch = 3000 if args.epoch ==0 else args.epoch
        # 保存点
        save_point = [1,10,100,500, 1000, 2000, 3000, 4000] 
    
    # # 记录设置信息
    # with open(os.path.join(save_dir, 'args.json'), 'w') as f:
    #     json.dump()


    # data_file_list
    args.dataset_dir = "./dip/testset/%s/" % args.eval_data
    
    file_names = os.listdir(args.dataset_dir)
    file_list =[]
    for file_name in file_names:
        file_list.append(args.dataset_dir+file_name)
    
    # print(file_list)
    file_list = sorted(file_list)
    net_stat_list =[]
    # indi = None
    # indi_len = len(indis)
    # processes = [None for _ in range(indi_len)]
    # for indi_index in range(indi_len):
    #     processes[indi_index] = Process(target=image_restorazation, args=(file_list[indi_index], indis[indi_index]))
    
    # for indi_index in range(indi_len):
    #     processes[indi_index].start()
    
    # for indi_index in range(indi_len):
    #     processes[indi_index].join()
    len_net = len(indis)
    for net_index in range(len_net):
        stat_list =[]
        t1 = time.time()
        for file in file_list:
            
            stat = image_restorazation(file,args,indis[net_index])
            print(stat)

            # stat_list = image_restorazation(file, args, indis[net_index])
            # stat_list = image_restorazation_DDP(file, args,indis)

            # processes = []
            # dist.init_process_group(backend='nccl')
            # mp.set_start_method('spawn')

            # len_indi = len(indis)
            # pool = mp.Pool(processes=2)

            # for net_index in range(len(indis)):
            #     pool.apply_async(image_restorazation_DDP,args=(file,args,indis[net_index], net_index))
            
            # pool.close()
            # pool.join()
            
            # for net_index in range(len(indis)):
            #     # stat_list = image_restorazation(file, args, indis[net_index])
                
            #     process = Process(target=image_restorazation_DDP, args=(file,args,indis[net_index], len(indis)))
            #     process.start()
            #     processes.append(process)
            
            # for process in processes:
            #     process.join()

            # dist.destroy_process_group()
            # stat_list = image_restorazation_alternate(file,args,indis)
            # stat_list = image_restorazation_parallel(file,args,indis)
            # indi_len = len(indis)
            # # image_restorazation
            # processes = [None for _ in range(indi_len)]
            # for indi_index in range(indi_len):
            #     processes[indi_index] = Process(target=image_restorazation, args=(file, args, indis[indi_index]))
            
            # for indi_index in range(indi_len):
            #     processes[indi_index].start()
            
            # for indi_index in range(indi_len):
            #     processes[indi_index].join()
            # stat = image_restorazation_parallel(file, args, indis)
            stat_list.append(stat)
        t2 = time.time()-t1
        print("consum_time", t2)
        net_stat_list.append(stat_list)

    # all_data_set_data_csv
    # for net_index in range(len(net_stat_list)):
    #     data = pd.DataFrame(net_stat_list[net_index], index= [i.split("/")[-1] for i in file_list])
    #     print(data)
    #     os.makedirs("./dip/csv/%s/%s/"  % (args.task_type, args.exp_tag), exist_ok=True)
    #     data.to_csv("./dip/csv/%s/%s/%s.csv" % ( args.task_type, args.exp_tag ,args.dip_type+args.desc+indis[net_index].id))
    #     # net_data_index[net_index] = net_stat_list[net_index]["net_indi"]
    
    # net_data_index = []
    net_data = pd.DataFrame(net_stat_list)
    os.makedirs("./dip/csv/%s/%s/"  % (args.task_type, args.exp_tag), exist_ok=True)
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    net_data.to_csv("./dip/csv/%s/%s/%s.csv" % ( args.task_type, args.exp_tag ,args.dip_type+args.desc+"_all_indi_"+time_str))
    print(net_data)
    # data = pd.DataFrame(net_stat_list)
    
    # os.makedirs("./dip/csv/%s/%s/"  % (args.task_type, args.exp_tag), exist_ok=True)
    # data.to_csv("./dip/csv/%s/%s/%s.csv" % ( args.task_type, args.exp_tag ,args.dip_type+args.desc+indis[net_index].id))



if __name__=="__main__":
    from genetic.individual import Individual
    from setting.config import Config
   
    dict1 = {
            "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'convolution_d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                ]
        }

    dict2 = {
            "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'convolution_d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
        }

    dict3 = {
            "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
        }

    dict_maxpool = {
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
    }

    dict_area_d_bilinear_u = {
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'area_d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'area_d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'area_d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'area_d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'area_d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'bilinear_u'},
                ]
    }

    dict_conv3x3 = {
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
    }
    dict_conv5x5 ={
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_5x5_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_5x5_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
    }
    dict_conv7x7 ={
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_7x7_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_7x7_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
    }


    dict_easy_conv3x3 = {
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'nearest_u'},
                ]
    }
    dict_dilconv3x3 = {
        "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_3x3', 'downsample_type': 'maxpool2d'}, 
            ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_3x3', 'upsample_type': 'nearest_u'},
                ]
    }
    dict_dilconv5x5 = {
        "level_amount":5,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'}, 
            {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'}, 
            {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'}, 
        ],

        # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            ]
    }

    dict_level_3_conv3x3 = {
        "level_amount":3,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'maxpool2d'},  
        ],

        # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},
           
            ]
    }

    dict_best_indi = {
        'level_amount': 5,
        'encoder_units':[
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3', 'downsample_type': 'area_d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'bilinear_d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'dil_conv_5x5', 'downsample_type': 'bilinear_d'},
            {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'nearest_d'},
            {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_7x7', 'downsample_type': 'maxpool2d'},],
        'decoder_units':[
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5', 'upsample_type': 'bilinear_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_5x5', 'upsample_type': 'bilinear_u'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3', 'upsample_type': 'sub_pixel_u'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'bilinear_u'},
            {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'dil_conv_5x5', 'upsample_type': 'nearest_u'},],
        }
    

    dict_indi_scale = {
        
        "level_amount":5,
        "scale":[0,0,0,0,0],
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
        ],

            # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
            ]
    }
    inid_scale = Individual(Config.population_params, "indi_scale")
    inid_scale.initialize_scale_by_designed(dict_indi_scale)
    level_mount = 5
    scale_list = []
    dict_list =[]
    dict_name_list = []
    # 十进制转二进制
    print("pow",pow(2,level_mount-1))
    for i in range(pow(2,level_mount-1 )):
  
        scale = [0 for _ in range(level_mount)]
        # scale[0] = 0
        val = i
        j = level_mount-1
        while(val>0):
            scale[j] = val % 2
            j -= 1
            val = val//2
        
        print(scale)
        scale_list.append(scale)
    for scale in scale_list:
        scale_val = 0
        for i in range(level_mount):
            scale_val += pow(2, level_mount-1-i)*scale[i]
        dict_name = "scale_" + str(scale_val)
        dict = {
            "level_amount":5,
            "encoder_units":[ 
                {'encoder_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 1, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'},
                {'encoder_unit_id': 2, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 3, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
                {'encoder_unit_id': 4, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU', 'downsample_type': 'maxpool2d'}, 
            ],

                # "middle_units": [{'middle_unit_id': 0, 'block_amount': 1, 'features': 48, 'conv_type': 'conv_3x3_leakyReLU'}],

            "decoder_units": [
                {'decoder_unit_id': 0, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 1, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 2, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 3, 'block_amount': 2, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                {'decoder_unit_id': 4, 'block_amount': 1, 'features': 96, 'conv_type': 'conv_3x3_leakyReLU', 'upsample_type': 'nearest_u'},
                ]
        }
        dict['scale'] = scale
        # print(dict['scale'])
        # print(dict_name)
        dict_list.append(dict)
        dict_name_list.append(dict_name)
    
    indis = []
    for index in range(len(dict_list)):
        indi = Individual(Config.population_params,dict_name_list[index])
        # indi.initialize_by_designed(dict_list[index])
        indi.initialize_scale_by_designed(dict_list[index])
        indis.append(indi)
        # print(dict_list[index])

            
        
    

    

    conv_op = ["conv_5x5_leakyReLU", "conv_7x7_leakyReLU"]
    downsample_op = ["maxpool2d", "convolution_d", "bilinear_d", "nearest_d", "area_d"]
    upsample_op = ["deconv", "sub_pixel_u", "bilinear_u", "nearest_u","area_u"]

    

    # dict_list =[]
    # dict_name_list = []
    # len_conv_op = len(conv_op)
    # len_downsample_op = len(downsample_op)
    # len_upsample_op = len(upsample_op)
    
    # for conv_op_index in range(len_conv_op):
    #     for downsample_op_index in range(len_downsample_op):
    #         for upsample_op_index in range(len_upsample_op):
    #             dict = {}
    #             dict["level_amount"] = 5
    #             encoder_units=[None for _ in range(dict["level_amount"])]
    #             decoder_units= [None for _ in range(dict["level_amount"])]
    #             for level in range(dict["level_amount"]):
    #                 encoder_unit = {}
    #                 encoder_unit["encoder_unit_id"] =level
    #                 encoder_unit["block_amount"] = 1
    #                 encoder_unit["features"] = 48
    #                 encoder_unit["conv_type"] = conv_op[conv_op_index]
    #                 encoder_unit["downsample_type"] = downsample_op[downsample_op_index]
    #                 encoder_units[level] = encoder_unit

    #                 decoder_unit={}
    #                 decoder_unit["decoder_unit_id"] = level
    #                 if level == dict['level_amount']-1:
    #                     decoder_unit["block_amount"] = 2
    #                 else:
    #                     decoder_unit['block_amount'] = 1
    #                 decoder_unit["features"] = 96
    #                 decoder_unit["conv_type"] = conv_op[conv_op_index]
    #                 decoder_unit["upsample_type"] = upsample_op[upsample_op_index]
    #                 decoder_units[level] = decoder_unit

    #                 encoder_units[level]["encoder_unit_id"] = level
    #                 encoder_units[level]["encoder_unit_id"] = level
    #             dict["encoder_units"] = encoder_units
    #             dict["decoder_units"] = decoder_units
    #             dict_name = conv_op[conv_op_index]+"_"+downsample_op[downsample_op_index]+"_"+upsample_op[upsample_op_index]
    #             dict_name_list.append(dict_name)
    #             dict_list.append(dict)

    # indis = []
    # for index in range(len(dict_list)):
    #     indi = Individual(Config.population_params,dict_name_list[index])
    #     indi.initialize_by_designed(dict_list[index])
    #     indis.append(indi)
        # print(dict_list[index])





  
    indi1 = Individual(Config.population_params, 0)
    indi2 = Individual(Config.population_params, 1)
    indi3 = Individual(Config.population_params, 2)
    indi4 = Individual(Config.population_params, 3)
    indi5 = Individual(Config.population_params, 4)
    indi6 = Individual(Config.population_params, 5)

    indi1.initialize_by_designed(dict2)
    indi2.initialize_by_designed(dict2)
    indi3.initialize_by_designed(dict2)
    indi4.initialize_by_designed(dict2)
    indi5.initialize_by_designed(dict2)
    indi6.initialize_by_designed(dict2)
    indi_maxpool = Individual(Config.population_params, "indi_maxpool")
    indi_maxpool.initialize_by_designed(dict_maxpool)

    # conv5x5, 7x7 padding ,ep
    indi_conv5x5 = Individual(Config.population_params,"net_conv5x5")
    indi_conv5x5.initialize_by_designed(dict_conv5x5)
    indi_conv7x7 = Individual(Config.population_params, "net_conv7x7")
    indi_conv7x7.initialize_by_designed(dict_conv7x7)

    indi_easy_conv3x3 = Individual(Config.population_params,"net_easy_conv3x3")
    indi_easy_conv3x3.initialize_by_designed(dict_easy_conv3x3)
    indi_dil_conv3x3 = Individual(Config.population_params,"net_dil_conv3x3")
    indi_dil_conv3x3.initialize_by_designed(dict_dilconv3x3)
    indi_dil_conv5x5 = Individual(Config.population_params,"net_dil_conv5x5")
    indi_dil_conv5x5.initialize_by_designed(dict_dilconv5x5)
    

    indi_area_d_bilinear_u = Individual(Config.population_params, "net_area_d_bilinear_u")
    indi_area_d_bilinear_u.initialize_by_designed(dict_area_d_bilinear_u)

    indi_level_3_conv_3x3 = Individual(Config.population_params, "net_level_3_conv3x3")
    indi_level_3_conv_3x3.initialize_by_designed(dict_level_3_conv3x3)

    indi_best = Individual(Config.population_params, 'net_indi_best')
    indi_best.initialize_by_designed(dict_best_indi)

    # indis = [inid_scale]

    # indis = [indi_area_d_bilinear_u,indi_area_d_bilinear_u]
    # run
    denoise(None, indis)


    # os.system("shutdown -s -t 60 ")
    # os.system("shutdown -h now")
    # for net_index in range(len(indis)):
    #     denoise(None, indis[net_index])



# denoise("1")
    


