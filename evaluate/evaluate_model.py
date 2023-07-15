# import argparse
# 
import os
import sys
# dir=os.path.abspath('..')
# print(dir)
# sys.path.append(dir)
# print(os.getcwd())
import math
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from tensorboardX import SummaryWriter
# 将路径添加到系统变量中
dir  = os.getcwd()
sys.path.append(dir)
# print(os.getcwd())
# print(sys.path)

from evaluate.DnCNN.dataset import prepare_data, Dataset

from evaluate.DnCNN.model import Unet


from model.unet import DnUnet3,DnCNN,DnUnet31


# from model.unet import UNet
# from models import DnCNN
# from model.DHDN_gray import Net
# from model.unet import DnUnet, DnUnet3, DnUnet4

import time
import logging

import  matplotlib.pyplot as plt

from setting.config import Config

from genetic.individual import Individual





# 权重初始化
# 有conv,linear进行kaiming_normal 初始化
# kaiming_normal： 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    # PSNR (Peak Signal-to-Noise Ratio) 峰值信噪比
    Img = img.data.cpu().numpy().astype(np.float32) # 模型输出图像数据
    Iclean = imclean.data.cpu().numpy().astype(np.float32) # 清晰图像数据
    PSNR = 0
    # PSNR
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def output_psnr_mse(img_orig, img_out):
    img_orig = img_orig.detach().cpu().numpy()
    img_out = img_out.detach().cpu().numpy()
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def data_partition(dataset_train,partition_degree,batch_size):

    print('# of training samples: %d\n' % int(len(dataset_train)))
    # 划分数据集
    train_size = len(dataset_train)//partition_degree
    print("训练样本个数%s" % str(train_size))
    
    indices = list(range(train_size)) # 每个训练集个索引
    shuffle = True
    random_seed = 12
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    # return train_sampler

    loader_train = DataLoader(dataset_train, batch_size=batch_size,num_workers=4,sampler=train_sampler)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
    print("training samples: %d\n" % int(len(loader_train)))
    return loader_train

def net_train(indi, params):
    """
    params: 
        unet_arichtect: gene解码后的unet结构字典
        mode: 降噪模式
        noiseL: 噪声等级
        val_noiseL: 噪声等级
    """
    # 日志：


    epochs = params["epoch"]
    milestone = params["milestone"]
    lr = params["lr"]
    batch_size  = params["batch_size"]
    mode = params["noise_mode"]
    noiseL = params["noiseL"]
    val_noiseL = params["val_noiseL"]
    partition_degree = params["partition_degree"]
    outf = params["outf"]
    logging_file = params["log_path"]
    
    file = open(logging_file, encoding='utf-8',mode='a')
    # log_format = '%(asctime)s %(message)s'
    logging.basicConfig(format='%(asctime)s %(message)s',
                    stream=file, 
                    level=logging.INFO)
    # load dataset
    start_time = time.time()
    # print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    # loader_train =DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size,shuffle=True)
    loader_train = data_partition(dataset_train,partition_degree=partition_degree, batch_size=batch_size)
    # print('# of training samples: %d\n' % int(len(dataset_train)))

    # Build model

    # net = DnUnet4()
    # net = Unet(indi,in_channels= 1, features=64)
    net = DnUnet31()
    net.apply(weights_init_kaiming)
    # net = DnUnet3()
    # net.apply(weights_init_kaiming)
    # unet_model = Decode2Unet(unet_arichtect)
    # DnCNN
    # net = DnCNN(channels=1)
    # net.apply(weights_init_kaiming)
    # weights_init_kaiming： 权重初始化
    # nn.module.apply(fn)：递归的将fn函数应用于模块的每个子模块，即网络每层进行权重初始化
    # net.apply(weights_init_kaiming)
    # DHDN
    # net = Net()
    # net.apply(weights_init_kaiming) # DHDN 无
    # NAFNet
    # from model.NAFNet_models.NAFNet_arch import NAFNet

    # enc_blks = [3, 3]
    # middle_blk_num = 3
    # dec_blks = [3, 3]
    
    # net = NAFNet(img_channel=1, width=32, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    # 均方误差
    criterion = nn.MSELoss(reduction='sum') # reduction='sum' 所有项求和求平均

    # criterion = nn.L1Loss() # DHDN的loss
    # Move to GPU
    device_ids = [0]

    # 多卡gpu 运行
    model = nn.DataParallel(net,device_ids=device_ids).cuda()
    criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 0
    noiseL_B = [0, 55]
    best_psnr = 0.0
    current_lr = lr
    for epoch in range(epochs):

        if epoch < milestone:
            current_lr = lr
        else:
            current_lr = lr/10.
        # if ((epoch+1) %10) == 0:
        #     current_lr = current_lr/2.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # print('learning rate %f' % current_lr)
        # logging.info('learning rate %f' % current_lr)

        # psnr_train_list=[]
        # psnr_val_list=[]
        # loss_list=[]

        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            # noise 噪声
            if mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=noiseL/255.)
            if mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)


            # 加了噪声的图片
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            # 模型输出结果
            out_train = model(imgn_train)

            # # 标准 方法loss
            # loss  = criterion(out_train, img_train) 
            # loss.backward()
            # optimizer.step()
            # psnr_train = output_psnr_mse(img_train,out_train)
            # # out_train = torch.clamp(out_train, 0., 1.)
            # # psnr_train = batch_PSNR(out_train,img_train, 1.)

            # loss DnCNN残差方法
            # loss = criterion(out_train, img_train)/(imgn_train.size()[0]*2)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()

            # results
            model.eval()
            # torch.clamp（x,min,max），将所有元素压缩到min,max范围内 ，将模型训练后输出的值与实际值之间的差距缩放到[0,1]

            # out_train = torch.clamp(out_train, 0., 1.) # 正常out_train
            out_train =torch.clamp(imgn_train-model(imgn_train),0.,1.) # 数据集残差处理
            # out_train =torch.clamp(imgn_train-out_train,0.,1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            # loss_list.append(loss.item())
            # psnr_train_list.append(psnr_train)
            
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # logging.info('epoch: %d lr: %e loss: %.4f psrnr_train: %.4f',epoch, lr,loss.item(), psnr_train)
            # logging.info("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if step % 10 == 0:
            #     # Log the scale values
                # writer.add_scalar('loss', loss.item(), step)
                # writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        # logging.info('epoch: %d lr: %e loss: %.4f psrnr_train: %.4f',epoch, current_lr, loss.item(), psnr_train)
        print('epoch: %d lr: %e loss: %.4f psrnr_train: %.4f',epoch, current_lr, loss.item(), psnr_train)

        # validate
        
        model.eval()
        with torch.no_grad():
            psnr_val =0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0 )
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0,std=val_noiseL/255.)
                imgn_val = img_val + noise
                # 残差方法的eval
                img_val, imgn_val = Variable(img_val.cuda()),Variable(imgn_val.cuda(),volatile=True)
                out_val = torch.clamp(imgn_val - model(imgn_val),0., 1.) # 残差方法
                # out_val = torch.clamp(model(imgn_val), 0., 1.) # 正常方法
                psnr_val += batch_PSNR(out_val, img_val, 1.)

                # out_val  = torch.clamp(model(imgn_val), 0., 1.)
                # psnr_val += batch_PSNR(out_val, img_val, 1.)
                # # 标准方法的
                # out_val =  model(imgn_val)
                # psnr_val += output_psnr_mse(img_val,out_val)

            psnr_val /= len(dataset_val)
            # psnr_val_list.append(psnr_val)
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                torch.save(model.state_dict(), os.path.join(outf,'DnUnet4_5_22.pth'))
            # print("[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            print('epoch: %d lr: %e loss: %.4f psnr_train: %.4f, psnr_val: %.4f' % (epoch, current_lr, loss.item(), psnr_train,psnr_val))
            logging.info('epoch: %d lr: %e loss: %.4f psnr_train: %.4f psnr_val: %.4f ' % (epoch, current_lr, loss.item(), psnr_train, psnr_val))
        # logging.info('epoch: %d psnr_val: %.4f' %(epoch, psnr_val))
        
    torch.save(model.state_dict(), os.path.join(outf,'DnUnet4_final.pth'))
    end_time = time.time()
    consume_time = (end_time - start_time)/60
    # consum_time: 10126.0855 ,2.812h
    # print("start_channels: 64,训练耗时：",consume_time)
    logging.info("best_psnr: %.4f consume_time: %.4f" %(best_psnr, consume_time))
    print("best_psnr: %.4f consume_time: %.4f" % (best_psnr, consume_time))
    # plt.plot(psnr_train_list)
    # plt.savefig("psnr_train.png")

    # plt.plot(psnr_val_list)
    # plt.savefig("psnr_val.png")

    # plt.cla()
    # plt.plot(loss_list)
    # plt.savefig("loss.png")


    return best_psnr  

if __name__=='__main__':
    params = Config.train_params
    indi = Individual(Config.population_params,0)
    dict = {
        "level_amount":2,
        "middle_unit_amount":1,
        "encoder_units":[ 
            {'encoder_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'},
            {'encoder_unit_id': 1, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'downsample_type': 'maxpool2d'}, ],

        "middle_units": [{'middle_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3'}],

        "decoder_units": [
            {'decoder_unit_id': 0, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},
            {'decoder_unit_id': 1, 'block_amount': 3, 'features': 64, 'conv_type': 'conv_3x3', 'upsample_type': 'deconv'},]
    }
    indi.initialize_by_designed(dict)
    net_train(indi,params)

    

    

    