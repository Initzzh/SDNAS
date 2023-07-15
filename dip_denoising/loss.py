import torch
import torch.nn as nn
import additional_utils
from utils.blur_utils import *  # blur functions

class DIPloss(nn.Module):
    def __init__(self, net, net_input,  args):
        super(DIPloss, self).__init__()
        self.net = net
        self.dip_type = args.dip_type
        self.reg_noise_std = args.reg_noise_std
        self.task_type = args.task_type
        self.dtype = args.dtype
        self.mse = torch.nn.MSELoss(reduction="sum")
        self.net_input_saved = net_input.detach().clone() 
        self.cnt = 0
        self.epsilon_decay = False
        self.reduction = "mean"

    def set_sigma(self, sigma):
        self.sigma = sigma  # 25
        self.sigma_y = sigma # 25
        self.sigma_z = sigma * self.arg_sigma_z  # 25*0.5
        self.eps_y = torch.ones([1], device="cuda").reshape([-1, 1, 1, 1]) * self.sigma_y / 255.0
        self.eps_tf = self.eps_y * self.arg_epsilon
        self.eps_tf_init = self.eps_tf.clone()
        self.vary = (self.eps_y) ** 2

    def DIP(self, net_input, noisy_torch):
        if self.reg_noise_std > 0:
            net_input = self.net_input_saved + (torch.rand_like(self.net_input_saved).normal_() * self.reg_noise_std) # net_input + 一点偏差
        out = self.inference(net_input)
        total_loss = torch.mean((out - noisy_torch) ** 2)
        return total_loss, out



    def DIP_SURE_MIMO(self, net_input, noisy_torch):
        if self.sigma_z > 0 or self.uniform_sigma:
            if self.uniform_sigma:
                self.eSigma = np.random.uniform(0, self.sigma_z) / 255.0
            else:
                self.eSigma = self.sigma_z  / 255.0
            net_input = self.net_input_saved + torch.randn_like(net_input).type(self.dtype) * self.eSigma # 
        net_input = net_input.requires_grad_() # 自动进行梯度更新。

        
        out = self.inference(net_input)#.contiguous(memory_format=torch.channels_last)
        noisy_torch2 = F.interpolate(noisy_torch, scale_factor=0.5, mode='bilinear' )
        noisy_torch4 = F.interpolate(noisy_torch,scale_factor=0.25, mode='bilinear' )
        # lable_img
        # divergence1 = self.divergence(, out)

        l1 =  self.SURE(out[0], noisy_torch4,divergence, self.vary)
        l2 =  self.SURE(out[1], noisy_torch2)
        l3 =  self.SURE(out[2], noisy_torch )
        loss_content = l1+l2+l3

        noisy_fft1 = torch.rfft(noisy_torch4, signal_ndim=2, normalized=False, onesided=False)
        out_fft1 = torch.rfft(out[0], signal_ndim=2, normalized=False, onesided=False)
        noisy_fft2 = torch.rfft(noisy_torch2, signal_ndim=2, normalized=False, onesided=False)
        out_fft2 = torch.rfft(out[1], signal_ndim=2, normalized=False, onesided=False)
        noisy_fft3 = torch.rfft(noisy_torch, signal_ndim=2, normalized=False, onesided=False)
        out_fft3 = torch.rfft(out[2], signal_ndim=2, normalized=False, onesided=False)

        f1 =  self.SURE(out_fft1, noisy_fft1)
        f2 =  self.SURE(out_fft2, noisy_fft2)
        f3 =  self.SURE(out_fft3, noisy_fft3)
        loss_fft = f1+f2+f3
        total_loss = loss_content + 0.1 * loss_fft

        # total_loss = self.SURE(out, noisy_torch, divergence, self.vary)
        
        return total_loss, out[2]

    def SURE(self, output, target, divergence, sigma):
        batch, c, h, w = output.shape
        divergence = divergence * sigma
        mse = (output - target) ** 2
        esure = mse + 2 * divergence - sigma
        esure = torch.sum(esure)
        esure = esure if self.reduction == "sum" else esure / (h * w * c)
        return esure

    def DIP_SURE(self, net_input, noisy_torch):
        if self.sigma_z > 0 or self.uniform_sigma:
            if self.uniform_sigma:
                self.eSigma = np.random.uniform(0, self.sigma_z) / 255.0
            else:
                self.eSigma = self.sigma_z  / 255.0
            net_input = self.net_input_saved + torch.randn_like(net_input).type(self.dtype) * self.eSigma
        net_input = net_input.requires_grad_()

        out = self.inference(net_input)#.contiguous(memory_format=torch.channels_last)
        divergence = self.divergence(net_input, out)
        total_loss = self.SURE(out, noisy_torch, divergence, self.vary)
        return total_loss, out

    def divergence_ty(self, net_input, out):
        if self.epsilon_decay:
            self.eps_tf = self.eps_tf_init * (0.9 ** (self.cnt // 200))
        b_prime = torch.randn_like(net_input).type(self.dtype)
        out_ptb = self.inference(net_input + b_prime * self.eps_tf)
        divergence = (b_prime * (out_ptb - out)) / self.eps_tf
        return divergence

    def divergence_new(self, net_input, out):
        b_prime = torch.randn_like(net_input).type(self.dtype)
        nh_y = torch.sum(b_prime * out, dim=[1, 2, 3])
        vector = torch.ones(1).to(out)
        divergence = b_prime * \
                     torch.autograd.grad(nh_y, net_input, grad_outputs=vector, retain_graph=True, create_graph=True)[0]
        return divergence

    def inference(self, x):
        return self.net(x)

    def forward(self, input, target):
        self.cnt += 1
        return self.loss(input, target)

class Denoising_loss(DIPloss):
    def __init__(self, net, net_input,  args):
        super(Denoising_loss, self).__init__(net, net_input, args)
        # parameter related to SURE.
        self.arg_sigma_z = args.sigma_z
        self.arg_epsilon = args.epsilon
        self.set_sigma(args.sigma)
        self.cnt = 0
        self.epsilon_decay = False
        self.reduction = "mean"

        print("[*] loss type : %s" % args.dip_type)
        print("[*] sigma : %.2f" % self.sigma)
        print("[*] sigma_z : %.2f" % self.sigma_z)

        self.divergence = self.divergence_ty
        self.uniform_sigma = False
        self.clip_divergence = False
        if self.dip_type == "dip":
            self.loss = self.DIP
        elif self.dip_type == "dip_sure":
            self.sigma_z = 0
            self.loss = self.DIP_SURE
        elif self.dip_type == "dip_sure_new":
            self.sigma_z = 0
            self.loss = self.DIP_SURE
            self.divergence = self.divergence_new
        elif self.dip_type == "eSURE":
            self.loss = self.DIP_SURE
        elif self.dip_type == "eSURE_alpha":
            self.epsilon_decay = True
            self.loss = self.DIP_SURE
        elif self.dip_type == "eSURE_new":
            self.divergence = self.divergence_new
            self.loss = self.DIP_SURE
        elif self.dip_type == "eSURE_uniform":
            self.uniform_sigma = True
            self.divergence = self.divergence_new
            self.loss = self.DIP_SURE
        elif self.dip_type == "dip_sure_MIMO":
            self.sigma_z = 0
            self.loss = self.DIP_SURE
            self.divergence = self.divergence_new
        else:
            print("[!] Not defined loss function.")
            raise NotImplementedError
        

class Poisson_loss(DIPloss):
    def __init__(self, net, net_input,  args):
        super(Poisson_loss, self).__init__(net, net_input, args)
        self.net = net
        self.dip_type = args.dip_type
        self.reg_noise_std = args.reg_noise_std
        self.task_type = args.task_type
        self.dtype = args.dtype
        self.mse = torch.nn.MSELoss(reduction="sum")
        self.net_input_saved = net_input.detach().clone()
        
        self.arg_sigma_z = args.sigma_z
        self.arg_epsilon = args.epsilon
        self.divergence = self.divergence_ty
        self.uniform_sigma = False
        self.clip_divergence = False
        
        # parameter related to SURE.
        self.scale = args.scale
        self.eps = 0.01
        self.epsilon_decay = False
        self.reduction = "mean"

        print("[*] loss type : %s" % args.dip_type)
        print("[*] Poisson scale : %.2f" % self.scale)
        # print("[*] sigma_z : %.2f" % self.sigma_z)
        self.uniform_sigma = False
        self.eps_decay = False
        if self.dip_type == "dip":
            self.loss = self.DIP
        elif self.dip_type == "PURE":
            self.loss = self.DIP_PURE
        elif self.dip_type == "PURE_dc":
            self.eps = 0.1
            self.eps_decay = True
            self.loss = self.DIP_PURE

    def PURE(self, output, target, scale):
        Y_ = output
        Y  = target
        b_prime = 2*(torch.randint_like(target, 0, 2) - 0.5) # [-1, 1] random vector
        if self.eps_decay and (self.cnt % 20 == 9):
            self.eps *= 0.9
        Z  = Y + self.eps * b_prime
        Z_ = self.inference(Z)
        batch, c, h, w = output.shape
        mse = torch.mean((Y - Y_) ** 2)
        T1  = - scale * torch.mean(target)# / batch
        gradient = 2*(scale / (self.eps * batch)) * torch.mean((b_prime *Y) * (Z_ - Y_))
        return mse + T1 + gradient

    def DIP_PURE(self, net_input, noisy_torch):
        out = self.inference(net_input)  # .contiguous(memory_format=torch.channels_last)
        total_loss = self.PURE(out, noisy_torch, self.scale)
        return total_loss, out


def get_loss(net, net_input, args):
    if args.task_type == "denoising":
        print("[!] Denoising mode setup.")
        return Denoising_loss(net, net_input, args)
    elif args.task_type == "poisson":
        return Poisson_loss(net, net_input, args)
    else:
        raise NotImplementedError

