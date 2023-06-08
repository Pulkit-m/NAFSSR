# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import torchvision 
import scipy 
from scipy.optimize import curve_fit 
from joblib import Parallel, delayed 

from torchvision import transforms
from torchvision.models import vgg19 
from torchvision.transforms import GaussianBlur

import random, os 
from tqdm import tqdm 
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()





#### KLTStereoSRQA
class KLTStereoSRQA(nn.Module): 
    """
        using the abstraction KLTSRQA 
        twice to implement the function for stereo image loss function 
    """
    def __init__(self, loss_weight = 1.0, KLT_kernel_path = '../KLT_kernel_64.mat'): 
        super(KLTStereoSRQA, self).__init__() 
        self.loss_weight = loss_weight 

        self.KLT_kernel_path = KLT_kernel_path 
        self.leftKLTLoss = KLTSRQA(KLT_kernel_path=self.KLT_kernel_path) 
        self.rightKLTLoss = KLTSRQA(KLT_kernel_path = self.KLT_kernel_path)

        
    def forward(self, pred, target):
        # print("DEVICE OF PRED: ", pred.requires_grad, "\nDEVICE OF TARGET: ", target.requires_grad)

        sr_l, sr_r = torch.split(pred,(3,3),dim=1)  
        hr_l, hr_r = torch.split(target,(3,3),dim=1) 

        try: 
            self.loss_left = self.leftKLTLoss(sr_l, hr_l) 
            self.loss_right = self.rightKLTLoss(sr_r, hr_r) 

            # print(self.loss_left, self.loss_right)

            return self.loss_weight*(self.loss_left + self.loss_right) 
        except:
            print("wtf",0,0) 
            return None



#### KLTSRQA 
class KLTSRQA(nn.Module): 
    """ 
        Introduction of the module in detail 
    """

    def __init__(self, loss_weight = 1.0, reduction = 'mean', KLT_kernel_path = '../KLT_kernel_64.mat'): 
        super(KLTSRQA, self).__init__() 
        self.loss_weight = loss_weight 
        self.reduction = reduction 

        # one time loaders
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gaussian_filter = GaussianBlur(kernel_size= 7, sigma = 7/6)
        # print(self.gaussian_filter)
        self.KLT_kernel_64 = torch.tensor(scipy.io.loadmat(KLT_kernel_path)['KLT_kernel_64'], device = self.device)  

        
    def forward(self, pred, target): 
        """ 
            pred: should be of shape: [N,3,H,W] 
            target: should be of shape: [N,3,H,W]
            equivalent to KLT_Features function in the matlab code 

            returns: a single value equivalent to MSE Loss or any other loss between 
                    KLT Features of pred and KLT Features of target
        """
        with torch.no_grad():    
            pred_features = self.KLT_Features(pred) 
            target_features = self.KLT_Features(target) 

        # if self.reduction == 'mean': 
            # return ((pred_features - target_features)**2)/777.0  
        return torch.mean((pred_features - target_features)**2) 


    def KLT_Features(self, img):   
        """ 
            img: should be of shape: [N, 3, H, W]
            returns: KLT Features: shape: [N, 777]
        """

        # color space change 
        img_o = self.color_change(img)  
        img_o1 = img_o[:,0,:,:] 
        img_o2 = img_o[:,1,:,:]
        img_o3 = img_o[:,2,:,:] 

         
        # MSCN Coefficients map compute 
        o1_MSCN = self.MSCN_compute(img_o1) 
        o2_MSCN = self.MSCN_compute(img_o2) 
        o3_MSCN = self.MSCN_compute(img_o3)  

        # img patch extract 
        # kernel size = 64, using pre trained kernel directly 
        # MSCN coefficients are of shape [N,1,H,W] 
        test_matrix_o1_64 = self.patch_extract(o1_MSCN,64,2)
        test_matrix_o2_64 = self.patch_extract(o2_MSCN,64,2)
        test_matrix_o3_64 = self.patch_extract(o3_MSCN,64,2)    

        # KLT Kernel loaded already 
        # print(KLT_kernel_64.shape, test_matrix_o1_64.shape) 

        # kernel_size = 64 
        # shapes: 64,64 and N,64,NumPatches 
        kernel1 = self.KLT_kernel_64[:,:,0][None, :, :]
        # print(kernel1.device, test_matrix_o1_64.device)
        KLT_o1_64 = torch.matmul( torch.transpose(kernel1,1,2) , test_matrix_o1_64) 
        kernel2 = self.KLT_kernel_64[:,:,1][None, :, :] 
        KLT_o2_64 = torch.matmul( torch.transpose(kernel2,1,2) , test_matrix_o2_64)
        kernel3 = self.KLT_kernel_64[:,:,2][None, :, :] 
        KLT_o3_64 = torch.matmul( torch.transpose(kernel3,1,2) , test_matrix_o3_64)      


     
        # KLT Energy Compute 
        E_o1_64 = torch.mean(KLT_o1_64*KLT_o1_64, dim=2)
        E_o2_64 = torch.mean(KLT_o2_64*KLT_o2_64, dim=2)
        E_o3_64 = torch.mean(KLT_o3_64*KLT_o3_64, dim=2)

        N, _ = E_o1_64.shape  
        index = torch.arange(0,64)
        # exp_o1 = self.run_parallel(N, index, E_o1_64.cpu().numpy())
        exp_o1 = self.run_parallel(N, index, E_o1_64) 
        exp_o2 = self.run_parallel(N, index, E_o2_64) 
        exp_o3 = self.run_parallel(N, index, E_o3_64) 
        # print(exp_o3)
        # print(exp_o3.shape)
        feat = torch.cat([exp_o1, exp_o2, exp_o3], dim=1) 
        # print(feat.shape)
        sample = self.estimateaggdparam(KLT_o1_64.flatten(start_dim=1, end_dim=2)) 
        # print("Checking:" ,sample.requires_grad)
        feat = torch.cat([      self.estimateaggdparam(KLT_o1_64.flatten(start_dim=1, end_dim=2)), 
                                self.estimateaggdparam(KLT_o2_64.flatten(start_dim=1, end_dim=2)),
                                self.estimateaggdparam(KLT_o3_64.flatten(start_dim=1, end_dim=2))], dim = 1) 

        # print(feat.shape) 
        # print(KLT_o1_64.shape)

        for k in range(64): 
            feat = torch.cat([
                feat, 
                self.estimateaggdparam(KLT_o1_64[:,k,:]), 
                self.estimateaggdparam(KLT_o2_64[:,k,:]), 
                self.estimateaggdparam(KLT_o3_64[:,k,:])
            ], dim = 1)

        # print("Requires grad: ",feat.requires_grad)  
        return feat      


    def color_change(self, img):
        """ 
            takes input image batch of shape: N, 3, H, W 
            outputs modified image batch of shape: N, 3, H, W
        """
        img_out = torch.zeros(img.shape)
        
        img_r = img[:, 0, :, :]
        img_g = img[:, 1, :, :]
        img_b = img[:, 2, :, :]

        # color space change
        img_o1 = 0.06 * img_r + 0.63 * img_g + 0.27 * img_b
        img_o2 = 0.30 * img_r + 0.04 * img_g - 0.35 * img_b
        img_o3 = 0.34 * img_r - 0.60 * img_g + 0.17 * img_b

        img_out[:, 0, :, :] = img_o1
        img_out[:, 1, :, :] = img_o2
        img_out[:, 2, :, :] = img_o3

        return img_out

    def MSCN_compute(self, imdist): 
        """ 
            input is of shape: N, H, W 
            output is of shape: N, 1, H, W 
        """ 
        imdist = imdist[:,None,:,:]
        with torch.no_grad():
            mu = self.gaussian_filter(imdist) # [N,H,W]
            mu_sq = mu*mu   
            sigma = torch.sqrt(torch.abs(self.gaussian_filter(imdist*imdist)-mu_sq)) 
        structdis = (imdist - mu) / (sigma + 1) 
        return structdis[:,:,:]  
    


    def modcrop(self, img, q): 
        """
            modcrop - Crops an image so that the output M-by-N image satisfies mod(M,q)=0 and mod(N,q)=0
            image will of shape: N, 1, H, W 
            output should be N, 1, H', W' 
        """   
        sz = torch.tensor(img.shape, device = 'cpu') 
        sz = sz - sz%q 
        img = img[:,:,:int(sz[2]), :int(sz[3])] 
        return img 

    def patch_extract(self, img, k= 64, id= 2): 
        """ 
            input img is of shape: N, 1, H, W 
            output is of shape: N, 64, NumPatches
        """ 

        k_sqrt = int(torch.sqrt(torch.Tensor([k])))
        img = self.modcrop(img, k_sqrt)   
        N, C, H, W = img.shape 
        m = H//k_sqrt 
        n = W//k_sqrt 

        # print(N,C,H,W)
        patch_matrix = [] 

        for i in range(m): 
            for j in range(n): 
                patch = img[:,:,i * k_sqrt : (i + 1) * k_sqrt, j * k_sqrt : (j + 1) * k_sqrt] 
                # print(patch.shape)
                patch_vec = patch.reshape(N,k_sqrt*k_sqrt,1) 
                # print(patch_vec.shape) 
                if id == 1: 
                    continue 
                elif id ==2: 
                    patch_matrix.append(patch_vec) 

        return torch.cat(patch_matrix, dim = 2).type(torch.float64).to(self.device)
    
    def run_parallel(self, N, index, data):
        def f(x, a, b, c):
            return a * np.exp(b * x) + c
        
        def f_torch(x, a, b, c): 
            return a * torch.exp(b * x) + c

        def fit_curve(index, data):
            # try: 
            c, _ = curve_fit(f, index.cpu().numpy(), data.detach().cpu().numpy(), p0=[0, 0, 0], maxfev=5000)
            out = f_torch(index, *c) #this is exp_o1/2/3 
            # print("ouput of the function: ",out.shape) 
            return out[None,:]
            # except: 
            #     print(data)
            

        exp_results = Parallel(n_jobs=N)(delayed(fit_curve)(index, data[i]) for i in range(N))
        # print(exp_results)
        # return torch.tensor(exp_results) 
        # print(exp_results[0].shape) 
        # print(torch.cat(exp_results, dim = 0).shape)
        return torch.cat(exp_results, dim = 0)
    

    def torch_gamma(self, x):
        return torch.exp(torch.lgamma(x))


    def estimateaggdparam(self, vec): 
        """ 
            input should be a numpy.ndarray
            input is of shape: N, NumPatches*64
            expecting output to be returns an array of 3 numpy arrays each having a vector of size 4
        """
        N,_  = vec.shape

        gam = torch.arange(0.2, 10.0001, 0.001).to(self.device)
        r_gam = ((self.torch_gamma(2 / gam)) ** 2) / (self.torch_gamma(1 / gam) * self.torch_gamma(3 / gam))
        r_gam = r_gam.view(1, -1).expand(N, -1)
        # print("r_gam:",r_gam.device)

        throwAwayThresh = 0.0
        comp1 = vec < -throwAwayThresh
        comp1_sum = comp1.sum()
        leftstd = torch.sqrt(torch.sum((comp1 * vec) ** 2, dim=1) / comp1_sum)

        comp2 = vec > throwAwayThresh
        comp2_sum = comp2.sum()
        rightstd = torch.sqrt(torch.sum((comp2 * vec) ** 2, dim=1) / comp2_sum)

        # print(leftstd.shape, rightstd.shape)
        gammahat = torch.divide(leftstd, rightstd)
        # print("gammahat:",gammahat.shape)

        vec1 = vec
        rhat = torch.mean(torch.abs(vec1), dim=1) ** 2 / torch.mean((vec1) ** 2, dim=1)
        # print("rhat:",rhat.device)
        rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
        rhatnorm = rhatnorm.view(-1, 1)
        # print("rhatnorm:",rhatnorm.device)

        array_position = torch.argmin((r_gam - rhatnorm) ** 2, dim=1)
        alpha = gam[array_position]
        # print(alpha, leftstd, rightstd) 
        # print(alpha.shape, leftstd.shape, rightstd.shape)
        para = torch.stack([alpha, leftstd, rightstd], dim=1)
        return para




#### TEXTURE LOSS 

class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(0), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features


class TextureLossVGG19(torch.nn.Module):
    def __init__(self, loss_weight = 1.): 
        super(TextureLossVGG19, self).__init__()
        self.loss_weight = loss_weight 
        
        texture_layers = ['8','17','26','35']
        vgg_layers = [int(i) for i in texture_layers]
        self.vgg_texture = VGG(layers=vgg_layers, replace_pooling = False)
        if torch.cuda.is_available():
            self.vgg_texture = self.vgg_texture.cuda()
    
    
    def forward(self, inp, gt): 
        sr_l, sr_r = torch.split(inp,(3,3),dim=1)  
        hr_l, hr_r = torch.split(gt,(3,3),dim=1) 
        
        # we process both the left view and right view one at a time
        # Processing the Left View: 
        vgg_sr = self.vgg_texture.forward(sr_l) 
        vgg_gt = self.vgg_texture.forward(hr_l) 
        text_loss_l = []
        gram_sr = [self.gram_matrix(y) for y in vgg_sr]
        gram_gt = [self.gram_matrix(y) for y in vgg_gt]
        for m in range(0,len(vgg_sr)):
            text_loss_l += [self.criterion(gram_sr[m],gram_gt[m])]
        text_loss_l = sum(text_loss_l)
       
        # Processing the right view: 
        vgg_sr = self.vgg_texture.forward(sr_r) 
        vgg_gt = self.vgg_texture.forward(hr_r) 
        text_loss_r = [] 
        gram_sr = [self.gram_matrix(y) for y in vgg_sr]
        gram_gt = [self.gram_matrix(y) for y in vgg_gt]
        for m in range(0,len(vgg_sr)):
            text_loss_r += [self.criterion(gram_sr[m],gram_gt[m])]
        text_loss_r = sum(text_loss_r)
        
        texture_loss = text_loss_l + text_loss_r 
        texture_loss = texture_loss * self.loss_weight 
        
        return texture_loss
    
        
    
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def criterion(self,a, b):
        return torch.mean(torch.abs((a-b)**2).view(-1))
    




### PERCEPTUAL LOSS ####
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, loss_weight=1, reduction='mean'):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_weight = loss_weight

        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # if input.shape[1] != 3:
        #     input = input.repeat(1, 3, 1, 1)
        #     target = target.repeat(1, 3, 1, 1)
        
        # print(input.shape)
        sr_l, sr_r = torch.split(input,(3,3),dim=1)  
        hr_l, hr_r = torch.split(target,(3,3),dim=1) 


        sr_l = (sr_l - self.mean) / self.std 
        hr_l = (hr_l-self.mean)/self.std
        if self.resize: 
            sr_l = self.transform(sr_l, mode = 'bilinear', size = (224,224), align_corners=False) 
            hr_l = self.transform(hr_l, mode = 'bilinear', size = (224,224), align_corners=False) 
        loss_l = 0.0 
        x = sr_l 
        y = hr_l  
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_l += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss_l += torch.nn.functional.l1_loss(gram_x, gram_y)

        
        sr_r = (sr_r - self.mean) / self.std 
        hr_r = (hr_r-self.mean)/self.std
        if self.resize: 
            sr_r = self.transform(sr_r, mode = 'bilinear', size = (224,224), align_corners=False) 
            hr_r = self.transform(hr_r, mode = 'bilinear', size = (224,224), align_corners=False) 
        loss_r = 0.0 
        x = sr_r 
        y = hr_r  
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_r += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss_r += torch.nn.functional.l1_loss(gram_x, gram_y)

        # # ORIGINAL CODE FOR VGG LOSS
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        # loss = 0.0
        # x = input
        # y = target
        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     y = block(y)
        #     if i in feature_layers:
        #         loss += torch.nn.functional.l1_loss(x, y)
        #     if i in style_layers:
        #         act_x = x.reshape(x.shape[0], x.shape[1], -1)
        #         act_y = y.reshape(y.shape[0], y.shape[1], -1)
        #         gram_x = act_x @ act_x.permute(0, 2, 1)
        #         gram_y = act_y @ act_y.permute(0, 2, 1)
        #         loss += torch.nn.functional.l1_loss(gram_x, gram_y)


        # print("#################################\n \
        #         #################################\n \
        #         THIS IS THE PERCEPTUAL LOSS\n")
        # print(loss_l, loss_r) 
        # print("#################################\n \
        #         #################################\n ")
        loss_l = loss_l * self.loss_weight
        loss_r = loss_r * self.loss_weight
        return loss_l, loss_r  
