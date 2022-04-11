from locale import normalize
import os
from tkinter import Image
from torchvision.utils import save_image, make_grid
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import spectral_norm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from dataset import MultiResolutionDataset
from models.diff_augment import DiffAugment
from models.discriminator import ProjectedDiscriminator
from models.spectral import SpectralNorm
from random import random
import math
from PIL import Image

def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API

def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def denormalize(x, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)



class InitLayer(nn.Module):
    def __init__(self, nz, channel, sz=4):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel*2, sz, 1, 0, bias=False),
            NormLayer(channel*2),
            GLU(),
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlockSmall(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NormLayer(out_planes*2), GLU())
    return block



def UpBlockBig(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU()
        )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, z_dim=100, nc=3, im_size=1024):
        super().__init__()

        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = im_size
        self.init = InitLayer(z_dim, channel=nfc[4])
        
        # UpBlock = UpBlockSmall if lite else UpBlockBig
        
        self.feat_8   = UpBlockBig(nfc[4], nfc[8])
        self.feat_16  = UpBlockSmall(nfc[8], nfc[16])
        self.feat_32  = UpBlockBig(nfc[16], nfc[32])
        self.feat_64  = UpBlockSmall(nfc[32], nfc[64])
        self.feat_128 = UpBlockBig(nfc[64], nfc[128])  
        self.feat_256 = UpBlockSmall(nfc[128], nfc[256]) 

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=True)

    
        
        if im_size >= 512:
            self.feat_512 = UpBlockBig(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size >= 1024:
            self.feat_1024 = UpBlockSmall(nfc[512], nfc[1024])
        
    def forward(self, input):
        
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        if self.img_resolution >= 64:
            feat_last = feat_64

        if self.img_resolution >= 128:
            feat_last = self.se_128(feat_8,  self.feat_128(feat_last))

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)
        return self.to_big(feat_last)






class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

    
class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)
        
class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)
    
    
    
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)        
    


class ProjectedGAN(LightningModule):
    def __init__(self,
                image_size: int = 64,
                 z_dim: int = 128,
                 lr: float = 0.001,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 preview_num: int =  16,
                 traindataset = None,
                 preview_path = "default",
                 s2c = None,
                 c2s = None,
                 **kwargs):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.size = image_size
        self.z_dim = z_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.preview_num = preview_num
        # networks
        self.G = Generator(z_dim=self.z_dim,im_size=image_size)
        self.G.apply(weights_init)
        
        self.D = ProjectedDiscriminator(im_res=image_size,backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'])
        # self.D = ProjectedDiscriminator(im_res=image_size,backbones=['tf_efficientnet_lite0'])
        self.D.feature_networks.eval()
        self.validation_z = self._make_noise(self.z_dim, 1)

        self.s2c = s2c
        self.c2s = c2s
        self.preview_path = preview_path
        self.traindataset = traindataset
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),

        ])

        

    def forward(self, z):
        return self.G(z)

    def process_cmd(self):
        if not self.s2c.empty():
            msg = self.s2c.get()
            if msg == "preview":
                self.send_previw()
            elif msg == "random_z":
                self.validation_z =  self._make_noise(self.z_dim, 1)
            elif msg == "stop":
                self.trainer.should_stop = True
            else:
                pass

    def send_previw(self):
        with torch.no_grad():
            z = self.validation_z.to(self.device)
            sample_imgs = self.G(z).cpu()
            self.c2s.put({'op':"show",'previews': sample_imgs})              

            
    def _make_noise(self, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(self.preview_num, latent_dim, device=self.device)
        return torch.randn(n_noise, self.preview_num, latent_dim, device=self.device)


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.process_cmd()
        imgs = batch
        self.D.feature_networks.requires_grad_(False)
        # train generator
        if optimizer_idx == 0:
            f_preds = self._get_fake_predict()
            g_loss = sum([(-l).mean() for l in f_preds])
            self.log('g_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True)
            return g_loss
        # train discriminator
        if optimizer_idx == 1:
            r_preds = self.D(imgs)
            f_preds = self._get_fake_predict()
            d_loss_real =  sum([(F.relu(torch.ones_like(l) - l)).mean() for l in r_preds])
            d_loss_fake = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in f_preds])
            d_loss = d_loss_real + d_loss_fake
            self.log('d_loss', d_loss, on_step=False,on_epoch=True, prog_bar=True)
            return d_loss
        
        
    def _get_fake_predict(self):
        fake_img = self._get_fake_img()
        return self.D(fake_img)
    
    def _get_fake_img(self):
        noise = self._make_noise(self.z_dim, 1)
        return self.G(noise)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1, b2))
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g,[10,20,30], gamma=0.1, last_epoch=-1)

        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1, b2))
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_g,[10,20,30], gamma=0.1, last_epoch=-1)

        return [opt_g, opt_d] ,[scheduler_g, scheduler_d]

    def train_dataloader(self):
        train = MultiResolutionDataset(self.traindataset,self.data_transform,resolution=self.size)
        return DataLoader(train, batch_size=self.batch_size,num_workers=24,persistent_workers = True,shuffle=True,pin_memory=True,drop_last=True)

    def on_train_epoch_end(self):
        if not os.path.exists(f'previews/{self.preview_path}'):
            os.makedirs(f'previews/{self.preview_path}')
        z = self.validation_z.to(self.device)
        imgs = self(z)
        nrow = int(math.sqrt(len(z)))
        save_image(imgs,f"previews/{self.preview_path}/epoch_{self.current_epoch+1}.png",value_range=(-1,1),normalize=True,nrow =nrow)

