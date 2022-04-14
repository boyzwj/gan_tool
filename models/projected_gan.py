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
from random import random
import math
from PIL import Image
from torch_utils.ops import conv2d_gradfix

from models.fastgan import Generator


class ProjectedGAN(LightningModule):
    def __init__(self,
                image_size: int = 64,
                 z_dim: int = 64,
                 lr: float = 0.0025,
                 b1: float = 0,
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
        self.blur_fade_kimg = 300
        self.blur_init_sigma = 2

        self.pl_weight = 10
        self.pl_decay = 0.01
        self.pl_mean = torch.zeros([], device=self.device)
        # networks
        self.G = Generator(z_dim=self.z_dim,im_size=image_size)
        
        self.D = ProjectedDiscriminator(im_res=image_size,backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'])
        # self.D = ProjectedDiscriminator(im_res=image_size,backbones=['tf_efficientnet_lite0'])
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
            sample_imgs,_ws = self.G(z)
            self.c2s.put({'op':"show",'previews': sample_imgs.cpu()})              

            
    def _make_noise(self, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(self.preview_num, latent_dim, device=self.device,requires_grad=True)
        return torch.randn(n_noise, self.preview_num, latent_dim, device=self.device,requires_grad=True)


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.process_cmd()
        imgs = batch
        blur_sigma = max(1 - self.global_step / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        # train generator
        if optimizer_idx == 0:
            noise = self._make_noise(self.z_dim, 1)
            fake_img, _ws = self.G(noise)
            g_loss = sum([(-l).mean() for l in f_preds])
            if self.global_step % 4 == 0:
                pl_noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
                with conv2d_gradfix.no_weight_gradients(True):
                    pl_grads = torch.autograd.grad(outputs=[(fake_img * pl_noise).sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    pl_penalty = (pl_lengths - pl_mean).square()
                    pl_loss = pl_penalty * self.pl_weight
                    g_loss = g_loss + pl_loss.mean()
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            self.D.feature_networks.requires_grad_(False)
            r_preds = self.D(imgs,blur_sigma)
            f_preds = self._get_fake_predict(blur_sigma)
            d_loss_real =  sum([(F.relu(torch.ones_like(l) - l)).mean() for l in r_preds])
            d_loss_fake = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in f_preds])
            d_loss = d_loss_real + d_loss_fake
            self.log('d_loss', d_loss, on_step=True,on_epoch=True, prog_bar=True)
            return d_loss

         

        
    def _get_fake_predict(self, blur_sigma):
        fake_img = self._get_fake_img()
        return self.D(fake_img,blur_sigma)
    
    def _get_fake_img(self):
        noise = self._make_noise(self.z_dim, 1)
        img,_ws = self.G(noise)
        return img

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
        return DataLoader(train, batch_size=self.batch_size,num_workers=8,persistent_workers = True,shuffle=True,pin_memory=True,drop_last=True)

    def on_train_epoch_end(self):
        if not os.path.exists(f'previews/{self.preview_path}'):
            os.makedirs(f'previews/{self.preview_path}')
        z = self.validation_z.to(self.device)
        imgs = self(z)
        nrow = int(math.sqrt(len(z)))
        save_image(imgs,f"previews/{self.preview_path}/epoch_{self.current_epoch+1}.png",value_range=(-1,1),normalize=True,nrow =nrow)

