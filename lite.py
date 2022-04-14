import io
import torch
from torch import nn
from pytorch_lightning.lite import LightningLite
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.fastgan import Generator
from models.discriminator import ProjectedDiscriminator
from models.discriminator import ProjectedDiscriminator
from torch.utils.data import DataLoader
from dataset import MultiResolutionDataset
from tqdm import tqdm
from os.path import exists


class Lite(LightningLite):

    def _make_noise(self, latent_dim, n_noise):
        if n_noise == 1:
            return torch.randn(self.preview_num, latent_dim, device=self.device)
        return torch.randn(n_noise, self.preview_num, latent_dim, device=self.device)
        
    def _get_fake_predict(self, blur_sigma):
        fake_img = self._get_fake_img()
        return self.D(fake_img,blur_sigma)

    def _get_fake_img(self):
        noise = self._make_noise(self.z_dim, 1)
        img,_ws = self.G(noise)
        return img

    def process_cmd(self):
        if self.s2c == None:
            return
        if not self.s2c.empty():
            msg = self.s2c.get()
            if msg == "preview":
                self.send_previw()
            elif msg == "random_z":
                self.validation_z =  self._make_noise(self.z_dim, 1)
            elif msg == "stop":
                self.save_ckpt()
                self.should_stop = True
            else:
                pass 

    def send_previw(self):
        with torch.no_grad():
            z = self.validation_z.to(self.device)
            sample_imgs,_ws = self.G(z)
            self.c2s.put({'op':"show",'previews': sample_imgs.cpu()}) 


    def save_ckpt(self):
        dict = {
            'G': self.G.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'D': self.D.state_dict(),
            'opt_D': self.opt_D.state_dict(),
        }
        self.save(dict, f"./check_points/{self.preview_path}/last_lite.ckpt")
        
    def load_ckpt(self):
        ckpt_path =  f"./check_points/{self.preview_path}/last_lite.ckpt"
        if exists(ckpt_path):
            print(f"load last_ckpt from {ckpt_path}")
            dict = self.load(ckpt_path)
            self.G.load_state_dict(dict['G'])
            self.opt_G.load_state_dict(dict['opt_G'])
            self.D.load_state_dict(dict['D'])
            self.opt_D.load_state_dict(dict['opt_D'])

    def run(self,num_epochs: int, im_size: int = 64,
                z_dim: int = 64,
                lr: float = 0.0025,
                b1: float = 0,
                b2: float = 0.999,
                batch_size: int = 64,
                preview_num: int =  16,
                traindataset = None,
                preview_path = "default",
                s2c = None,
                c2s = None):

        self.s2c = s2c
        self.c2s = c2s
        self.should_stop = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.im_size = im_size
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

        self.validation_z = self._make_noise(self.z_dim, 1)
        self.preview_path = preview_path
        self.traindataset = traindataset
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),

        ])
        self.pl_mean = torch.zeros([], device=self.device)



        self.G = Generator(z_dim=self.z_dim,im_size=self.im_size)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.LRS_G = torch.optim.lr_scheduler.MultiStepLR(self.opt_G,[10,20,30], gamma=0.1, last_epoch=-1)
        self.setup(self.G,self.opt_G)
        self.D = ProjectedDiscriminator(im_res=self.im_size,backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'])
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.LRS_D = torch.optim.lr_scheduler.MultiStepLR(self.opt_D,[10,20,30], gamma=0.1, last_epoch=-1)
        self.setup(self.D,self.opt_D)
        dataset = MultiResolutionDataset(self.traindataset,self.data_transform,resolution=self.im_size)
        data_loader = DataLoader(dataset,batch_size=self.batch_size,num_workers=8,persistent_workers = True,shuffle=True,pin_memory=True,drop_last=True)
        self.load_ckpt()
        self.G.train(True)
        self.D.train(True)
        self.global_step = 0
        for epoch in range(num_epochs):
            bar = tqdm(data_loader)
            for i, imgs in enumerate(bar):


                self.global_step += 1
                self.process_cmd()
                if self.should_stop:
                    break
                blur_sigma = max(1 - self.global_step / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
                # ========= TrainG ============
                self.G.requires_grad_(True)
                self.D.requires_grad_(False)
                noise = self._make_noise(self.z_dim, 1)
                fake_img, _ws = self.G(noise)
                f_preds =self.D(fake_img,blur_sigma)     
                g_loss = sum([(-l).mean() for l in f_preds])
                self.opt_G.zero_grad()
                self.backward(g_loss)
                self.opt_G.step()
                # ========= TrainD ============
                self.G.requires_grad_(False)
                self.D.discriminators.requires_grad_(True)
                # self.D.feature_networks.requires_grad_(False)
                imgs = imgs.to(device)
                r_preds = self.D(imgs,blur_sigma)
                f_preds = self._get_fake_predict(blur_sigma)
                d_loss_real =  sum([(F.relu(torch.ones_like(l) - l)).mean() for l in r_preds])
                d_loss_fake = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in f_preds])
                d_loss = d_loss_real + d_loss_fake
                self.opt_D.zero_grad()
                self.backward(d_loss)
                self.opt_D.step()


if __name__ == '__main__':
    trainer =  Lite(accelerator= "gpu")
    trainer.run(1,im_size=256,z_dim=64,batch_size=8,preview_num=9,traindataset="sexyface")

