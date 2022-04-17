import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c=None, **kwargs):
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
    
    
           


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


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
    def __init__(self, ngf=64, z_dim=100, nc=3, im_size=1024,lite = True):
        super().__init__()

        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = im_size
        self.init = InitLayer(z_dim, channel=nfc[4])
        self.mapping = DummyMapping()
        
        UpBlock = UpBlockSmall if lite else UpBlockBig
        
        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

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
        ws = self.mapping(input)
        feat_4 = self.init(ws)
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