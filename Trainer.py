from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import cv2
import matplotlib.pyplot as plt

from models.projected_gan import ProjectedGAN
import torch

def trainerThread(cfg, s2c, c2s):
    model = ProjectedGAN(
    image_size = cfg["image_size"],
    z_dim = cfg["latent_dim"],
    batch_size = cfg["batch_size"],
    preview_num = cfg["preview_num"],
    traindataset= "sexyface",
    preview_path = "pj_gan3",
    lr = 0.0025,
    s2c=s2c,
    c2s=c2s)


    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./check_points/{model.preview_path}/',
        # every_n_train_steps = 1000,
        save_on_train_epoch_end = True,
        save_last=True)

    resume_from_checkpoint = None
    if cfg["resume"]:
        resume_from_checkpoint = f'{checkpoint_callback.dirpath}/last.ckpt'


    trainer = Trainer(
            callbacks=[checkpoint_callback],
            gpus=1,
            amp_backend ="apex", 
            amp_level='O2',
            check_val_every_n_epoch=1000, 
            max_epochs=5000)

    # trainer.tune(model)
    trainer.fit(model,ckpt_path = resume_from_checkpoint)


