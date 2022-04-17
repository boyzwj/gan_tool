from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.projected_gan import ProjectedGAN
import torch
from lite  import Lite


def trainerThread(cfg, s2c, c2s):
    model = ProjectedGAN(
    image_size = cfg["image_size"],
    z_dim = cfg["latent_dim"],
    batch_size = cfg["batch_size"],
    preview_num = cfg["preview_num"],
    traindataset= cfg["dataset"],
    preview_path = cfg["train_name"],
    lr = cfg["learning_rate"],
    s2c=s2c,
    c2s=c2s)


    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./check_points/{model.preview_path}/',
        every_n_train_steps = 1000,
        save_on_train_epoch_end = True,
        save_last=True)

    resume_from_checkpoint = None
    if cfg["resume"]:
        resume_from_checkpoint = f'{checkpoint_callback.dirpath}/last.ckpt'


    trainer = Trainer(
            # precision=16,
            # amp_backend ="apex", 
            # amp_level='O1',
            callbacks=[checkpoint_callback],
            gpus=1,
            check_val_every_n_epoch=1000, 
            max_epochs=50)

    # trainer.tune(model)
    trainer.fit(model,ckpt_path = resume_from_checkpoint)


def trainerLite(cfg, s2c, c2s):
    trainer =  Lite(accelerator= "gpu",precision=16)
    trainer.run(1,
    im_size = cfg["image_size"],
    z_dim = cfg["latent_dim"],
    batch_size = cfg["batch_size"],
    preview_num = cfg["preview_num"],
    traindataset= "sexyface",
    preview_path = "pj_gan",
    lr = 0.003,
    s2c=s2c,
    c2s=c2s)