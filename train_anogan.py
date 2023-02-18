import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy

from models import Discriminator
from models import Generator
from utils import load_json
from utils import check_manual_seed
from utils import norm
from utils import denorm
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio.settings import TRAIN_PATIENT_IDS
from dataio.settings import TEST_PATIENT_IDS
from dataio import MNISTDataModule


class Anogan(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.

        # networks
        self.D = Discriminator().float()
        self.G = Generator(input_dim=self.config.model.input_dim, z_dim=self.config.model.z_dim, filters=self.config.model.gen_filters, activation=self.config.model.gen_activation, final_activation=self.config.model.gen_final_activation).float()


    def anomaly_score(self, input_image, fake_image, D):
        # Residual loss の計算
        residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

        # Discrimination loss の計算
        _, real_feature = D(input_image)
        _, fake_feature = D(fake_image)
        discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

        # 二つのlossを一定の割合で足し合わせる
        total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
        total_loss = total_loss_by_image.sum()

        return total_loss, total_loss_by_image, residual_loss

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss


    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (
            torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
        )
        return d_loss

    def criterion(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        return loss(y_hat, y)


    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']      

                    #self.G.eval()
                    #self.D.eval()

                    #z = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
                    #z.requires_grad = True
                    #z_optimizer = torch.optim.Adam([z], lr=1e-3)

                    #with torch.enable_grad():
                    #    for epoch in range(self.config.training.z_epochs):
                    #        #z探し
                    #        fake_images = self.G(z)
                    #        loss, _, _ = self.anomaly_score(image, fake_images, self.D)
                    #        z_optimizer.zero_grad()
                    #        loss.backward()
                    #        z_optimizer.step()
                    
                    #異常度の計算
                    #fake_images = self.G(z)

                    z_0 = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
                    sample_images = self.G(z_0)

                    image = image.detach().cpu()
                    #fake_images = fake_images.detach().cpu()
                    sample_images = sample_images.detach().cpu()

                    image = image[:self.config.save.n_save_images, ...]
                    #fake_images = fake_images[:self.config.save.n_save_images, ...]
                    sample_images = sample_images[:self.config.save.n_save_images, ...]
                    #self.logger.train_log_images(torch.cat([image, fake_images, sample_images]), self.current_epoch-1)
                    self.logger.train_log_images(torch.cat([image, sample_images]), self.current_epoch-1)

                    #self.G.train()
                    #self.D.train()

        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)
        
        g_optim, d_optim = self.optimizers()
        
        image = batch['image']
         
        # 潜在変数から偽の画像を生成
        z = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
        fake_images = self.G(z)
        d_out_fake, _ = self.D(fake_images)
        g_loss = - torch.mean(d_out_fake) 

        g_optim.zero_grad()
        self.manual_backward(g_loss * self.config.training.g_weight)
        g_optim.step()

        fake_images = self.G(z)

        for _ in range(self.config.training.n_inner_loops):
            l_real, _ = self.D(image.detach())
            l_fake, _ = self.D(fake_images.detach())

            d_loss = self.hinge_d_loss(l_real, l_fake) 

            d_optim.zero_grad()
            self.manual_backward(d_loss * self.config.training.d_weight)
            d_optim.step()

        if self.needs_save:
            self.log('G_Loss', g_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            self.log('D_Loss', d_loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
                
        return {'generator_loss': g_loss, 'discriminator_loss' : d_loss}


    def validation_step(self, batch, batch_idx):
        if self.config.training.val_mode == "train":
            for m in self.G.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

            for m in self.D.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']
                    
                    #z = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
                    #z.requires_grad = True
                    #z_optimizer = torch.optim.Adam([z], lr=1e-3)

                    #with torch.enable_grad():
                    #    for epoch in range(self.config.training.z_epochs):
                    #        #z探し
                    #        fake_images = self.G(z)
                    #        loss, _, _ = self.anomaly_score(image, fake_images, self.D)
                    #        z_optimizer.zero_grad()
                    #        loss.backward()
                    #        z_optimizer.step()
                    
                    #異常度の計算
                    #fake_images = self.G(z)

                    z_0 = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
                    sample_images = self.G(z_0)

                    image = image.detach().cpu()
                    #fake_images = fake_images.detach().cpu()
                    sample_images = sample_images.detach().cpu()

                    image = image[:self.config.save.n_save_images, ...]
                    #fake_images = fake_images[:self.config.save.n_save_images, ...]
                    sample_images = sample_images[:self.config.save.n_save_images, ...]
                    #self.logger.val_log_images(torch.cat([image, fake_images, sample_images]), self.current_epoch)
                    self.logger.val_log_images(torch.cat([image, sample_images]), self.current_epoch)

        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)

        image = batch['image']

        # 潜在変数から偽の画像を生成
        z = torch.randn(image.size(0), self.config.model.z_dim, self.config.model.z_w_h, self.config.model.z_w_h).type_as(image)
        fake_images = self.G(z)

        d_out_fake, _ = self.D(fake_images)
        g_loss = - torch.mean(d_out_fake) 


        fake_images = self.G(z)

        l_real, _ = self.D(image)
        l_fake, _ = self.D(fake_images)

        d_loss = self.hinge_d_loss(l_real, l_fake) 

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_G_Loss': g_loss.item(),
            'Val_D_Loss': d_loss.item()
            }
            self.logger.log_val_metrics(metrics)
                
        return {'generator_loss': g_loss, 'discriminator_loss' : d_loss}
        

    def configure_optimizers(self):
        g_optim = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.config.optimizer.gen_lr, [0.9, 0.9999])
        d_optim = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.config.optimizer.dis_lr, [0.9, 0.9999])
        return [g_optim, d_optim]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'G_Loss', 'D_Loss', 'Val_G_Loss', 'Val_D_Loss']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)
    
    #save config
    logger.log_hyperparams(config, needs_save)

    #set callbacks
    checkpoint_callback = ModelSaver(
        limit_num=config.save.n_saved,
        save_interval=config.save.save_epoch_interval,
        monitor=None,
        dirpath=logger.log_dir,
        filename='ckpt-{epoch:04d}',
        save_top_k=-1,
        save_last=False
    )

    #time per epoch
    timer = Time(config)

    dm = MNISTDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        gpus=1,
        sync_batchnorm=config.training.sync_batchnorm,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = Anogan(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = Anogan.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)
