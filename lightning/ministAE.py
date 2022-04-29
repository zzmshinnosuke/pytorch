#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2022-04-28 17:09:17
# @Author: zzm


import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np

# 一个简单的编码，解码器
class LitAutoEncoder(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )
        
        self.index = 0
    
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
     
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        # 可视化解码后的结果，和原图
        
        img_origin = x.view(28, 28)
        img_origin = img_origin.cpu().numpy()
        Min_origin = np.min(img_origin)
        Max_origin = np.max(img_origin)
        img_origin = (img_origin - Min_origin) / (Max_origin - Min_origin)
        img_origin = img_origin * 255
        img_origin = Image.fromarray(img_origin)
        img_origin = img_origin.convert('L')
        img_origin.save(os.path.join('/home/zzm/tmp/png', "{}_{}_origin.png".format(str(self.index), str(y.item()))))   #图
        
        #将生成后的结果映射到0-255，然后按灰度图保存为图像
        img = x_hat.view(28, 28)
        img = img.cpu().numpy()
        Min = np.min(img)
        Max = np.max(img)
        img = (img - Min) / (Max - Min)
        img = img * 255
        im = Image.fromarray(img)
        im = im.convert('L')
        im.save(os.path.join('/home/zzm/tmp/png', "{}_{}_new.png".format(str(self.index), str(y.item()))))   #图像保存是纯黑色的还是不行
        self.index += 1
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    dataset = MNIST("../", download=True, transform = transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size = 256, num_workers = 32)
    
    test_ds = MNIST("../", train=False, transform = transforms.ToTensor())
    val_loader = DataLoader(test_ds, batch_size = 256, num_workers=5) #没有划分valitation，直接用test代替
    test_loader = DataLoader(test_ds, batch_size = 1, num_workers=5)
    
    autoencoder = LitAutoEncoder()
    
    logger=TensorBoardLogger(save_dir = '~/tmp/logs/pytorch',
                             name = 'ministAE',
                            default_hp_metric = False)
    
    trainer = pl.Trainer(gpus = 1, max_epochs = 200, logger=logger)
    trainer.fit(autoencoder, train_loader, val_loader)
    
    trainer.test(autoencoder, test_loader)
    