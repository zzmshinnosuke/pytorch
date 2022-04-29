#!/usr/bin/env python333
# -*- coding: utf-8 -*-
# Created on 2021-06-06 16:20:27
# @Author: zzm

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch.nn.functional as F

learning_rate = 0.01
momentum = 0.5

class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
#         self.l1 = torch.nn.Linear(28 * 28, 10)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
#         return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        
        preds = torch.argmax(self(x), dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.02)
        return torch.optim.SGD(self.parameters(), lr=learning_rate,
                      momentum=momentum)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        return acc
    def test_epoch_end(self, outputs):
        final_values = 0
        for dataloader_outputs in outputs:
            final_values+=dataloader_outputs.cpu().numpy()

        print('final_metric', final_values/len(outputs))
    
    

# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST("../", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32, num_workers=5)

test_ds = MNIST("../", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=32, num_workers=5)


# Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=2, progress_bar_refresh_rate=20)

# Train the model 
trainer.fit(mnist_model, train_loader, test_loader)

trainer.test(mnist_model, test_loader)




