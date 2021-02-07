# GOAL:
# reconstruct thermal images from rgb images using Wasserstein GAN

# Wasserstein tutorial:
# https://github.com/mickypaganini/GAN_tutorial/blob/master/GANtutorial_PyTorch_FashionMNIST.ipynb

# Implementation by Brevin Tilmon for EEL6935 Advanced Robot Perception Spring 2021

import os, time, datetime, stat, shutil, sys
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image
from PIL import Image
import torchvision.models as models
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
import logging
import logging.config

from model import RTFNet
from networks import encoder, unet
from util.MF_dataset import MF_dataset
from ganmodels import Generator, Discriminator
from options import Options

class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.models = {}
        
        self.models["D"] = Discriminator().cuda(self.opts.gpu)#encoder.ResnetEncoder().cuda(self.opts.gpu)
        self.models["G"] = unet.Model().cuda(self.opts.gpu)
        self.G_optim = optim.RMSprop(self.models["G"].parameters(), self.opts.lr_start)
        self.D_optim = optim.RMSprop(self.models["D"].parameters(), self.opts.lr_start)

        self.D_epochs = 0
        self.augmentation_methods = [
            torchvision.transforms.Grayscale()
            #RandomFlip(prob=0.5),
            #RandomCrop(crop_rate=0.1, prob=1.0),
        ]

        if os.path.exists("./ganruns"):
            shutil.rmtree("./ganruns")

        weight_dir = os.path.join("./ganruns")
        os.makedirs(weight_dir)
        self.writer = SummaryWriter("./ganruns")

        print('training on GPU #%d with pytorch' % (self.opts.gpu))
        print('from epoch %d / %s' % (self.opts.epoch_from, self.opts.epoch_max))
        print('weight will be saved in: %s' % weight_dir)

        train_dataset = MF_dataset(data_dir=self.opts.data_dir, 
                                   split='train', 
                                   transform=self.augmentation_methods)

        self.train_loader  = DataLoader(
            dataset     = train_dataset,
            batch_size  = self.opts.batch_size,
            shuffle     = True,
            num_workers = self.opts.num_workers,
            pin_memory  = True,
            drop_last   = False
        )

        self.n_batches = len(train_dataset) // self.opts.batch_size
        print(self.n_batches, len(train_dataset), self.opts.batch_size)

    def save_model(self, state_dict, model, name):
        path = os.path.join('./ganruns', 'dcgan_{}_{}.pth'.format(name, self.epo))
        torch.save(state_dict, path)

    def train(self):
        for self.epo in range(self.opts.epoch_from, self.opts.epoch_max):
            print('\ntrain epo #%s begin...' % (self.epo))
            self.D_epochs = 0
            self.D_loss_list = []
            self.G_loss_list = []
            self.err_real_list = []
            self.err_fake_list = []
            self.process_batch()
            
            '''
            if (self.epo % 10 == 0):
                    self.save_model({
                        'epoch': self.epo + 1,
                        'state_dict': self.models["G"].state_dict(),
                        'optimizer': self.G_optim.state_dict()
                    }, self.models["G"], 'generator')
                    self.save_model({
                        'epoch': self.epo + 1,
                        'state_dict': self.models["D"].state_dict(),
                        'optimizer': self.D_optim.state_dict()
                    }, self.models["D"], 'discriminator')
            '''

        


    def process_batch(self):
        self.models["G"].train()
        self.models["D"].train()

        for it, (rgb, thermal, names) in enumerate(self.train_loader):
            rgb = rgb.cuda(self.opts.gpu)
            thermal = thermal.cuda(self.opts.gpu)

            ###########################
            # FIRST TRAIN DISCRIMINATOR 
            ###########################
        
            if self.D_epochs < self.opts.D_max_epochs:
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()

                # clamp network weights
                for p in self.models["D"].parameters(): 
                    p.data.clamp_(-0.01, 0.01)
            
                # thermal errors
                #print(self.models["D"](thermal).shape)
                err_real = torch.mean(self.models["D"](thermal))

                # fake images 
                fake_images = self.models["G"](rgb)
                
                # rgb errors
                err_fake = torch.mean(self.models["D"](fake_images))

                # minimizing err_real - err_fake is maximizing err_fake - err_real
                D_loss = err_fake - err_real
                D_loss.backward()
                self.D_optim.step()
                self.D_epochs += 1
                self.D_loss_list.append(D_loss.cpu().data)
                self.err_fake_list.append(err_fake.cpu().data)
                self.err_real_list.append(err_real.cpu().data)

            ######################
            # THEN TRAIN GENERATOR
            ######################
            else:
                self.D_optim.zero_grad()
                self.G_optim.zero_grad()
                
                fake_images = self.models["G"](rgb)
                outputs = self.models["D"](fake_images)
                
                G_loss = -torch.mean(outputs)
                G_loss.backward()
                self.G_optim.step()
                self.D_epochs = 0
                self.G_loss_list.append(G_loss.cpu().data)
                

        
        save_image(thermal.data, './data/wgan_real_images-%0.3d.png' %(self.epo + 1))
        save_image(fake_images.data, './data/wgan_fake_images-%0.3d.png' %(self.epo + 1))
        save_image(rgb.data, './data/RGB-%0.3d.png' %(self.epo + 1))

        print('Epoch [%d/%d], d_loss: %.4f, '
              'g_loss: %.4f, Mean D(x): %.2f, Mean D(G(z)): %.2f' 
              %(self.epo,
                self.opts.epoch_max,
                np.array(self.D_loss_list).mean(),
                np.array(self.G_loss_list).mean(),
                np.array(self.err_real_list).mean(),
                np.array(self.err_fake_list).mean())
        )


             
if __name__ == '__main__':
    options = Options()
    opts = options.parse()
   
    torch.cuda.set_device(opts.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    trainer = Trainer(opts)
    trainer.train()
