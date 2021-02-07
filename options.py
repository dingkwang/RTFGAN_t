import argparse
import os

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train with pytorch')
        self.parser.add_argument('--batch_size', '-b', type=int, default=8) 
        self.parser.add_argument('--lr_start', '-ls', type=float, default=5e-6)
        self.parser.add_argument('--gpu', '-g', type=int, default=0)
        self.parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
        self.parser.add_argument('--epoch_max', '-em', type=int, default=500) # please stop training mannully 
        self.parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
        self.parser.add_argument('--num_workers', '-j', type=int, default=8)
        self.parser.add_argument('--n_class', '-nc', type=int, default=9)
        self.parser.add_argument('--D_max_epochs', type=int, default=5)
        self.parser.add_argument('--data_dir', '-dr', type=str, default='/blue/eel6935/share/GAN/RTFNet/dataset/')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
