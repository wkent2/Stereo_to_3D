import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as L
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint,ModelSummary
from architecture import *
from utils import *
import argparse
SEED = 100

import warnings

def parseargs():
    p = argparse.ArgumentParser(description="Trains ANN to predict 3D microstructure properties from 2D images")
    p.add_argument('micros_3D',help="Path to folder containing 3D micros files")
    p.add_argument('micros_2D',help="Path to .csv file containing microstruture characteristics")
    p.add_argument("-a","-architecture", type=str, default='d128,d64', help="Architecture string")
    p.add_argument("-lf","-loss_fn",type=str,default="MSE")
    p.add_argument("-pin","-params_in",type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,9],help='Column indexes for characteristics to train on. Should be formatted as a list of integers seperated by a single space ' '.')
    p.add_argument("-pout","-params_out",type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,9,10,11,12],help='Column indexes for characteristics to train on. Should be formatted as a list of integers seperated by a single space ' '.')
    p.add_argument("-g","-gamma",type=float,default=0.9)

    args = p.parse_args()
    
    return args

warnings.filterwarnings(
    "ignore",
    message=r"Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found.",
    category=UserWarning,
    module=r"pytorch_lightning\.trainer\.connectors\.logger_connector\.logger_connector"
)

class stereo_to_3D(L.LightningModule):
    def __init__(self,arch_string,target_path,input_path,inparams,outparams,learning_rate,b_size,val_split,checkpoint):
        super().__init__()

        self.target_path = target_path
        self.input_path = input_path
        self.arch_string = arch_string
        self.inparams = inparams
        self.outparams = outparams
        self.learning_rate = learning_rate
        self.b_size = b_size
        self.val_split = val_split
        self.checkpoint = checkpoint
        
        # Initialize mode using Module class
        self.model = surrogate_arch_mod(len(self.inparams),len(self.outparams),self.arch_string)
    
        # Auto=log all hyperparameters to logger
        self.save_hyperparameters()

    def forward(self,x):
        # Forward pass
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        
        y_hat = self(x)
        loss = F.mse_loss(y_hat.flatten(), y.flatten())
        self.log('train_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step to monitor performance
        x, y = batch
        y_hat = self(x)
        
        val_loss = F.mse_loss(y_hat.flatten(), y.flatten())
        self.log('val_loss', val_loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        # Configure the optimizer and optionally a scheduler
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        # Split the dataset for training and validation
        
        full_dataset = SurrogateDataset(self.input_path, self.target_path,self.inparams,self.outparams)

        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size

        generator1 = torch.Generator().manual_seed(SEED)
        # Split train and validation datasets 
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size],generator=generator1)

    def train_dataloader(self):
        # Training data loader
        return DataLoader(self.train_dataset, batch_size=self.b_size,shuffle=True)

    def val_dataloader(self):
        # Validation data loader
        return DataLoader(self.val_dataset, batch_size=self.b_size)


def main(arch_string,target_path,input_path,inparams,outparams,learning_rate,b_size,val_split,checkpoint,n_epochs,do_soft):

    seed_everything(SEED, workers=True)

    # Initialize model and dataloader classes
    model = stereo_to_3D(arch_string,target_path,input_path,inparams,outparams,learning_rate,b_size,val_split,checkpoint)

    # Checkpoints the model for best validation loss
    best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="best_loss-{epoch:03d}-{loss:.6f}",
    )
    
    # Checkpoints the model every 2 epochs
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="epoch",
        mode="max",
        every_n_epochs=n_epochs // 2,
        save_last=True,
        filename="loss-{epoch:03d}",
    )

    # Implement model training
    trainer = Trainer(
        default_root_dir='./results',
        max_epochs=n_epochs,
        # set this to auto when GPU available
        accelerator="mps",
        deterministic=True,
        callbacks=[
            TQDMProgressBar(),
            best_callback,
            checkpoint_callback,
        ],
        # Model weights and parameters are save in checkpoint.
        # Supply this if you want to start from previous traininge
        resume_from_checkpoint=checkpoint,
        detect_anomaly=True,
        log_every_n_steps=1,
    )
    trainer.fit(model)


if __name__ == '__main__':

    # Parse arguments
    args = parseargs()

    arch_string = args.a
    target_path = args.micros_3D
    input_path = args.micros_2D
    inparams = args.pin
    outparams = args.pout
    learning_rate = 1.0e-4
    b_size = 32
    val_split = 0.2
    checkpoint = None
    n_epochs = 1000
    do_soft = False
    fine_tune = False
    loss_func = args.lf
    gamma = args.g

main(arch_string,target_path,input_path,inparams,outparams,learning_rate,b_size,val_split,checkpoint,n_epochs,do_soft)


