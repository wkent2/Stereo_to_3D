import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as L
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from architecture import *
from utils import *
import pandas as pd
import argparse
import yaml
from stereo_to_3D import stereo_to_3D

def parseargs():
    p = argparse.ArgumentParser(description="Plots results from ANN training")
    p.add_argument('version',type=int,help="training version")
    p.add_argument('-r',type=str,default="./resultsANN",help="folder containing all training results")
    p.add_argument('-b',type=bool,default=True,help="Whether to plot the best or the last checkpoint")
    p.add_argument('-inp',type=str,default=None,help="Option to test different input data")
    p.add_argument('-outp',type=str,default=None,help="Option to plot different output data.")
    p.add_argument('-s',type=int,default=100,help="Random seed")
    p.add_argument('-swap',type=bool,default=False,help="Whether or not to phase swap")

    args = p.parse_args()
    
    return args

def load_hyperparameters(file_path):
    with open(file_path, "r") as file:
        hyperparams = yaml.safe_load(file)  # Load YAML content safely
    return hyperparams

def check_file_exists(filepath,err_msg):
    if not os.path.exists(filepath):
        print(err_msg)
        exit()


def get_checkpoint_path(results_path,best):

    # Construct checkpoints folder path
    ckpt_folder = os.path.join(results_path,'checkpoints')

    # Check that checkpoints folder exists
    check_file_exists(ckpt_folder,"There is no checkpoints folder in the version folder")

    # Checks if using best checkpoint
    if best:
        # Checks the best checkpoint exists
        files = [file for file in os.listdir(ckpt_folder) if file.endswith('.ckpt')]
        checkp = next((f for f in files if f.startswith('best_loss')), None)
        checkp = os.path.join(ckpt_folder,checkp)
        check_file_exists(checkp,"Cannot find the best_loss-.....ckpt checkpoint")
    else:
        # Check that last checkpoint exists
        checkp = os.path.join(ckpt_folder,'last.ckpt')
        check_file_exists(checkp,"Cannot find the last.ckpt checkpoint")

    return checkp

def get_chan(dmap_in):
    if dmap_in == 'None':
        return 3
    elif dmap_in == 'single':
        return 4
    elif dmap_in == 'multi':
        return 6
    else:
        print("Unreccongized dmap input")
        exit()



if __name__ == "__main__":

    # Parse command line arguments
    args = parseargs()

    # Find .yaml file with hyperparameters
    version_str = 'version_'+str(args.version)
    path_to_res = os.path.join(args.r,'lightning_logs',version_str)
    check_file_exists(path_to_res,"Cannot find version folder")

    # Checks if hparams.yaml exists along path
    check_file_exists(os.path.join(path_to_res,'hparams.yaml'),f"Error: There is no hparams.yaml file in the version folder")
    # Get hyperparameters 
    hparams = load_hyperparameters(os.path.join(path_to_res,'hparams.yaml'))

    # Applies random seed to everything
    seed_everything(args.s, workers=True)

    # Sets new input and output 
    if args.inp:
        check_file_exists(args.inp,f"Error: Can't find input file")
        hparams['input_path'] = args.inp
    if args.outp:
        check_file_exists(args.outp,f"Error: Can't find output file")
        hparams['target_path'] = args.outp


    print("Constructing dataset")
    full_dataset = SurrogateDataset(hparams['input_path'], 
                                    hparams['target_path'],
                                    hparams['inparams'],
                                    hparams['outparams'],
                                    args.swap)



    # Applies extra random seed. Not sure why it needs this but data does not 
    # split correctly without it. 
    generator1 = torch.Generator().manual_seed(args.s)

    # Figures out train and validation data split
    val_size = int(hparams['val_split'] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Splits data into train and validation datasets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=generator1)
    
    # Initializes train and validation data loaders
    train_DL = DataLoader(train_dataset, batch_size=hparams['b_size'], num_workers=1,shuffle=True)
    val_DL = DataLoader(val_dataset, batch_size=hparams['b_size'],num_workers=1)

    chk_path = get_checkpoint_path(path_to_res,args.b)

    # Initialize model and dataloader classes
    model = stereo_to_3D.load_from_checkpoint(chk_path)

    # Figures out whether or not to use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prints out model for sanity check
    for name, layer in model.named_children():
        print(f"Layer name: {name}, Layer: {layer}")

    # Sets model to evaluation mode (no gradient computations)
    model.eval()

    # Initializes arrays for storing data. 
    ys_train = []
    yhats_train = []
    ys_val = []
    yhats_val = []

    # Iterates through TRAINING data    
    for batch, (X,y) in enumerate(tqdm(train_DL)):
        y_hat = model(X.to(device))
        ys_train.append(y.detach().cpu().numpy())
        yhats_train.append(y_hat.cpu().detach().numpy())

    # Iterates through VALIDATION data
    for batch, (X,y) in enumerate(tqdm(val_DL)):
        y_hat = model(X.to(device))
        ys_val.append(y.detach().cpu().numpy())
        yhats_val.append(y_hat.detach().cpu().numpy())


    # Concatenates to numpy array 
    ys_train_full = np.concatenate(ys_train)
    yhats_train_full = np.concatenate(yhats_train)
    ys_val_full = np.concatenate(ys_val)
    yhats_val_full = np.concatenate(yhats_val)

    # Saves raw targets and predictions
    np.save(os.path.join(path_to_res,'ys_train.npy'),ys_train_full)
    np.save(os.path.join(path_to_res,'yhats_train.npy'),yhats_train_full)
    np.save(os.path.join(path_to_res,'ys_val.npy'),ys_val_full)
    np.save(os.path.join(path_to_res,'yhats_val.npy'),yhats_val_full)

    # PLOTS RESULTS 
    # This is just to quickly asses if the plotting was done 
    # correctly on Joule. Not final results 

    # Gets keys for the variables
    df = pd.read_csv(hparams['target_path'],header=0,index_col=0)
    names = df.keys()

    print("Plotting results")
    for i in range(len(names)):
        plt.close('all')
        plt.figure(figsize=(10,5))

        plt.suptitle(names[i],fontsize=16)

        fulldata = np.concatenate([ys_train_full,yhats_train_full,
                                    ys_val_full,yhats_val_full])

        bin_w = np.linspace(np.amin(fulldata),np.amax(fulldata),200)

        plt.subplot(1,2,1)
        plt.hist2d(ys_train_full[:,i],yhats_train_full[:,i],bins=bin_w,cmap='Blues',density=True)
        plt.plot(bin_w,bin_w,color='black',linewidth=0.5)
        plt.gca().set_box_aspect(1.0)
        plt.xlabel('Target',fontsize=12)
        plt.ylabel('Predicted',fontsize=12)
        plt.title('Training Data',fontsize=14)

        plt.subplot(1,2,2)
        plt.hist2d(ys_val_full[:,i],yhats_val_full[:,i],bins=bin_w,cmap='Blues',density=True)
        plt.plot(bin_w,bin_w,color='black',linewidth=0.5)
        plt.gca().set_box_aspect(1.0)
        plt.xlabel('Target',fontsize=12)
        plt.ylabel('Predicted',fontsize=12)
        plt.title('Validation Data',fontsize=14)


        savename = names[i]+'.png'
        plt.savefig(os.path.join(path_to_res,savename),dpi=300)






