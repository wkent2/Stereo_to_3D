import os
import shutil
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from time import time

class SurrogateDataset(Dataset):
    def __init__(self, input_path,target_path,in_param,out_param):
        # Filename with target values (3D values)
        self.target_p = target_path
        # Filename with input values (3D)
        self.input_p = input_path
        # Creates array of all training images
        self.input_vals, self.target_vals = create_training_data_array(self.target_p,self.input_p,in_param,out_param)

    def __len__(self):
        return len(self.input_vals)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.input_vals[idx])
        targets = torch.tensor(self.target_vals[idx])

        return inputs, targets



def create_training_data_array(target_p,input_p,in_param,out_param):
    
    # Load input values
    input_df = pd.read_csv(input_p,header=0,index_col=0)

    # Load target values
    target_df = pd.read_csv(target_p,header=0,index_col=0)

    input_data = np.zeros(shape=(len(input_df),len(in_param)),dtype=np.float32)

    target_data = np.zeros(shape=(len(input_df),len(out_param)),dtype=np.float32)

    # Removes file extension from data frame indexes
    input_df.index = df.index.str.replace(r'\.\w+$', '', regex=True)

    # Removes file extension from data frame indexes
    target_df.index = df.index.str.replace(r'\.\w+$', '', regex=True)

    print("Making training dataset")
    for i in tqdm(range(len(input_df))):
        index_of_row = input_df.index[i]
        for j in range(len(in_param)):
            input_data[i,j] = input_df[input_df.keys()[in_param[j]]][index_of_row]
        for j in range(len(out_param)): 
            ind_npy = index_of_row[0:-2]
            target_data[i,j] = target_df[target_df.keys()[out_param[j]]][ind_npy]


    nan_rows_mask = np.isnan(input_data).any(axis=1) + np.isnan(target_data).any(axis=1)
    input_cleaned = input_data[~nan_rows_mask]
    target_cleaned = target_data[~nan_rows_mask]

    print(len(np.where(nan_rows_mask)[0]), 'rows were removed because they contained NAN values')


    print("Shape of input data is",input_cleaned.shape)
    print("Shape of target data is",target_cleaned.shape)


    return input_cleaned,target_cleaned

def create_and_delete_directory(directory_path):
    # Check if directory already exists
    if os.path.exists(directory_path):
        print(f"Directory '{directory_path}' already exists. Deleting it...")
        try:
            shutil.rmtree(directory_path)  # Removes the directory
            print(f"Directory '{directory_path}' deleted successfully.")
        except OSError as e:
            print(f"Error: {directory_path} : {e.strerror}")
    else:
        print(f"Creating directory '{directory_path}'...")
    
    # Create the directory
    try:
        os.makedirs(directory_path)  # Creates the directory
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as e:
        print(f"Error: {directory_path} : {e.strerror}")


def rotations(img):
    augmented = []
    for i in range(1,4):
        augmented.append(torch.rot90(img,i))
    return augmented

def augment_slice(img_original):
    '''
    This function augments training data by applying rotations and flips to 
    the input image. 4 rotations X 2 flips creates increases training data by 8 sizes

    Inputs:
    
    img: 2D torch tensor
        Image to augment
    micro: numpy array
        1D arry containing microstructure parameters

    Returns:
    
    augmented_ims: List of 2D torch tensors
        List of augmented images, not including the original image
    micros: List of numpy arrays
        Corresponding microstructure parameters 
    '''
    
    augmented = []

    # Apply flips
    flipped = flip(img_original)
    augmented.append(flipped)

    # Apply rotations
    for img in [img_original,flipped]:
        rotated = rotations(img)
        for rot in rotated:
            augmented.append(rot)
    
    return augmented


def augment_dataset(img_dir,micro_file):
    '''
    This function augments training data by applying rotations and flips to 
    the input image. 4 rotations X 2 flips creates increases training data by 8 sizes

    Inputs:
    
    img_dir: string
        Path to directory containing npy_files
    micro: string
        Path to microstructure file. Should be .csv ouput from Microstructure Code (Billy Epting's)=

    Returns:
    
    data_dir: string
        Path to folder containing augmented 2D images and corresponding microstructure information
    '''

    # Create directory for agumented data
    data_dir = os.path.join(os.path.dirname(img_dir),'augmented_data')
    create_and_delete_directory(data_dir)

    # Collect image files
    img_files = [file for file in os.listdir(img_dir) if file.endswith('.npy')]
    #Collect Microstructure information
    micro_df = pd.read_csv(micro_file,header=0,index_col=0)

    # Iterate through files for augmentation
    for file in tqdm(img_files):
        # Load image
        img = np.load(os.path.join(img_dir,file))

        # Create augmented images
        augmented_imgs = augment_slice(torch.tensor(img))

        # Save images and microstructure info
        np.save(os.path.join(data_dir,file),img)
        for i in range(len(augmented_imgs)):
            # Save .npy file
            augname = file[0:-4] + '_' + str(i) + '.npy'
            np.save(os.path.join(data_dir,augname),augmented_imgs[i].numpy())
            # Append corresponding microstructure information to 
            micro_df.loc[augname] = micro_df.loc[file].to_list()

    # Save microstructure information

    micro_df.to_csv(os.path.join(data_dir,'augmented_microstructure_info.csv'))

    return data_dir


def to_2D(img_dir,micro_file):
    img_files = [file for file in os.listdir(img_dir) if file.endswith('.npy')]

    #Collect Microstructure information
    micro_df = pd.read_csv(micro_file,header=6,index_col=0)

    print(micro_df.keys())
    
    # Initialize 2D micro_dataframe
    micro_2D = pd.DataFrame(columns=[micro_df.keys()])
    

    # Create directory for agumented data
    data_dir = os.path.join(os.path.dirname(img_dir),'2D_data')
    create_and_delete_directory(data_dir)

    # Get data side length
    test_vol = np.load(os.path.join(img_dir,img_files[0]))
    index = int(test_vol.shape[0]/2) # Should be 50 for side len of 101

    # Iterate through files
    for img in img_files:
        # Load volume
        vol = np.load(os.path.join(img_dir,img))
        
        # Generate filenames
        name_z,name_y,name_x = [img[0:-4]+end+'.npy' for end in ('_z','_y','_x')]

        # Save 2D images
        np.save(os.path.join(data_dir,name_z),vol[index,:,:])
        np.save(os.path.join(data_dir,name_y),vol[:,index,:])
        np.save(os.path.join(data_dir,name_x),vol[:,:,index])

        # Add microstructure information 
        for name in [name_z,name_y,name_x]:
            micro_2D.loc[name] = micro_df.loc[img].to_list()

    # Save microstructure info
    micro_2D.to_csv(os.path.join(data_dir,'2D_micro_params.csv'))

    print("2D extraction finished. Data saved in",data_dir)


def check_test_train_split(split,num_imgs):

    train = int((1-split)*num_imgs) 
    test = int(split*num_imgs)

    while test + train != num_imgs:
        train+=1

    return test,train

def log_process(file_dir, message):
    # Create a log file name based on the process name
    log_file_name = f"training_log.txt"

    # Create or append to the log file
    with open(os.path.join(file_dir,log_file_name), "a") as log_file:
        
        log_file.write(f"{message}\n")




def write_training_params(file_dir,img_dir,micro_path,param_ind,do_trans,learning_rate,b_size,arch_string,model_params,loss_fun,epochs,test_train_split):

    message = "Training images are from " + img_dir
    log_process(file_dir,message)
    message = "Microstrcutre parameters are from " + micro_path
    log_process(file_dir,message)
    message = "Parameter index is " + str(param_ind) + " (" + str(pd.read_csv(micro_path,header=0,index_col=0).keys()[param_ind]) + ")"
    log_process(file_dir,message)
    if do_trans:
        message = "Augmentation applied to data" 
        log_process(file_dir,message)
    message = "Learning rate: " + str(learning_rate)
    log_process(file_dir,message)
    message = "Batch size: " + str(b_size)
    log_process(file_dir,message)
    message = "Architecture string " + arch_string
    log_process(file_dir,message)
    if model_params:
        message = "Initialized with model weights from " + model_params
        log_process(file_dir,message)
    message = "Loss function:" + loss_fun
    log_process(file_dir,message)
    message = "Number of epochs " + str(epochs)
    log_process(file_dir,message)
    message = "Test/train split " + str(test_train_split)
    log_process(file_dir,message)


def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    training_loss = 0
    ys = []
    preds = []
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)
        training_loss += loss.item()

        for y_i in y:
            ys.append(y_i.item())
        for p_i in pred:
            preds.append(p_i.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
       
    return training_loss / num_batches, ys,preds

def test(dataloader, model, loss_fn,epoch,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    ys = []
    preds = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            for y_i in y:
                ys.append(y_i.item())
            for p_i in pred:
                preds.append(p_i.item())

            
            test_loss += loss_fn(pred.squeeze(), y).item()
    test_loss /= num_batches
   
    return test_loss, ys,preds

def plot_test(out_dir,test_y,test_p,epoch):

    plt.close('all')
    plt.hist2d(test_y,test_p,bins=100,range=[[min(test_y), max(test_y)],[min(test_y), max(test_y)]],cmap='BuGn')
    # Generate the line y = x
    x_line = np.linspace(min(test_y), max(test_y), 100)
    y_line = x_line
    plt.plot(x_line, y_line, color='black')
    plt.xlabel('Target Data',fontsize=12)
    plt.ylabel('Predicted Data',fontsize=12)
    title_text = 'Test Data: Epoch '+ str(epoch)
    plt.title(title_text,fontsize=12)
    savename = 'trained_model_'+str(epoch)+'_test.png'
    plt.savefig(os.path.join(out_dir,savename),dpi=300)

def plot_training(out_dir,train_y,train_p,epoch):

    plt.close('all')
    plt.hist2d(train_y,train_p,bins=100,range=[[min(train_y), max(train_y)],[min(train_y), max(train_y)]],cmap='BuGn')
    # Generate the line y = x
    x_line = np.linspace(min(train_y), max(train_y), 100)
    y_line = x_line
    plt.plot(x_line, y_line, color='black')
    plt.xlabel('Target Data',fontsize=12)
    plt.ylabel('Predicted Data',fontsize=12)
    title_text = 'Training Data: Epoch '+ str(epoch)
    plt.title(title_text,fontsize=12)
    savename = 'trained_model_'+str(epoch)+'_training.png'
    plt.savefig(os.path.join(out_dir,savename),dpi=300)

def MAPE_loss(output, target):
    # MAPE loss
    return 100*torch.mean(torch.abs((target - output) / target))

def plot_losses(out_dir,training_l,test_l,ylim=None):
    plt.close('all')
    plt.plot(training_l,'b-',label='Training')
    plt.plot(test_l,'g--',label='Test')
    plt.xlabel('Epoch',fontsize=12)
    plt.ylabel('MSE',fontsize=12) 
    if ylim:
        plt.ylim(0,ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'losses.png'),dpi=300)

def save_best_params(out_dir,model,epoch):
    savename ='model_params_best.pth'
    torch.save(model.state_dict(), os.path.join(out_dir,savename))

def make_output_directory():
    # Folder is named after date and time of run

    now = datetime.now()

    # dd/mm/YY H:M:S
    filepath = './'+now.strftime("%d%m%Y_%H%M%S")
    os.mkdir(filepath)

    return filepath


def save_filenames_to_txt(filenames, filename):
    """
    Save a list of filenames to a .txt file.
    
    :param filenames: List of filenames to save.
    :param filename: The name of the .txt file where filenames will be saved.
    """
    with open(filename, 'w') as file:
        for name in filenames:
            file.write(name + '\n')

def read_filenames_from_txt(filename):
    """
    Read filenames from a .txt file and put them into a list.
    
    :param filename: The name of the .txt file to read from.
    :return: A list of filenames.
    """
    filenames = []
    with open(filename, 'r') as file:
        filenames = [line.strip() for line in file]
    return filenames
