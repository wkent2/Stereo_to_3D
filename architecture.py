import torch
import torch.nn as nn
from torch.nn import MaxPool2d,AvgPool2d,LeakyReLU,Conv2d,Module, Linear, Sequential,ReLU, LazyLinear,Softmax
import torch.optim as optim
from utils import *

def get_int(string,index):
    '''
    Extracts integer number from string.
    Starts looking at index
    '''

    while index < len(string) and string[index].isdigit()  :
        index += 1
    num = int(string[1:index])

    return num,index
    

def parse_string(string):
    '''
    This function takes in a string and determines what architecture layer
    it is coding for and extracts necessary parameters.

    Types;
    'c' : Conv2d layer
    
    
    '''

    # Check layer type
    if string[0] == 'c':
        layer = "Conv2d"
    elif string[0] == 'a':
        layer = "AvgPool2d"
    elif string[0] == 'm':
        layer = "MaxPool2d"
    elif string[0] == 'l' or string[0] == 'd':
        layer = "Linear"

    nfilter, index = get_int(string,1)


    options = string[index:]
    # Kernel size
    if 'k' in options:
        k = int(options[options.index('k')+1])
    else:
        k=3
    #Let's see if stride was specified
    if 's' in options:
        s = int(options[options.index('s')+1])
    else:
        s=1
    #Let's see if padding was specified
    if 'p' in options:
        p = options[options.index('k')+1]
    else:
        p = 1

    return [layer,nfilter,k,s,p]

def get_convolution_dims(layer_params):
    '''
    Adds input and ouput dimensions for each convolution
    '''
    first = True
    for i in range(len(layer_params)):
        if layer_params[i][0] == "Conv2d":
            # Check first 
            if first:
                layer_params[i][1] = [1,layer_params[i][1]]
                first = False
            else:
                # Find previous Conv2d layer
                index = i - 1
                while layer_params[index][0] != "Conv2d":
                    index -=1
                # Set input channels to be output channels of last Conv2d layer
                layer_params[i][1] = [layer_params[index][1][1],layer_params[i][1]]
    return layer_params

def get_pool_dim(in_dim,layer_params):

    H_in, W_in = in_dim[0], in_dim[1]

    name,nfilter,k,s,p = layer_params

    H_out = ((H_in+2*p-k)/s)+1
    W_out = ((W_in+2*p-k)/s)+1

    return (int(H_out),int(W_out))

def get_conv_dim(in_dim,layer_params):

    H_in, W_in = in_dim[0], in_dim[1]

    name,nfilter,k,s,p = layer_params

    H_out = ((H_in+2*p-k)/s)+1
    W_out = ((W_in+2*p-k)/s)+1

    return (int(H_out),int(W_out))

            
def parse_arch(arch_string):
    '''
    Parses input archictecture string into layer parameters
    '''
    arch_string = arch_string.split(',')
    layer_params = []
    names = []
    for i in range(len(arch_string)):
        layer_params.append(parse_string(arch_string[i]))
        names.append(str(i))

    return layer_params,names
    
def get_linear_size(layer_params,imsize):
    '''
    Calculates the flattened inputs size for first linear layer
    '''
    # Find params for all preceding layers
    
    index = 0
    # Find the first linear layer
    for i in range(len(layer_params)):
        if layer_params[i][0]=='Linear':
            index=i
            break

    # Extract image sizes
    H,W= imsize

    # Iterate through layers to get dimensions
    for i in range(index):
        l_type = layer_params[i][0]
        if  l_type == 'Conv2d':
            H,W = get_conv_dim((H,W),layer_params[i])
        elif (l_type == 'AvgPool2d') or (l_type == 'MaxPool2d'):
            H,W = get_pool_dim((H,W),layer_params[i])

   
    # Comput linear dimension (heigh*width*n_channel)
    linear_size = H*W*layer_params[index-2][1][1]

    return linear_size



class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x,start_dim=1)

class surrogate_arch_mod(Module):   
    def __init__(self,inparams,outparams,arch_string):
        super().__init__()
        
        # Extract layer parameters from arch. string
        layer_params, names = parse_arch(arch_string)
              
        # Initialize sequential layers
        layers = Sequential()
        
        # Initialize first layer

        layers.add_module('Input',Linear(inparams,layer_params[0][1]))

        for i in range(1,len(layer_params)):

            layers.add_module(names[i],Linear(layer_params[i-1][1],layer_params[i][1]))
            layers.add_module(names[i]+'_ReLu',ReLU())

        layers.add_module("Output",Linear(layer_params[i][1],outparams))
 

        self.network = layers
    
    # Defining the forward pass    
    def forward(self, x):
        x = self.network(x)

        x[:, :3] = Softmax(dim=1)(x[:, :3])

        return x




