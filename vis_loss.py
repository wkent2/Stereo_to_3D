#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:46:06 2024

@author: williamkent
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

def average_values_by_key(list1, list2):
    # Create lists to store the keys and averages
    keys = []
    averages = []
    
    # Create a dictionary to store the sum and count of values for each key
    sum_count = {}
    
    # Iterate over the lists simultaneously
    for value, key in zip(list1, list2):
        if key in sum_count:
            # If the key exists, update the sum and count
            sum_count[key][0] += value
            sum_count[key][1] += 1
        else:


            # If the key does not exist, initialize the sum and count
            sum_count[key] = [value, 1]
    
    # Calculate the average for each key and append to the lists
    for key, (sum_value, count) in sum_count.items():
        keys.append(key)
        averages.append(sum_value / count)
    
    return keys, averages

# Example usage:



DIR_PATH = './results/lightning_logs/version_17'

data = pd.read_csv(os.path.join(DIR_PATH,'metrics.csv'),header=0)

epochs = []
train_loss = []
val_loss = []
train_loss_step = []
epoch_step = []

for i in range(len(data)):
    if math.isnan(data['train_loss_epoch'][i]) != True:
        epochs.append(data['epoch'][i])
        train_loss.append(data['train_loss_epoch'][i])
    if math.isnan(data['val_loss_epoch'][i]) != True:
        val_loss.append(data['val_loss_epoch'][i])
    if math.isnan(data['train_loss_step'][i]) != True:
        train_loss_step.append(data['train_loss_step'][i])
        epoch_step.append(data['epoch'][i])
        

epochs_2,train_loss_2 = average_values_by_key(train_loss_step,epoch_step)
    
    
start_index = 2


plt.close('all')
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_2[start_index:],train_loss_2[start_index:],color='blue',label='Train')
plt.plot(epochs[start_index:],val_loss[start_index:],color='green',label='Validation')
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('MSE',fontsize=12)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_2[start_index:],train_loss_2[start_index:],color='blue',label='Train')
plt.plot(epochs[start_index:],val_loss[start_index:],color='green',label='Validation')
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('MSE',fontsize=12)
plt.yscale('log')
plt.legend()
# plt.ylim(0,0.0005)
plt.tight_layout()

plt.savefig(os.path.join(DIR_PATH,'loss.png'),dpi=300)

# plt.close('all')
# plt.figure(figsize=(5,5))

# plt.subplot(1,1,1)
# plt.plot(epochs_2[start_index:],train_loss_2[start_index:],color='blue',label='Padded Train')
# plt.plot(epochs[start_index:],val_loss[start_index:],'b--',label='Padded Validation')


# DIR_PATH = './results/lightning_logs/version_72'

# data = pd.read_csv(os.path.join(DIR_PATH,'metrics.csv'),header=0)

# epochs = []
# train_loss = []
# val_loss = []
# train_loss_step = []
# epoch_step = []

# for i in range(len(data)):
#     if math.isnan(data['train_loss_epoch'][i]) != True:
#         epochs.append(data['epoch'][i])
#         train_loss.append(data['train_loss_epoch'][i])
#     if math.isnan(data['val_loss_epoch'][i]) != True:
#         val_loss.append(data['val_loss_epoch'][i])
#     if math.isnan(data['train_loss_step'][i]) != True:
#         train_loss_step.append(data['train_loss_step'][i])
#         epoch_step.append(data['epoch'][i])
        

# epochs_2,train_loss_2 = average_values_by_key(train_loss_step,epoch_step)

# plt.plot(epochs_2[start_index:],train_loss_2[start_index:],'g-',label='No Padding Train')
# plt.plot(epochs[start_index:],val_loss[start_index:],'g--',label='No Padding Validation')

# plt.xlabel('Epoch',fontsize=12)
# plt.ylabel('MSE',fontsize=12)
# plt.yscale('log')
# plt.legend()
# # plt.ylim(0,0.0005)
# plt.tight_layout()

# # plt.savefig(os.path.join(DIR_PATH,'loss_comp_47.png'),dpi=300)

