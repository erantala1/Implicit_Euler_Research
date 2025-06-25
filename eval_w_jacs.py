import numpy as np
#import scipy.io
import torch
import os
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
#import netCDF4 as nc
#from prettytable import PrettyTable
#from count_trainable_params import count_parameters    
import pickle
#import matplotlib.pyplot as plt
from nn_FNO import FNO1d
#from nn_MLP import MLP_Net
from nn_step_methods import *

skip_factor = 0 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/erantala/project_runs/code/outputs' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/erantala/project_runs/chkpt_FNO_Eulerstep_implicit_lead1_epoch38.pt"
print(net_file_name)
#change this to use a different network

step_func = Implicit_Euler_step#this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

eval_output_name = 'KS_pred_Implicit_Euler_step_FNO_jacs_for_1k'  # what to name the output file, .mat ending not needed
print(eval_output_name)

with open("/glade/derecho/scratch/erantala/project_runs/KS_1024.pkl", 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN=15000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024 

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])
print(label_test_torch.shape)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters

input_size = 1024 
hidden_layer_size = 2000 
output_size = 1024

modes = 256 # number of Fourier modes to multiply
width = 256  # input and output chasnnels to the FNO layer


my_net = FNO1d(modes, width, time_future, time_history)
# my_net = MLP_Net(input_size, hidden_layer_size, output_size)

my_net.load_state_dict(torch.load(net_file_name))
my_net.cuda()

#step_method = step_func(my_net, device, time_step)

num_iters = 10
step_method = step_func(my_net, device, num_iters, time_step)  #for implicit methods

# M = int(np.floor(99998/lead))
M = label_test_torch.shape[0] - 1
M = 100
net_pred = np.zeros([M,np.size(label_test,1)])
print(M)
print('Model loaded')

noise_var = 0

print('Noise number: ', noise_var)

noised_input = (noise_var)*torch.randn(1,1024).cuda()
noised_input = label_test_torch[0,:].cuda() + noised_input
ygrad = torch.zeros([M,num_iters,input_size, input_size]) #added num_iters dimension 
ygrad_truth = torch.zeros([M,num_iters,input_size, input_size])

print(noised_input.size())

for k in range(0,M):
 
    if (k==0):

        net_output = step_method(torch.reshape(noised_input,(1,input_size,1)))
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        ygrad[k] = step_method.iter_jacs
        print(sum(sum(abs(net_pred))))
        temp_mat = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))) #Use these for FNO
        ygrad[k] = step_method.iter_jacs # stores whole stack of jacobians: one per iteration
        ygrad_truth[k] = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))).reshape(1,input_size, input_size)

    else:

        net_output = step_method(torch.reshape(torch.from_numpy(net_pred[k-1,:]),(1,input_size,1)).float().cuda()) 
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        temp_mat = torch.autograd.functional.jacobian(step_method, net_output) #Use these for FNO
        ygrad[k] = step_method.iter_jacs
        ygrad_truth[k] = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))).reshape(1,input_size, input_size)

    if k%10==0:
        print(k) 
       
print('Eval Finished')

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

#this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes
truth_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)
net_pred_fspec_x = np.zeros(np.shape(net_pred[:,:]), dtype=complex)

for n in range(np.shape(net_pred)[0]):
    truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test[n,:])) 
    net_pred_fspec_x[n,:] = np.abs(np.fft.fft(net_pred[n,:])) 

# calculate time derivative using 1st order finite diff
truth_dt = np.diff(label_test, n=1, axis=0)
net_pred_dt = np.diff(net_pred, n=1, axis=0)

# calculate fourier spectrum of time derivative along a single timestep
truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
net_pred_fspec_dt = np.zeros(np.shape(net_pred_dt[:,:]), dtype=complex)

for n in range(np.shape(net_pred_dt)[0]):
    truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    net_pred_fspec_dt[n,:] = np.abs(np.fft.fft(net_pred_dt[n,:])) 


ygrad = ygrad.detach().cpu().numpy()

def RMSE(y_hat, y_true):
    return np.sqrt(np.mean((y_hat - y_true)**2, axis=1, keepdims=True)) 

def calc_save_chunk(net_pred_chunk, label_test_chunk, chunk_num, ygrad_chunk, ygrad_truth_chunk):
    pred_RMSE = np.zeros([net_pred_chunk.shape[0]])
    # truth_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:]), dtype=complex)
    net_pred_chunk_fspec_x = np.zeros(np.shape(net_pred_chunk[:,:]), dtype=complex)
    # truth_dt = np.diff(label_test_chunk, n=1, axis=0)
    # net_pred_chunk_dt = np.diff(net_pred_chunk, n=1, axis=0)
    # truth_fspec_dt = np.zeros(np.shape(truth_dt[:,:]), dtype=complex)
    # net_pred_chunk_fspec_dt = np.zeros(np.shape(net_pred_chunk_dt[:,:,:]), dtype=complex)

    #this is the fourier spectrum across a single timestep, output has rows as timestep and columns as modes

    pred_RMSE = RMSE(net_pred_chunk, label_test_chunk[0:net_pred_chunk.shape[0]]).reshape(-1)

    # for n in range(np.shape(net_pred_chunk)[0]):
    #     # truth_fspec_x[n,:] = np.abs(np.fft.fft(label_test_chunk[n,:])) 
    #     net_pred_chunk_fspec_x[n ,:] = np.abs(np.fft.fft(net_pred_chunk[n, :])) 

    # for n in range(np.shape(net_pred_chunk_dt)[0]):
    #     # truth_fspec_dt[n,:] = np.abs(np.fft.fft(truth_dt[n,:])) 
    #     net_pred_chunk_fspec_dt[n, ens, :] = np.abs(np.fft.fft(net_pred_chunk_dt[n, ens, :])) 

    net_pred_chunk_fspec_x = np.abs(np.fft.fft(net_pred_chunk[:], axis=1)) 
    print('Calculation Finished')

    matfiledata_output = {}
    matfiledata_output[u'prediction'] = net_pred_chunk
    # matfiledata_output[u'Truth'] = label_test_chunk
    matfiledata_output[u'RMSE'] = pred_RMSE
    # matfiledata_output[u'Truth_FFT_x'] = truth_fspec_x
    matfiledata_output[u'pred_FFT_x'] = net_pred_chunk_fspec_x
    # matfiledata_output[u'Truth_FFT_dt'] = truth_fspec_dt
    # matfiledata_output[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt
    matfiledata_output[u'Jacobians'] = ygrad_chunk
    matfiledata_output[u'Jacobians_truth'] = ygrad_truth_chunk


    print('First save done')
    np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_chunk_'+str(chunk_num), matfiledata_output)


    print('Saved main file')

    if skip_factor!=0: #check if not == 0
        matfiledata_output_skip = {}
        matfiledata_output_skip[u'prediction'] = net_pred_chunk[0::skip_factor,:]
        # matfiledata_output_skip[u'Truth'] = label_test_chunk[0::skip_factor,:]
        matfiledata_output_skip[u'RMSE'] = pred_RMSE[0::skip_factor,:]
        # matfiledata_output_skip[u'Truth_FFT_x'] = truth_fspec_x[0::skip_factor,:]
        matfiledata_output_skip[u'pred_FFT_x'] = net_pred_chunk_fspec_x[0::skip_factor,:]
        # matfiledata_output_skip[u'Truth_FFT_dt'] = truth_fspec_dt[0::skip_factor,:]
        # matfiledata_output_skip[u'pred_FFT_dt'] = net_pred_chunk_fspec_dt[0::skip_factor,:,:]
        matfiledata_output_skip[u'Jacobians'] = ygrad_chunk[0::skip_factor,:]
        matfiledata_output_skip[u'Jacobians_truth'] = ygrad_truth_chunk[0::skip_factor,:]

        
        np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_skip'+str(skip_factor)+'_chunk_'+str(chunk_num), matfiledata_output_skip)



if not os.path.exists(path_outputs+'/'+eval_output_name+'/'):
    os.makedirs(path_outputs+'/'+eval_output_name+'/')
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' created.")
else:
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' already exists.")

prev_ind = 0
chunk_count = 0
num_chunks = 100
for chunk in np.array_split(net_pred, num_chunks):
    current_ind = prev_ind + chunk.shape[0]
    calc_save_chunk(chunk, label_test[prev_ind:current_ind], chunk_count, ygrad[prev_ind:current_ind], ygrad_truth[prev_ind:current_ind])
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')