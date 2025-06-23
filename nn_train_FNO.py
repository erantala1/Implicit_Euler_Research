import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
from count_trainable_params import count_parameters
import pickle
from nn_FNO import FNO1d
from nn_step_methods import *
from nn_spectral_loss import spectral_loss
from nn_jacobian_loss import *

time_step = 5e-2
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

#chkpts_path_outputs = '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'


#net_name = 'FNO_Eulerstep_lead'+str(lead)+''
#print(net_name)

#net_file_path = "/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/FNO_Eulerstep_lead50/chkpt_FNO_Eulerstep_lead50_epoch53.pt"
#print(net_file_path)
starting_epoch = 1
print('Starting epoch '+str(starting_epoch))

# to change from normal loss to spectral loss scroll down 2 right above train for loop

with open("/Users/evanrantala/Downloads/Training_data_pus_2_models/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250])
print(data.shape)
trainN = 100
input_size = 1024
output_size = 1024
hidden_layer_size = 100
input_train_torch = torch.from_numpy(np.transpose(data[:,0:trainN]))
label_train_torch = torch.from_numpy(np.transpose(data[:,lead:lead+trainN]))
du_label_torch = (input_train_torch - label_train_torch)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cpu'  #change to cpu if no cuda available

#model parameters
modes = 20 # number of Fourier modes to multiply
width = 20  # input and output channels to the FNO layer

learning_rate = 1e-3
lr_decay = 0.4

mynet = FNO1d(modes, width, time_future, time_history)

#mynet.load_state_dict(torch.load(net_file_path))
#print('state dict loaded')

step_net = Implicit_Euler_step(mynet, device, time_step)

count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

epochs = 50
batch_size = 25
# batch_size = 100
print('Batch size ', batch_size)
wavenum_init = 100
lamda_reg = 5

loss_fn = nn.MSELoss()  #for basic loss func
loss_fc = spectral_loss #for spectral loss in tendency, also change loss code inside for loop below
torch.set_printoptions(precision=10)

for ep in range(starting_epoch, epochs+1):
    running_loss = 0
    indices = np.random.permutation(torch.arange(trainN))
    for step in range(0,trainN,batch_size):
        batch_indices = indices[step:step + batch_size]
        # indices = np.random.permutation(np.arange(start=step, step=1 ,stop=step+batch_size))
        input_batch, label_batch = input_train_torch[batch_indices], label_train_torch[batch_indices]
        input_batch = torch.reshape(input_batch,(batch_size,input_size,1)).float()
        label_batch = torch.reshape(label_batch,(batch_size,input_size,1)).float()
        # du_label_batch = du_label_torch[indices]
        # du_label_batch = torch.reshape(du_label_batch,(batch_size,input_size,1)).float()
        #pick a random boundary batch
        
        optimizer.zero_grad()
        # outputs = step_net(input_batch)
        
        # loss = loss_fn(outputs, label_batch)  # use this loss function for mse loss

        # outputs_2 = step_func(mynet, outputs, time_step) #use this line and line below for spectral loss
        # loss = loss_fc(outputs, outputs_2, label_batch, du_label_batch, wavenum_init, lamda_reg, time_step)

        loss = jacobian_loss(step_net, input_batch, label_batch)
        #loss = spectral_jacobian_loss(step_net, input_batch, label_batch, 1, 1)
  
        loss.backward()

        optimizer.step()
        running_loss += loss.clone().detach()

    if ep % 1 == 0:
        print('Epoch', ep)
        print ('Train Loss', float(running_loss/int(trainN/batch_size)))
        with torch.no_grad():
            key = np.random.randint(0, trainN, 100)
            temp_loss = F.mse_loss(step_net(input_train_torch[key].reshape(100,input_size,1).float()), label_train_torch[key].reshape(100,input_size,1).float())
            print('One step loss:', float(temp_loss))

        #torch.save(mynet.state_dict(), '/glade/derecho/scratch/cainslie/conrad_net_stability/model_chkpts/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch'+str(ep)+'.pt')
 

#torch.save(mynet.state_dict(), net_name+'.pt')
#torch.set_printoptions(precision=4)
#print("Model Saved")