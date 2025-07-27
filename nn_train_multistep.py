import numpy as np
import torch
import os
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
#from count_trainable_params import count_parameters
import pickle
from nn_FNO import FNO1d
from nn_step_methods import *
#from nn_spectral_loss import spectral_loss
#from nn_jacobian_loss import *
from hyper_fno import HyperNetwork

time_step = 1e-1
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

net_name = 'Hyper_FNO_ImplicitEulerstep_lead'+str(lead)+'_train_multistep'
print(net_name)

chkpts_path_outputs = '/glade/derecho/scratch/erantala/project_runs/model_chkpts'
net_chkpt_path = '/glade/derecho/scratch/erantala/project_runs/model_chkpts/'+str(net_name)+'/'

starting_epoch = 0
print('Starting epoch '+str(starting_epoch))

if not os.path.exists(net_chkpt_path):
    os.makedirs(net_chkpt_path)
    print(f"Folder '{net_chkpt_path}' created.")
else:
    print(f"Folder '{net_chkpt_path}' already exists.")


with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024
hidden_layer_size = 2000


epochs = 60
# batch_size = 50
batch_size = 100
batch_time = 2
print('Batch size ', batch_size)
wavenum_init = 100
lamda_reg = 5
evalN = 10000
batch_time_test = 20
print('Batch time test: '+str(batch_time_test))

def Dataloader(data,batch_size,batch_time, key):
    time_chunks = []
    for i in range(data.shape[0] - batch_time*lead):
        time_chunks.append(data[i:i+batch_time*lead:lead])
    extra = len(time_chunks) % batch_size
    if extra==0:
        time_chunks = np.array(time_chunks)
    else:
        time_chunks = np.array(time_chunks[:-extra])
    rng = np.random.default_rng(key)
    split = rng.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split



device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 256  # input and output channels to the FNO layer
in_dim = input_size
hyper_hidden = 256
learning_rate = 1e-5
lr_decay = 0.4

mynet = FNO1d(modes, width, 1, 1).cuda()
my_hypernet = HyperNetwork(in_dim, hyper_hidden, mynet).cuda()
# mynet.load_state_dict(torch.load(net_file_path))
# print('state dict loaded')

step_net = Switch_Euler_step(my_hypernet, device, time_step)

#count_parameters(mynet)

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
optimizer_hyper = optim.AdamW(my_hypernet.parameters(), lr = learning_rate/10)
scheduler_hyper = optim.lr_scheduler.ExponentialLR(optimizer_hyper, 0.95)

rng = np.random.default_rng()
key = rng.integers(100, size=1)
train_data = Dataloader(data[:,0:trainN+lead].T, batch_size = batch_size, batch_time = batch_time, key=key)
train_data = torch.from_numpy(train_data).float()

rng = np.random.default_rng()
key = rng.integers(100, size=1)
test_data = Dataloader(data[:,trainN+lead:].T, batch_size = batch_size, batch_time = batch_time_test, key=key)
test_data = torch.from_numpy(test_data).float()

class Loss_Multistep(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func

    def forward(self, batch):        
        x_i = self.model.implicit_forward(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        #batch_time = 2 so loop never executes
        for i in range(2, self.batch_time):
            x_i = self.model.implicit_forward(x_i.detach())
            loss += self.loss_func(x_i, batch[:,i])
        return loss


loss_fn = nn.MSELoss(reduction='mean')  #for basic loss func
loss_func = lambda e: torch.linalg.norm(e, dim=1).mean(0) 

loss_net_test = Loss_Multistep(step_net, batch_time_test, loss_fn)


torch.set_printoptions(precision=10)
best_loss = 1e5
for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    for n in range(train_data.shape[0]):
        batch = train_data[n].unsqueeze(-1).to(device)
        optimizer.zero_grad()
        optimizer_hyper.zero_grad()
        loss = loss_net_test(batch)

        loss.backward()
        
        optimizer.step()
        optimizer_hyper.step()
        running_loss += loss.detach().item()
        # print(loss)

    net_loss = (running_loss/(train_data.shape[0]))
    key = np.random.randint(len(test_data))
    with torch.no_grad():
        test_loss = loss_net_test(test_data[key].unsqueeze(-1).to(device))
    scheduler.step()
    scheduler_hyper.step()
    print(f'Epoch : {ep}, Train Loss : {net_loss/(batch_time-1)}, Test Loss : {test_loss/(batch_time_test-1)}')
    print('Learning rate', scheduler._last_lr)
    
    if best_loss > test_loss:
        print('Saved!!!')
        torch.save(mynet.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        print('Checkpoint updated')
        print(chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        best_loss = test_loss

    if ep % 10 == 0:
        print(chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')
        torch.save(mynet.state_dict(), chkpts_path_outputs+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')

torch.save(mynet.state_dict(), net_chkpt_path+'chkpt_'+net_name+'_final.pt')
torch.set_printoptions(precision=4)
print("Model Saved")