import torch
import torch.nn as nn
from collections import OrderedDict
from nn_FNO import FNO1d
from torch.func import functional_call


class HyperNetwork(nn.Module):
    def __init__(self, in_dim, hyper_hidden, network):
        super().__init__()
        self.fno = network
        self.names = []
        self.shapes = []
        self.is_complex = []
        self.hyper_layers = nn.ModuleList()
        self.params = OrderedDict()


        #for each parameter there will be a simple two linear layer mlp
        #keep track of whether param is complex, output dim is doubled
        for name, param in self.fno.named_parameters():
            self.names.append(name)
            self.shapes.append(param.shape)
            self.is_complex.append(param.is_complex())
            out_dim = param.numel()
            if param.is_complex():
                out_dim *= 2
            mlp = nn.Sequential(
                nn.Linear(in_dim, hyper_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hyper_hidden,out_dim)
                )
            self.hyper_layers.append(mlp)
            self.params[name] = param.detach().clone()


    #forward pass calls FNO1d with new params and input state u
    #when params are complex split mlp vector into the two real and imag chunks then convert back to torch.complex
    def forward(self, u_0, u_1):
        vec_u = u_0.reshape(1,-1)
        for name, param_shape, mlp, cplx in zip(self.names, self.shapes, self.hyper_layers, self.is_complex):
            batch_param_shape = (-1,) + param_shape
            flat = mlp(vec_u)[0]
            if cplx:
                real, imag  = flat.chunk(2,dim=0)
                update = torch.complex(real,imag).view(batch_param_shape)
            else:
                update = flat.view(batch_param_shape)
            self.params[name] = self.params[name] + update
        return functional_call(self.fno, self.params, (u_1,)) 