import torch
import torch.nn as nn
from collections import OrderedDict
from nn_FNO import FNO1d
from torch.func import functional_call
import torch.nn.functional as F
from torch.func import vmap
class MLP_net_variable(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation=F.gelu, use_act = True, use_dropout=True):
        super().__init__()
        self.linear_in = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.linear_in.weight)
        self.activation = activation
        self.layers_1 = nn.ModuleList()
        if use_dropout:
            self.drop = nn.ModuleList()
        self.use_drop = use_dropout
        self.num_layers = num_layers
        self.use_act = use_act
        for i in range(0,num_layers): 
            self.layers_1.append(nn.Linear(hidden_dim, hidden_dim))
            torch.nn.init.xavier_normal_(self.layers_1[i].weight)
            if use_dropout:
                self.drop.append(nn.Dropout(p=0.9))
        self.linear_out = nn.Linear(hidden_dim, out_dim)
        torch.nn.init.xavier_normal_(self.linear_out.weight)

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        x_0 = x
        for i in range(0,self.num_layers):
            x = self.activation(self.layers_1[i](x)) 
            if self.use_drop:
                x = self.drop[i](x)
            # x = x + x_0
        x = self.linear_out(x)
        if self.use_act:
            x = self.activation(x)
        return x

class HyperNetwork(nn.Module):
    def __init__(self, in_dim, hyper_hidden, network, device):
        super().__init__()
        self.fno = network.to(device)
        self.device = device
        self.names = []
        self.shapes = []
        self.is_complex = []
        self.hyper_layers = nn.ModuleList()

        #for each parameter there will be a simple two linear layer mlp
        #keep track of whether param is complex, output dim is doubled
        #spectral conv special case
        for name, param in self.fno.named_parameters():
            self.names.append(name)
            self.shapes.append(param.shape)
            #print(f"param shape: {param.shape}")
            self.is_complex.append(param.is_complex())
            out_dim = sum(param.shape)
            if param.is_complex():
                out_dim *= 2
            hyper_hidden_layer = out_dim * 2
            mlp = MLP_net_variable(in_dim, out_dim, hyper_hidden_layer, num_layers=3, activation=F.gelu, use_act = False, use_dropout=False).to(device)
            self.hyper_layers.append(mlp)
            #print("mlp initialized")

    def make_vectors(self, mlp_out, param_shape):
        return list(mlp_out.split(param_shape, dim=1))

    def split_complex(self, mlp_out, param_shape):
        real, imag = mlp_out.chunk(2,dim=1)
        update_real = self.make_vectors(real, param_shape)
        update_imag = self.make_vectors(imag, param_shape)
        return update_real, update_imag
    
    def broadcasting(self, vecs, param_shape):
        #vecs is a list containing each tensor to be used in outer product
        if len(vecs) == 3:
            v1,v2,v3 = vecs
            d1,d2,d3 = param_shape
            v1 = v1.view(-1, d1, 1, 1)
            v2 = v2.view(-1, 1,  d2, 1)
            v3 = v3.view(-1, 1,  1,  d3)
            update = v1 * v2 * v3
        elif len(vecs) == 2:
            v1,v2 = vecs
            d1,d2 = param_shape
            v1 = v1.view(-1, d1, 1)
            v2 = v2.view(-1, 1,  d2)
            update = v1 * v2
        elif len(vecs) == 1:
            update = vecs[0]
        return update

    def batch_functional(self, params, u):
        #print(f"u shape: {u.shape}")
        return functional_call(self.fno, params, (u.unsqueeze(0),),strict=True).squeeze(0)

    def forward(self, u_0, u_1):
        vec_u = u_0.reshape(u_0.shape[0],-1)
        new_params = OrderedDict()
        for name, param_shape, mlp, cplx in zip(self.names, self.shapes, self.hyper_layers, self.is_complex):
            flat = mlp(vec_u)
            if cplx:
                update_real, update_imag = self.split_complex(flat,param_shape)
                update = torch.complex(self.broadcasting(update_real, param_shape), self.broadcasting(update_imag, param_shape))
            else:
                vecs = self.make_vectors(flat,param_shape)
                update = self.broadcasting(vecs, param_shape)
            
            for param_name, parameter in self.fno.named_parameters():
                if param_name == name:
                    break
            new_params[name] = parameter + update
        return torch.vmap(self.batch_functional, in_dims = (0,0))(new_params, u_1)

#new shifts/scales hypernet class
class Modulations(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_modulations, num_layers, device):
        super().__init__()
        self.device = device
        self.in_dim  = in_dim
        self.hidden_dim = hidden_dim
        self.num_modulations = num_modulations
        self.num_layers = num_layers
        self.mlp = mlp = MLP_net_variable(self.in_dim, self.num_modulations, self.hidden_dim, num_layers=self.num_layers, activation=F.gelu, use_act = False, use_dropout=False).to(device)

    def forward(self, u_0):
        vec_u = u_0.reshape(u_0.shape[0],-1)
        out = self.mlp(vec_u)
        return out
        #output shape of [B,num_modulations]

