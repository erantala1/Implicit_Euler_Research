import torch
import torch.nn as nn
from collections import OrderedDict
from nn_FNO import FNO1d
from torch.func import functional_call
import torch.nn.functional as F
from torch.func import vmap

#mlp class to be used by hypernet
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

#main hypernetwork class
#creates mlp, each with specified number of layers and hidden dim, for each param of outer network
#returns dictionary with updates for main network's params
class HyperNetwork(nn.Module):
    #def __init__(self, num_mlp_layers, in_dim, hyper_hidden_scale, skip_conv, skip_bias, rank, network, device):
    def __init__(self, num_mlp_layers, in_dim, hyper_hidden_scale, which_params, rank, network, device):
        super().__init__()
        self.fno = network.to(device)
        self.device = device
        self.names = []
        self.shapes = []
        self.is_complex = []
        self.hyper_layers = nn.ModuleList()
        self.indices = {}
        self.num_mlp_layers = num_mlp_layers
        self.hyper_hidden_scale = hyper_hidden_scale
        #self.skip_conv = skip_conv
        #self.skip_bias = skip_bias
        self.rank = rank
        self.which_params = which_params

        #each mlp's output dimension will be the sum of the outer network parameter's shape
        for i in range(self.rank):
            for name, param in self.fno.named_parameters():
                if name not in self.which_params:
                    continue
                '''
                if self.skip_conv is True:
                    if name.startswith("conv"):
                        continue
                if self.skip_bias is True:
                    if "bias" in name:
                        continue
                '''
                self.names.append(name)
                self.shapes.append(param.shape)
                self.is_complex.append(param.is_complex())
                out_dim = sum(param.shape)
                if param.is_complex():
                    out_dim *= 2
                hyper_hidden_layer = int(out_dim * hyper_hidden_scale)
                mlp = MLP_net_variable(in_dim, out_dim, hyper_hidden_layer, self.num_mlp_layers, activation=F.gelu, use_act = False, use_dropout=False).to(device)
                self.indices[name+f"{i}"] = len(self.hyper_layers)
                self.hyper_layers.append(mlp)
                print(f"Name:{name}")

    #mlp output dimension is sum of shape
    #output will be split along param shape
    def make_vectors(self, mlp_out, param_shape):
        return list(mlp_out.split(param_shape, dim=1))
    
    #complex case where there are twice as many vectors (real and im)
    def split_complex(self, mlp_out, param_shape):
        real, imag = mlp_out.chunk(2,dim=1)
        update_real = self.make_vectors(real, param_shape)
        update_imag = self.make_vectors(imag, param_shape)
        return update_real, update_imag
    
    #makes rank 1 outer product between each of the mlp output vectors
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

    #update function used for each param
    #split mlp output into vectors, then make outer product update
    #real and imag update for complex param
    def make_update(self, flat, param):
        if param.is_complex():
            real, imag = flat.chunk(2, dim=1)
            upd_r = self.broadcasting(self.make_vectors(real, param.shape), param.shape)
            upd_i = self.broadcasting(self.make_vectors(imag, param.shape), param.shape)
            update = torch.complex(upd_r, upd_i)
        else:
            vecs = self.make_vectors(flat, param.shape)
            update = self.broadcasting(vecs, param.shape)
        return update
    
    #for each param make rank number of updates and add to each parameter
    #returns dictionary of updated parameters to be used in functional_call
    def forward(self, u_0):
        B = u_0.shape[0]
        vec_u = u_0.reshape(B, -1)
        new_params = OrderedDict()

        for name, param in self.fno.named_parameters():              
            '''
            if name.startswith("conv") and self.skip_conv:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
          
            elif "bias" in name and self.skip_bias:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
            '''
            if name not in self.which_params:
                new_params[name] = param.unsqueeze(0).expand(B, *param.shape)
            else:
                for i in range(self.rank):
                    index = self.indices[name+f"{i}"]
                    mlp = self.hyper_layers[index]
                    flat = mlp(vec_u)
                    update = self.make_update(flat, param)
                    if i == 0:
                        new_params[name] = param.unsqueeze(0) + update
                    else:
                        new_params[name] += update

        return new_params
    