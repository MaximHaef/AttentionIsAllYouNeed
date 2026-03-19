from torch import nn
import torch
import math
class DotProdAttention(nn.Module):
    def __init__(self, k_dims):
        super.__init__()
        self.k_dims = k_dims
    def forward(self, q, k, v):
        attention = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.k_dims)
        attention = torch.softmax(attention, -1)
        v = torch.matmul(attention, v)
        return v
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, k_dim, v_dim, nheads):
        super.__init__()
        self.model_dim = model_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.nheads = nheads
        self.w_q = nn.Linear(model_dim, self.k_dim*self.nheads)
        self.w_k = nn.Linear(model_dim, self.k_dim*self.nheads)
        self.w_v = nn.Linear(model_dim, self.v_dim*self.nheads) 
        self.w_o = nn.Linear(self.v_dim*self.nheads, model_dim)
        self.attention = DotProdAttention(self.k_dim)
    def forward(self, x):
        q = self.w_q(x).view(x.shape[0], x.shape[1], self.nheads, self.k_dim).transpose(1,2) 
                          
        k = self.w_k(x).view(x.shape[0], x.shape[1], self.nheads, self.k_dim).transpose(1,2) 
        v = self.w_v(x).view(x.shape[0], x.shape[1], self.nheads, self.v_dim).transpose(1,2) 
        out = self.attention(q,k,v)
        out = out.transpose(1,2)
        out = out.reshape(out.shape[0], out.shape[1], self.v_dim*self.nheads)
        out = self.w_o(out)
        return out
        