from torch import nn
import torch
import math
class DotProdAttention(nn.Module):
    def __init__(self, k_dim):
        super().__init__()
        self.k_dim = k_dim
    def forward(self, q, k, v, masked = False):
        attention = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.k_dim)
        if masked:
            mask = torch.tril(torch.ones(attention.shape, device=attention.device)) 
            attention = attention.masked_fill(mask==0, -float('inf'))
        attention = torch.softmax(attention, -1)
        v = torch.matmul(attention, v)
        return v
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, k_dim, v_dim, nheads, masked = False):
        super().__init__()
        self.masked = masked
        self.model_dim = model_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.nheads = nheads
        self.w_q = nn.Linear(model_dim, self.k_dim*self.nheads)
        self.w_k = nn.Linear(model_dim, self.k_dim*self.nheads)
        self.w_v = nn.Linear(model_dim, self.v_dim*self.nheads) 
        self.w_o = nn.Linear(self.v_dim*self.nheads, model_dim)
        self.attention = DotProdAttention(self.k_dim)
    def forward(self, x, context = None):
        if context == None:
            k = self.w_k(x).view(x.shape[0], x.shape[1], self.nheads, self.k_dim).transpose(1,2) 
            v = self.w_v(x).view(x.shape[0], x.shape[1], self.nheads, self.v_dim).transpose(1,2) 
        else:
            k = self.w_k(context).view(context.shape[0], context.shape[1], self.nheads, self.k_dim).transpose(1,2) 
            v = self.w_v(context).view(context.shape[0], context.shape[1], self.nheads, self.v_dim).transpose(1,2) 
            
        q = self.w_q(x).view(x.shape[0], x.shape[1], self.nheads, self.k_dim).transpose(1,2) 
                          
        out = self.attention(q,k,v, self.masked)
        out = out.transpose(1,2)
        out = out.reshape(out.shape[0], out.shape[1], self.v_dim*self.nheads)
        out = self.w_o(out)
        return out
        