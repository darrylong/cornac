import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.base import DGLError
import torch
import torch.nn as nn


class GCMCGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, apply_weight=True, device=None, dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if apply_weight:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, input, weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat
            cj = graph.srcdata["cj"]
            ci = graph.dstdata["ci"]
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError("Weight parameters are already specified by module. Use the parameter apply_weight=False.")
            else:
                weight = self.weight
            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)
            
            feat = feat * self.dropout(cj)

            graph.srcdata["h"] = feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))

            rst = graph.dstdata["h"]
            rst = rst * ci

        return rst

class GCEncoder(nn.Module):
     def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GCEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        

def dot_or_identity(A, B, device=None):
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B
    