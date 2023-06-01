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
    def __init__(self, rating_vals, u_feat_shape, i_feat_shape, msg_size, out_feature_size,  dropout_rate=0.0, agg="stack", agg_act=None, out_act=None, share_user_item_param=False, device=None):
        super(GCEncoder, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_size, out_feature_size)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_size, out_feature_size)

        if agg == "stack":
            assert msg_size % len(rating_vals) == 0, "msg unit size must be divisible by rating size to keep dimensionality"
            msg_size = msg_size // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()

        subGraphConv = {}

        for rating in rating_vals:
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            if share_user_item_param and u_feat_shape == i_feat_shape:
                self.W_r[rating] = nn.Parameter(torch.randn(u_feat_shape, msg_size))
                self.W_r["rev=%s" % rating] = self.W_r[rating]
                subGraphConv[rating] = GCMCGraphConv(
                    u_feat_shape, msg_size, apply_weight=False, device=device, dropout_rate=dropout_rate,
                )
                subGraphConv[rev_rating] = GCMCGraphConv(
                    u_feat_shape, msg_size, apply_weight=False, device=device, dropout_rate=dropout_rate
                )
            else:
                self.W_r = None
                subGraphConv[rating] = GCMCGraphConv(
                    u_feat_shape, msg_size, apply_weight=True, device=device, dropout_rate=dropout_rate
                )
                subGraphConv[rev_rating] = GCMCGraphConv(
                    i_feat_shape = msg_size, apply_weight=True, device=device, dropout_rate=dropout_rate
                )
        
        self.conv = dglnn.HeteroGraphConv(subGraphConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        assert device == self.device, "device must be the same"
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param == False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)


    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, u_feat=None, i_feat=None):
        in_feats = {"user", u_feat, "item", i_feat}
        mod_args = {}

        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = "rev-%s" % rating
            mod_args[rating] = (
                self.W_r[rating] if self.W_r is not None else None,
            )
            mod_args[rev_rating] = (
                self.W_r[rev_rating] if self.W_r is not None else None,
            )
        
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        u_feat = out_feats["user"]
        i_feat = out_feats["item"]
        u_feat = u_feat.view(u_feat.shape[0], -1)
        i_feat = i_feat.view(i_feat.shape[0], -1)

        u_feat = self.agg_act(u_feat)
        i_feat = self.agg_act(i_feat)
        u_feat = self.dropout(u_feat)
        i_feat = self.dropout(i_feat)
        u_feat = self.ufc(u_feat)
        i_feat = self.ufc(i_feat)
        return self.out_act(u_feat), self.out_act(i_feat)

        pass

class BiDecoder(nn.Module):
    def __init__(self, in_units, num_classes, num_basis=2, dropout_rate=0.0):
        pass


        




        

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
    