from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import block
from .pairwise import Pairwise
from .graph import GraphModule

class iteregCell(nn.Module):
    def __init__(self,
                 fusion={},
                 relations={},
                 control={},
                 graph={}):

        self.fusion = fusion
        self.relations = relations
        self.control = control
        self.graph = graph

        self.fuse = block.factory_fusion(self.fusion)
        self.graph_module = GraphModule(**graph)
        self.control_module =ControlModule(**control)
        self.relations_module = RelationsModule(**relations)


    def forward(self, q, s, coords=None):

        # 1. fuse q and s_(i-1) find m_i
        m_i = self.fusion_module(q, s)

        # 2. put q and c_(i-1) to recurrent unit, find c_i
        c_i = self.control_module(q)

        # 3. put c_i and m_i to the graph module => g_i = M*M'*c_i

        # graph learner
        g_i = self.graph_module(in_feature_dim=feat_dim + hid_dim,
                                        combined_feature_dim=512,
                                        K=36,
                                        dropout=dropout)


        # 4. put m_i and g_i to relations module =>
            # B(b_i,b_j,W) = r_ij, e_i = sum over j(gi*r_ij)
            # s_i = s_(i-1) + e_i
        e_i = self.relations_module(g_i, m_i)

        s_new = s + e_i

        return s_new, c_i



    def fusion_module(self, q, s):
        bsize = s.shape[0]
        n_regions = s.shape[1]
        s = s.contiguous().view(bsize*n_regions, -1)
        s = self.fuse([q, s])
        s = s.view(bsize, n_regions, -1)
        return s




