from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import block
from .pairwise import Pairwise
from .graph import GraphModule
from .control import ControlModule
from .relations import RelationsModule

class iteregCell(nn.Module):
    def __init__(self,
                 residual=True,
                 fusion={},
                 relations={},
                 control={},
                 graph={}):

        super(iteregCell, self).__init__()

        self.residual = residual
        self.fusion = fusion
        self.relations = relations
        self.control = control
        self.graph = graph

        self.fusion_module = block.factory_fusion(self.fusion)
        self.relations_module = RelationsModule(**relations)
        self.control_module = ControlModule(**control)
        self.graph_module = GraphModule(**graph)


    def forward(self, q, s_i, c_i, coords=None):

        # 1. fuse q and s_(i-1) find m_i
        m_new = self.process_fusion(q, s_i)

        # 2. put q and c_(i-1) to recurrent unit, find c_i
        c_new = self.control_module(q, c_i)

        # 3. put c_i and m_i to the graph module => g_i = M*M'*c_i

        # graph learner
        feat_dim = m_new.shape[1]
        hid_dim = c_new.shape[0]
        g_new = self.graph_module(in_feature_dim=feat_dim + hid_dim,
                                        combined_feature_dim=512,
                                        K=36,
                                        dropout=1)


        # 4. put m_i and g_i to relations module =>
            # B(b_i,b_j,W) = r_ij, e_i = sum over j(gi*r_ij)
            # s_i = s_(i-1) + e_i
        e_i = self.relations_module(g_new, m_new)

        s_new = s_i + e_i

        return s_new, c_new


    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]
        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm



