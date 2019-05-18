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
        bsize = s_i.shape[0]
        n_regions = s_i.shape[1]
        q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize*n_regions, -1)
        m_new = self.process_fusion(q_expand, s_i)

        # 2. put q and c_(i-1) to recurrent unit, find c_i
        c_new = self.control_module(q, c_i)

        # 3. put c_i and m_i to the graph module => g_i = M*M'*c_i
        g_new = self.graph_module(m_new, c_new)

        # 4. put m_i and g_i to relations module =>
            # B(b_i,b_j,W) = r_ij, e_i = sum over j(gi*r_ij)
            # s_i = s_(i-1) + e_i
        e_i = self.relations_module(m_new, g_new, coords)

        s_new = s_i + e_i

        return s_new, c_new


    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]
        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm



