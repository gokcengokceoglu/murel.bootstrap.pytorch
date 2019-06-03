import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class GraphModule(Module):
    def __init__(self, in_feature_dim, combined_feature_dim, K, dropout=0.0):
        super(GraphModule, self).__init__()

        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Parameters
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim
        self.K = K

        # Embedding layers
        self.edge_layer_1 = nn.Linear(in_feature_dim,
                                      combined_feature_dim)
        #self.edge_layer_2 = nn.Linear(combined_feature_dim,
        #                              combined_feature_dim)

        # Regularization
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        #self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, mm_new, ci):
        '''
        ## Inputs:
        - mm_new (batch_size, K, in_feat_dim): multimodal features
        - ci (in_c_dim): control vector

        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''

        bsize = mm_new.shape[0]
        n_regions = mm_new.shape[1]

        mm_new = mm_new.view(-1, mm_new.shape[2])

        #c_uns = ci.unsqueeze(dim=1)
        #c_expand = c_uns.expand(bsize, n_regions, c_uns.shape[2])
        #c_expand = c_expand.contiguous().view(-1, c_expand.shape[2])

        #graph_nodes = torch.cat((mm_new, c_expand), dim=1)

        # graph_nodes = image_qenc_cat.view(-1, self.in_dim)
        graph_nodes = mm_new

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        #h = self.edge_layer_2(h)
        #h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))

        return adjacency_matrix
