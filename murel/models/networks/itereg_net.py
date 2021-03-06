from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.vqa_net import mask_softmax
from block.models.networks.mlp import MLP
from .murel_cell import MuRelCell
from .itereg_cell import iteregCell


class iteReGNet(nn.Module):

    def __init__(self,
                 txt_enc={},
                 self_q_att=False,
                 n_step=3,
                 shared=False,
                 cell={},
                 agg={},
                 classif={},
                 wid_to_word={},
                 word_to_wid={},
                 aid_to_ans=[],
                 ans_to_aid={}):
        super(iteReGNet, self).__init__()
        self.self_q_att = self_q_att
        self.n_step = n_step
        self.shared = shared
        self.cell = cell
        self.agg = agg
        self.c_zero = torch.randn(size=(1, 512), dtype=torch.float)
        self.c_zero = self.c_zero.cuda()
        self.c_zero = nn.Parameter(self.c_zero)
        # self.c_expand = self.c[:, None, :].expand(4, self.c.shape[1])
        # self.c_expand = self.c_expand.contiguous().view(4, -1)
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        if self.shared:
            self.cell = iteregCell(**cell)
        else:
            self.cells = nn.ModuleList([iteregCell(**cell) for i in range(self.n_step)])

        if 'fusion' in self.classif:
            self.classif_module = block.factory_fusion(self.classif['fusion'])
        elif 'mlp' in self.classif:
            self.classif_module = MLP(self.classif['mlp'])
        else:
            raise ValueError(self.classif.keys())

        Logger().log_value('nparams',
                           sum(p.numel() for p in self.parameters() if p.requires_grad),
                           should_print=True)

        Logger().log_value('nparams_txt_enc',
                           self.get_nparams_txt_enc(),
                           should_print=True)

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        self.buffer = None

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def set_buffer(self):
        self.buffer = {}
        if self.shared:
            self.cell.pairwise.set_buffer()
        else:
            for i in range(self.n_step):
                self.cell[i].pairwise.set_buffer()

    def set_pairs_ids(self, n_regions, bsize, device='cuda'):
        if self.shared and self.cell.pairwise:
            self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
        else:
            for i in self.n_step:
                if self.cells[i].pairwise:
                    self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        coord = batch['norm_coord']

        # total_norm = 0
        # for p in list(filter(lambda p: p.grad is not None, self.parameters())):
        #     # print(p.grad.data.norm(2).item())
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print(total_norm)

        q = self.process_question(q, l)

        bsize = q.shape[0]
        n_regions = v.shape[1]

        # cell
        mm = v
        c_exp = self.c_zero.repeat(bsize, 1)
        for i in range(self.n_step):
            cell = self.cell if self.shared else self.cells[i]
            mm_nan = torch.isnan(mm)
            c_exp_nan = torch.isnan(c_exp)
            coord_nan = torch.isnan(coord)
            q_nan = torch.isnan(q)
            if q_nan.sum() > 0:
                print(q)
                print("q is nan")
            if mm_nan.sum() > 0:
                print(mm)
                print("mm is nan")
            if c_exp_nan.sum() > 0:
                print(c_exp)
                print("c_exp_nan is nan")
            if coord_nan.sum() > 0:
                print(coord)
                print("coord_nan is nan")

            mm, c_exp = cell(q, mm, c_exp, coord)

            if self.buffer is not None:  # for visualization
                self.buffer[i] = deepcopy(cell.pairwise.buffer)

        if self.agg['type'] == 'max':
            mm = torch.max(mm, 1)[0]
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)

        if 'fusion' in self.classif:
            logits = self.classif_module([q, mm])
        elif 'mlp' in self.classif:
            logits = self.classif_module(mm)

        out = {'logits': logits}
        return out

    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            # self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att * q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:, 0])
            q = self.txt_enc._select_last(q, l)

        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out

