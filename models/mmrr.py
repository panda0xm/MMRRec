# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from weighting.Aligned_MTL import Aligned_MTL
from weighting.STCH import STCH
from weighting.ExcessMTL import ExcessMTL
from weighting.FairGrad import FairGrad
from weighting.Nash_MTL import Nash_MTL

from common.abstract_arch import AbsArchitecture


class MMRR(GeneralRecommender,FairGrad,AbsArchitecture):
    def __init__(self, config, dataset):
        super(MMRR, self).__init__(config, dataset)
        self.init_param()
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        
        self.rec_weight = config['rec_weight']
        self.rec2_weight = config['rec2_weight']
        self.align_weight = config['align_weight']
        self.align2_weight = config['align2_weight']
        self.bpr_weight = config['bpr_weight']
        
        self.task_num=5
        self.train_batch_size=1
        

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        #if self.v_feat is not None:
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        nn.init.xavier_normal_(self.image_trs.weight)
        #if self.t_feat is not None:
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        nn.init.xavier_normal_(self.text_trs.weight)
        
        #decoder
        self.image_decoder = nn.Linear( self.feat_embed_dim, self.v_feat.shape[1])
        self.text_decoder = nn.Linear( self.feat_embed_dim, self.t_feat.shape[1])
        nn.init.xavier_normal_(self.image_decoder.weight)
        nn.init.xavier_normal_(self.text_decoder.weight)
        
        

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        #A._update(data_dict)
        for key, value in data_dict.items():
            A[key] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        #self.norm_adj是ID嵌入，得到e_u,e_i
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings=i_g_embeddings + h
        #以上是嵌入过程（是一种图卷积）,得到h_u,h_i
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interactions):
        # online network
        h_u, h_i = self.forward()
        
        h_v, h_t = None, None
        #if self.t_feat is not None:
        h_t = self.text_trs(self.text_embedding.weight)
        #if self.v_feat is not None:
        h_v = self.image_trs(self.image_embedding.weight)
        #以上是模态表示,得到h_v,h_t
        
        #以下是对比特征生成,得到u_o, u_t, i_o, i_t, v_o, v_t, t_o, t_t
        with torch.no_grad():
            u_t = h_u.clone()
            i_t = h_i.clone()
            
            u_t.detach()
            i_t.detach()
            
            u_t = F.dropout(u_t, self.dropout)
            i_t = F.dropout(i_t, self.dropout)

            #if self.t_feat is not None:
            t_t = h_t.clone()
            t_t = F.dropout(t_t, self.dropout)

            #if self.v_feat is not None:
            v_t = h_v.clone()
            v_t = F.dropout(v_t, self.dropout)

        u_o_all = self.predictor(h_u)
        i_o_all = self.predictor(h_i)
        users = interactions[0]
        pos_items = interactions[1]
        neg_items = interactions[2]
        
        u_o = u_o_all[users, :]
        i_o = i_o_all[pos_items, :]
        j_o = i_o_all[neg_items, :]
        
        u_t = u_t[users, :]
        i_t = i_t[pos_items, :]

        loss_it, loss_iv, loss_tt, loss_vv = 0.0, 0.0, 0.0, 0.0
        #if self.t_feat is not None:
        t_o = self.predictor(h_t)
        t_o = t_o[pos_items, :]
        t_t = t_t[pos_items, :]
        #if self.v_feat is not None:
        v_o = self.predictor(h_v)
        v_o = v_o[pos_items, :]
        v_t = v_t[pos_items, :]
        e_t=self.text_embedding.weight[pos_items, :]
        e_v=self.image_embedding.weight[pos_items, :]
        #然后是重构任务，交互图重构，模态重构，decoder,得到r_v,r_t        
        loss_ui = 1 - cosine_similarity(u_o, i_t.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_o, u_t.detach(), dim=-1).mean()
        r_v=self.image_decoder(v_o)
        r_t=self.text_decoder(t_o)
        loss_tt = 1 - cosine_similarity(r_t, e_t, dim=-1).mean()
        loss_vv = 1 - cosine_similarity(r_v, e_v, dim=-1).mean()
        
        loss_rec=(loss_ui + loss_iu)
        loss_rec2=(loss_tt+loss_vv)
        
        #然后是对齐任务，
        #模态和嵌入对齐(i_o,v_o)(i_o,t_o)，
        loss_it = 1 - cosine_similarity(t_o, i_t.detach(), dim=-1).mean()
        loss_iv = 1 - cosine_similarity(v_o, i_t.detach(), dim=-1).mean()
        #模态自己对齐(v_o,v_t)(t_o,t_t)，（是不是和模态重构重复，需要实验证实是否重复以及哪个效果更好）BM3本质上是通过对比学习重构模态，我有单独的重构任务，效果应该更好
        loss_tt1 = 1 - cosine_similarity(t_o, t_t.detach(), dim=-1).mean()
        loss_vv1 = 1 - cosine_similarity(v_o, v_t.detach(), dim=-1).mean()
        
        loss_align=(loss_it + loss_iv )
        loss_align2=(loss_tt1 + loss_vv1)
        
        #BPR任务
        pos_item_score, neg_item_score = torch.mul(u_o, i_o).sum(dim=1), torch.mul(u_o, j_o).sum(dim=1)
        loss_bpr = - (pos_item_score - neg_item_score).sigmoid().log().mean()
        
        
        #以下是最终损失                        
        loss_reg=self.reg_loss(h_u, h_i)
        #print('train loss_rec: %.4f loss_align_mask: %.4f loss_align: %.4f loss_mask: %.4f' % (loss_rec,loss_align_mask,loss_align,loss_mask))
        #loss = loss_rec + self.cl_weight * loss_align_mask + self.reg_weight * self.reg_loss(h_u, h_i)
        loss = self.rec_weight *loss_rec +self.rec2_weight *loss_rec2+ self.align_weight * loss_align + self.align2_weight * loss_align2 +self.bpr_weight * loss_bpr +  self.reg_weight * loss_reg
        loss_list=[loss_rec,loss_rec2,loss_align,loss_align2]
        train_losses = torch.zeros(self.task_num).to(self.device)
        train_losses[0]=self.rec_weight *loss_rec
        train_losses[1]=self.rec2_weight *loss_rec2
        train_losses[2]=self.align_weight * loss_align
        train_losses[3]=self.align2_weight * loss_align2
        train_losses[4]=self.bpr_weight * loss_bpr
        #print(train_losses)
        return train_losses
        
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui
        
