#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tot_emb.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-12-10 14:57   tangyubao      1.0         None
'''

# import lib
import torch
from . import embedding,summary_emb


def cat_embedding(word_vec_mat, max_length,word_embedding_dim, pos_embedding_dim):
    sample_emb=embedding.Embedding(word_vec_mat, max_length,word_embedding_dim, pos_embedding_dim)
    sum_emb=summary_emb.Summary_Embedding(word_vec_mat, max_length,word_embedding_dim)

    x = torch.cat([sample_emb,sum_emb], 2)
    return x



