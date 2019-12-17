#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_dataloader.py
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019-12-10 19:44   tangyubao      1.0         None
'''

# import lib
import numpy as np
import json
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder,CNNSentenceEncoderWithSummary
from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised,get_loader_with_summary
from fewshot_re_kit.framework import FewShotREFramework
import models
from torch import optim, nn
from models.proto import Proto
import os

max_length=128

glove_mat = np.load('./pretrain/glove/glove_mat.npy')
glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))

if not os.path.exists('test_checkpoint'):
    os.mkdir('test_checkpoint')

sentence_encoder = CNNSentenceEncoderWithSummary(glove_mat,glove_word2id,max_length)#only init
train_data_loader = get_loader_with_summary('val_wiki_with_void_smry', sentence_encoder,
                               N=5, K=5, Q=5, na_rate=0, batch_size=4)
# test_data_loader = get_loader('val_wiki', sentence_encoder,
#                                N=5, K=5, Q=5, na_rate=0, batch_size=4)
# val_data_loader = get_loader('val_wiki', sentence_encoder,
#                                N=5, K=5, Q=5, na_rate=0, batch_size=4)
framework = FewShotREFramework(train_data_loader, train_data_loader,train_data_loader)
model = Proto(sentence_encoder, hidden_size=230)

#要到train里 才会开始embedding数据
framework.train(model, 'prefix', 4, 5, 5, 5, 5,
                pytorch_optim=optim.SGD, load_ckpt=None, save_ckpt='test_checkpoint/test_dataloader.pth.tar',
                na_rate=0, val_step=10, fp16=False, pair=False,
                train_iter=30, val_iter=10, bert_optim=False)

acc = framework.eval(model, 4, 5, 5, 5, 10, na_rate= 0 , ckpt='test_checkpoint/test_dataloader.pth.tar', pair=False)
print("RESULT: %.2f" % (acc * 100))
