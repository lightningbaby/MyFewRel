import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout() #set some element value to 0 with some prob

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        # support emb size[200,230], query emb [200,230]
        support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D) [4,10,5,230] 4大组，每大组里10小组，每小组里5行，每行230列
        query = query.view(-1, total_Q, self.hidden_size) # (B, total_Q, D) [4,50,230]

        B = support.size(0) # Batch size
         
        # Prototypical Networks 
        # Ignore NA policy
        #support 原先是[[4,10,5,230],torch.mean(support,2)按照第2维求和取平均
        #相当于对多个[5,230]的向量mean，按行相加，按列平均，所以最后是[4,10,230]
        support = torch.mean(support, 2) # Calculate prototype for each class [4,10,230]，4组，10行，每行230列
        logits = -self.__batch_dist__(support, query) # (B, total_Q, N) [4,50,11]?
        minn, _ = logits.min(-1)#minn是最小的值，_是对应索引，min(-1)是按照行，找每一行最小的值,dim change[x,row,col]->[x,row] 每小组每行的最小值结合为了一行
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1) unsqueeze(2)在dim=2增加一维，
        #minn.unsqueeze(2) [x,row]->[x,row,1],minn的每一行有col个最小值，为了将他们分开，所以增加了一维，让每行对应一个最小值，为了与logits的维度相同
        #minn.unsqueeze(2) - 1 每个元素-1
        #torch.cat([a,b],2) a,b[x,row,col] 列拼接，b的列拼在a的列后面，最后[x,row,col_a+col+b]
        #finally logits[B,total_Q,N+1] N+1
        _, pred = torch.max(logits.view(-1, N+1), 1)
        #一个样本看作一行[-1,N+1]，按行求最大值，_是最大值，pred是index
        return logits, pred

    
    
    
