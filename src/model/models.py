import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
#! 直接用dgl的GCN不就好了，为啥自己写了一个
from .torch_gcn import GCN
from .torch_gat import GAT

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert.modules())[-2].out_features
        # self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.linear = th.nn.Linear(self.feat_dim, nb_class)
        # self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        #! 继承的nn.module里面的，如果是model.train()的话，这个就是true
        if self.training:
            #! 这样每次每个epoch 重复，bert不会覆盖原来上面的表征么？
            #! 我觉得初始化完就不用再去动它了啊
            cls_feats = self.bert(input_ids, attention_mask)[0][:, 0]
            #! g是那个图，这边是取出来cls作为每个document embedding
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            #! 没有训练的时候就直接取
            cls_feats = g.ndata['cls_feats'][idx]
        #! 这边cls_feats都是document的
        cls_logit = self.linear(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        #! 这里是指定idx来把相应的数据取出来的
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        #! 把cls的pred和gcn的pred加权了一下
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        #! 这个m就是两个的加权
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        #! 还取了一下log
        pred = th.log(pred)
        return pred
    
class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
