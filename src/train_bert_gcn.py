import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar  # 进度条
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import json
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
import networkx as nx
import wandb
import torch

wandb.init(project='crime-classification', name='bert-gcn', entity='hengyuan')

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=256, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='../models/pretrained_models/bert-base-chinese',
                    choices=['../models/pretrained_models/bert-base-chinese', 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default='../models/fine_tuned_models/crime_classifier_epoch8.pt')
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-5)
parser.add_argument('--bert_lr', type=float, default=1e-5)

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
#! 这个的意思就是用哪个种类的预训练BERT
bert_init = args.bert_init
#! 这个是可以在某个ckpt的基础上继续训练
pretrained_bert_ckpt = args.pretrained_bert_ckpt
# dataset = args.dataset
#! 模型的保存地址
checkpoint_dir = args.checkpoint_dir
#! 用GCN还是GAT
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = '../models/saved_models/bert_{}'.format(gcn_model)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
#! 把这个文件复制过去
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:6')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model

#! 可以看下这边load出来的都是什么，然后后面应该都不用改了了，就把load_corpus换成自己的

# Data Preprocess
#! 这边返回的对象全是numpy的，然后adj是压缩的形式csr:compressed sparse 
#! 我要改的应该就是和这个load_corpus的输出一样的就可以
#! 这个features 没有用到
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
#! feature应该是所有的node的embedding
'''

adj, weights, train_mask, val_mask, test_mask = json.load(open('../data/processed_data/graph.json')), \
    json.load(open('../data/processed_data/weights.json')), \
    torch.load('../data/processed_data/train_doc_mask.pt'), \
    torch.load('../data/processed_data/dev_doc_mask.pt'), \
    torch.load('../data/processed_data/test_doc_mask.pt')

id2document, word2id, labels = json.load(open('../data/processed_data/id2document.json')), \
    json.load(open('../data/processed_data/word2id.json')), \
    json.load(open('../data/processed_data/labels.json')), 
# compute number of real train/val/test/word nodes and number of classes
#! 原来里面有的是str, 有的是int, 之前没处理好，这里补救一下
labels = list(map(int, labels))
nb_node = train_mask.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
#! 这个应该指的是有几个单词
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = 61

# instantiate model according to class number
#! BERT init，就是Bert或者Roberta这些
if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)
else:
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)

#! 就在在这个任务的BERT的预训练模型，这里可以考虑用我之前fine-tuning的BERT
if pretrained_bert_ckpt is not None:
    logger.info('loading pretrained model..')
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.load_state_dict(ckpt, strict=False)
    # model.bert_model.load_state_dict(ckpt['bert_model'])
    # model.classifier.load_state_dict(ckpt['classifier'])


# load documents and compute input encodings
#! 这边是把数据集都放到一个list下面一起去encode
# corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
# with open(corpse_file, 'r') as f:
#     text = f.read()
#     text = text.replace('\\', '')
#     text = text.split('\n')
logger.info('adding content to text list..')
text = []
for idx, word_list in id2document.items():
    content = ''.join(word_list)
    text.append(content)


#! 这里就是把所有的文本都编码好
def encode_input(text, tokenizer):
    #! tokenizer这里直接调用的是这个类的__call__函数
    #! 这里padding = 'max_length'意思是用指定的max_length来pad, 还有一种是longest, 这个是按每个batch的sample的最大长度去pad
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask

logger.info('encoding text list ..')
input_ids, attention_mask = encode_input(text, model.tokenizer)

#! 这里就是把input_ids的顺序和node 矩阵对应上
#! 这边是把train、eval、(word_num, 512), test给拼起来
#! 因为原来的input_id 采用的是max_length的策略来padding的，所以这里可以把它们cat起来
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# transform one-hot label to class ID for pytorch computation
#! 把one-hot转换一下

#! 所有doc的label (one hot 形式)
# y = y_train + y_test + y_val
#! 单独只有y_train
# y_train = y_train.argmax(axis=1)
y_train = torch.zeros(nb_node)
y_train[train_mask] = torch.Tensor(labels)[train_mask]
#! 所有的y
# y = y.argmax(axis=1)

y = torch.Tensor(labels)

# document mask used for update feature
#! 所有的加起来，这里的doc_mask就是除了word_mask之外的
#! 意思就是用来后面把所有的document 筛选出来的
#! 这里向加的意思就是各个True的并的计算操作
doc_mask  = train_mask + val_mask + test_mask

# build DGL Graph
#! 这边是把邻接矩阵norm一下，这边在对角线上，也就是自己和自己的权重都加1来处理
# adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
#! 这里应该是邻接矩阵里面本身就有权重了
g = dgl.graph(adj)
g.edata['edge_weight'] = torch.FloatTensor(weights) 
# g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
#! 自己往g.ndata里面装新的东西进去
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
#! label包括train_doc, val_doc, test_doc的label, 也就是所有的label
#! train, val, test 就是把True - > 1.0这样
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    y.long(), train_mask.float(), val_mask.float(), test_mask.float()
#! 用这些train_label去更新图, 其他的label只是用来看效果
g.ndata['label_train'] = y_train.long()
#! 一开始全用0初始化，后面才去赋值的
#! 这里是先用0初始化了，到时候应该是直接加上去就可以
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

#! dataset也全是index
# create index loader
#! 这边使用index 去做dataloader的
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
#! 这个test是test_idx, 因为input_id的顺序是train_doc, val_doc, word, test_doc
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
#! 这是所有的document的idx
#! 除了train, val, test 之外，还有一个word呢
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)

idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=8
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            #! cls
            output = model.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    #! 刚才的cls_feats是用0去先占位置的，现在才是把document embedding放进去
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
        {'params': model.bert.parameters(), 'lr': bert_lr},
        {'params': model.linear.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    #! 记住，这边batch 取出来的x 都是idx
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    #! 用idx把train数据筛选出来
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    #! 只把train的pred筛选出来
    y_pred = model(g, idx)[train_mask]
    #! 也是一样
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    #! 防止g.ndata里面的document 的 embedding参与更新
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)
#! 这个是新添加的pbar
pbar = ProgressBar()
pbar.attach(trainer, ['loss'])  # loss表示要跟进度条一起显示的数据

@trainer.on(Events.EPOCH_COMPLETED)
#! reset_graph就是把document embedding都reset一下
def reset_graph(trainer):
    #! scheduler只是调整学习率
    scheduler.step()
    #! 每个epoch结束以后就重置所有的document embedding
    #! 因为BERT被微调了

    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        #! 所有document 的label, 这个label里面有train, 我觉得是错的，
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    #! 分别跑了三个idx_loader, 一个是train, 一个是val, 一个是test
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    wandb.log({'train_acc':train_acc, 'train_loss':train_nll, 'val_acc':val_acc, 'val_loss':val_nll, 'test_acc':test_acc, 'test_loss':test_nll})
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        #! 这里相当于是只保留了我要的东西而已
        th.save(
            {
                'bert_model': model.bert.state_dict(),
                'classifier': model.linear.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
#! 一开始先reset一下
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
