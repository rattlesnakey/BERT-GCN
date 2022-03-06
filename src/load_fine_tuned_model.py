# -*- encoding:utf-8 -*-
import torch
from model import * 

if __name__ == '__main__':
    model = BertGCN(nb_class=61, pretrained_model='../models/pretrained_models/bert-base-chinese', m=0.5, gcn_layers=2,
                n_hidden=200, dropout=0.5)
    state_dict = torch.load('../models/fine_tuned_models/crime_classifier_epoch8.pt', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    # temp_dict = {}
    # bert_dict = {}
    # linear_dict = {}
    # for k, v in x.items():
    #     if 'bert' in k:
    #         k = '.'.join(k.split('.')[1:])
    #         bert_dict[k] = v
    #     if k == 'linear.weight' or k == 'linear.bias':
    #         k = '.'.join(k.split('.')[1:])
    #         linear_dict[k] = v
    # temp_dict['bert'] = bert_dict
    # temp_dict['linear'] = linear_dict

    # # model.load_state_dict(x)
    # import pdb; pdb.set_trace()