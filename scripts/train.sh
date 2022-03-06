set -v
set -e
DROPOUT=0.3
BERT_LR=1e-5
GCN_LR=1e-5
GCN_LAYERS=2
python ../src/train_bert_gcn.py --max_length 256 \
    --dropout ${DROPOUT} \
    --bert_lr ${BERT_LR} \
    --gcn_lr ${GCN_LR} \
    --gcn_layers ${GCN_LAYERS} \
    tee ../logs/training.log
