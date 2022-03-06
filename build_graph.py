import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]
#! 用sys的argv 和 sys的exist来搭配使用
if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []
#! 这个不是corpus里面的，是data下面的，标记了是train还是test的文件
f = open('data/' + dataset + '.txt', 'r')
#! readlines是一次性全部读取进来
lines = f.readlines()
for line in lines:
    #! 这是所有按顺序的
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    #! 这边是给每个数据划分一下，看它是test还是train
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)
#! 原来就有clean，这个clean怎么来的得再看下论文，我看了下就是把一些符号弄掉
#! 这个是corpus里面的，就是真正的内容的
doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
#! 这是全部都放到content列表里面了
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

#! 这个是看当前这个train sample在总的数据集里面的index
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
#! suffle一下
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]
#! 把id全部写进去
train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

#! 同样的方法把test的id都写进去
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

#! 这是train和test所有的id
ids = train_ids + test_ids
print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    #! doc_name
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    #! 这是doc_name对应的doc_content
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
#! 也都分别写出去
f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
#! 把suffle后的doc content的内容，然后对他去
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)


#! 这个就是去得到整个数据集的倒排索引
word_doc_list = {}
#! i 是shuffle的doc的index
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    #! 这边是每个文档维护一个appear的set
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            #! word_doc_list是记录了每个单词在哪个doc出现过，有点像倒排索引那样
            doc_list = word_doc_list[word]
            doc_list.append(i)
            #! 这边就是更新了一下新的倒排索引上来，比较简单粗暴
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

#! 这个是记录每个word在几个doc里面出现过
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

#! 这个是id和word的映射
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

#! 把vocab也写出去了
vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

# label list
#! 这个就是label, 把label都写进来
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)
#! 划分出来的真的数据库
f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    #!初始化一个doc embedding
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    #! doc_words就是文档
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        #! 把有的word的embedding都加到doc embedding上面
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    #! i是行，j是列，i是每个doc的idx, j是每个word的embedding元素，我感觉好像是在存embedding?
    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
#! sp.csr_matrix是scipy里面的一个parse包，这里的x就是每个doc的embedding我觉得
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))


#! 这个y是label的one-hot向量
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)


#! 同样的操作，对test 数据集进行处理
# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

#! 这里是
word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)
#! 这个allx 和 ally应该是包含了document 和 word的一个整体的embedding矩阵
print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''
#! 这里要开始构建图了，co-occurence 这边是为了算互信息的，用的是window的机制去处理
# word co-occurence with context windows
window_size = 20
#! 放所有的window
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

#! 记录每个词在多少个window中出现
word_window_freq = {}
for window in windows:
    #! 以window 为单位的, window里面出现过1次就只算一次
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

#! 这个是去统计一个window下面，两个word 同时出现在同一个window下面的次数
#! 这个pair有正向的也有反向的
word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []


#! 这边就是去算两个word的互信息
# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    #! 这里train_size我的理解应该就是维护一个大的矩阵？
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
#! 这个是去算这个各个词在各个文档中所出现的次数
doc_word_freq = {}
for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

#! 算出idf 然后和 freq结合算出TF-IDF
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        #! weight是之前就定义的了
        weight.append(freq * idf)
        doc_word_set.add(word)

#! node 的数量
node_size = train_size + vocab_size + test_size
#! 所有的weight的矩阵，因为它把词和词以及文档和词的权重都放到weight数组里面了
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))


#! 然后全部保存起来
# dump objects
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()


#! 我发现它根本就没有用word_vector_map, 这里build_map应该是没有去构造document和word的embedding的