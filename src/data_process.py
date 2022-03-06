# -*- encoding:utf-8 -*-
from tqdm import tqdm
from loguru import logger
import jieba 
import torch
from math import log
import json
import sys

class DataProcess(object):
    def __init__(self, train_data_path, dev_data_path, test_data_path):
        self.train, self.dev, self.test = open(train_data_path, 'r'), open(dev_data_path, 'r'), \
        open(test_data_path, 'r')
        self.stop_words = {word.strip() for word in open('../data/stopwords.txt')}
        self.word2id, self.id2word, self.id2document = {}, {}, {}
        self.word_count, self.labels = {}, []
        self.word_freq_in_cur_content = {}
        self.word_in_content_freq = {}
        self.TF_IDF, self.PMI = {}, {}

    def __filter_stop_words(self, line):
        words_list = jieba.lcut(line)
        filter_words_list = [word.strip() for word in words_list if word not in self.stop_words]
        return filter_words_list

    def __count_for_single_dataset(self, dataset):
        for line in tqdm(dataset):
            content, label = line.strip().split('\t')
            cur_filter_word_list = self.__filter_stop_words(content)
            #! word_list 换成是char级别的
            # cur_filter_word_list = list(''.join(cur_filter_word_list))
            for word in cur_filter_word_list:
                if word not in self.word_count:
                    self.word_count[word] = 1
                else: self.word_count[word] += 1
                

    def __map_for_single_dataset(self, dataset, start_idx):
        dataset.seek(0)
        # process_bar = tqdm(desc=f'process {dataset} dataset', total=len(dataset))
        for idx, line in tqdm(enumerate(dataset, start=start_idx)):
            content, label = line.strip().split('\t')
            cur_filter_word_list = self.__filter_stop_words(content)                
            #! 换成char级别的
            # self.id2document[idx] = list(''.join(cur_filter_word_list))
            self.id2document[idx] = cur_filter_word_list
            self.labels.append(label)
            # process_bar.update(1)
        return idx


    def get_word_count_dict(self, train, dev, test):
        for dataset in [train, dev, test]:
            self.__count_for_single_dataset(dataset)

    def __generate_mask(self, global_node_num, part_start, part_end): 
        temp_mask = torch.zeros(global_node_num)
        temp_mask[part_start:part_end + 1] = 1
        return temp_mask == 1
    

    def do_all_map(self, train, dev, test):
        self.get_word_count_dict(train, dev, test)
        train_end_idx = self.__map_for_single_dataset(train, 0)
        dev_end_idx = self.__map_for_single_dataset(dev, train_end_idx + 1)
        for word_idx, word in tqdm(enumerate(self.word_count.keys(), start=dev_end_idx + 1)):
            self.word2id[word] = word_idx
            self.id2word[word_idx] = word
            #! 所有的word对应的label都是0
            self.labels.append(0)
        test_end_idx = self.__map_for_single_dataset(test, word_idx + 1)
        global_node_num = test_end_idx + 1
        train_doc_mask, dev_doc_mask, test_doc_mask = self.__generate_mask(global_node_num, 0, train_end_idx), \
            self.__generate_mask(global_node_num, train_end_idx + 1, dev_end_idx), \
            self.__generate_mask(global_node_num, word_idx + 1, test_end_idx)
            
        return train_doc_mask, dev_doc_mask, test_doc_mask, global_node_num, self.labels
    
    def __get_single_tfidf(
        self, 
        word_freq_in_content, 
        content_length, 
        all_document_num,
        word_in_content_freq
        ):
        TF = word_freq_in_content / content_length
        IDF = log(all_document_num / (word_in_content_freq + 1))
        return TF * IDF

    def get_tfidf(self, ):
        for idx, cur_word_list in tqdm(self.id2document.items()):
            #! 一个文档里面各个词出现的次数
            self.word_freq_in_cur_content[idx] = {}
            # cur_word_list = jieba.lcut(content)
            for word in cur_word_list:
                if word not in self.word_freq_in_cur_content[idx]:
                    # import pdb;pdb.set_trace()
                    self.word_freq_in_cur_content[idx][self.word2id[word]] = 1
                else:
                    self.word_freq_in_cur_content[idx][self.word2id[word]] += 1
            #! 词在多少个文档出现过
            for word_id in self.word_freq_in_cur_content[idx].keys():
                if word_id not in self.word_in_content_freq:
                    self.word_in_content_freq[word_id] = 1
                else: self.word_in_content_freq[word_id] += 1

        all_document_num = len(self.id2document.keys())
        self.TF_IDF = self.word_freq_in_cur_content
        #! 最终都是用有的东西的词典去构造
        # import pdb; pdb.set_trace()
        for doc, doc_word_dict in tqdm(self.word_freq_in_cur_content.items()):
            for word in doc_word_dict:
                self.TF_IDF[doc][word] = self.__get_single_tfidf(
                    self.word_freq_in_cur_content[doc][word],
                    len(self.id2document[doc]),
                    all_document_num,
                    self.word_in_content_freq[word]
                )
    def __get_single_pmi(self, co_count, num_window, word_freq_i, word_freq_j):
        pmi = log((1.0 * co_count / num_window) /
                        (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        return pmi

    def get_pmi(self, ):
        #! 这里要开始构建图了，co-occurence 这边是为了算互信息的，用的是window的机制去处理
        window_size = 30
        #! 放所有的window
        windows = []
        for idx, cur_word_list in tqdm(self.id2document.items()):
            # cur_word_list = jieba.lcut(content)
            cur_word_list_length = len(cur_word_list)
            if cur_word_list_length <= window_size:
                windows.append(cur_word_list)
            else:
                for j in range(cur_word_list_length - window_size + 1):
                    window = cur_word_list[j:j + window_size]
                    windows.append(window)

        #! 记录每个词在多少个window中出现
        word_window_freq = {}
        for window in tqdm(windows):
            #! 以window 为单位的, window里面出现过1次就只算一次
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[self.word2id[window[i]]] += 1
                else:
                    word_window_freq[self.word2id[window[i]]] = 1
                appeared.add(window[i])
        

        #! 这个是去统计一个window下面，两个word 同时出现在同一个window下面的次数
        word_pair_count = {}
        for window in tqdm(windows):
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i, word_j = window[i], window[j]
                    word_i_id, word_j_id = self.word2id[word_i], self.word2id[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    #! 这个pair有正向的也有反向的
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        #!算两个word的互信息
        num_window = len(windows)
        for key in word_pair_count:
            temp = key.split(',')
            #! i,j 是word的index
            i, j = int(temp[0]), int(temp[1])
            co_count = word_pair_count[key]
            word_freq_i, word_freq_j  = word_window_freq[i], word_window_freq[j]
            pmi = self.__get_single_pmi(co_count, num_window, word_freq_i, word_freq_j)
            if pmi <= 0:
                continue
            if word_freq_i not in self.PMI:
                self.PMI[word_freq_i] = {}
            self.PMI[word_freq_i][word_freq_j] = pmi
            
    def build_graph(self, ):
        self.get_tfidf()
        self.get_pmi()
        graph, weights = [], []

        for doc, doc_word_dict in tqdm(self.TF_IDF.items()):
            for word, tf_idf in doc_word_dict.items():
                graph.append(tuple((doc, word)))
                weights.append(tf_idf)

        for word_i, word_word_dict in tqdm(self.PMI.items()):
            for word_j, pmi in word_word_dict.items():
                graph.append(tuple((word_i, word_j)))
                weights.append(pmi)
        return graph, weights
 

    def __call__(self, ):
        logger.info('building mapping ...')
        train_doc_mask, dev_doc_mask, test_doc_mask, global_node_num, labels = self.do_all_map(self.train, self.dev, self.test)
        logger.info('building graph ...')
        graph, weights = self.build_graph()
        logger.info('saving mask, graph, weights ...')
        torch.save(train_doc_mask, '../data/processed_data/train_doc_mask.pt')
        torch.save(dev_doc_mask, '../data/processed_data/dev_doc_mask.pt')
        torch.save(test_doc_mask, '../data/processed_data/test_doc_mask.pt')
        json.dump(graph, open('../data/processed_data/graph.json', 'w+'))
        json.dump(weights, open('../data/processed_data/weights.json', 'w+'))
        json.dump(self.labels, open('../data/processed_data/labels.json', 'w+'))
        json.dump(self.id2document, open('../data/processed_data/id2document.json', 'w+'))
        json.dump(self.word2id, open('../data/processed_data/word2id.json', 'w+'))
        json.dump(self.id2word, open('../data/processed_data/id2word.json', 'w+'))
        
        
        return graph, weights, self.id2document, self.word2id, train_doc_mask, dev_doc_mask, test_doc_mask, global_node_num, labels
        

if __name__ == '__main__':
    # train_data_path = '../data/filter_train_processed.tsv'
    # dev_data_path = '../data/filter_valid_processed.tsv'
    # test_data_path = '../data/filter_test_processed.tsv'

    # train_data_path = '../data/test_whole_process_2.tsv'
    # dev_data_path = '../data/test_whole_process_2.tsv'
    # test_data_path = '../data/test_whole_process_2.tsv'
    train_data_path = '../data/filter_dataset/train.tsv'
    dev_data_path = '../data/filter_dataset/dev.tsv'
    test_data_path = '../data/filter_dataset/test.tsv'
    p = DataProcess(train_data_path, dev_data_path, test_data_path)
    sys.exit(p())