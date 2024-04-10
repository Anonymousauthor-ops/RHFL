from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist   # 调用

# from fedml_api.data_preprocessing.MNIST.data_loader import read_data   # 调用
import json
import logging
import os

import numpy as np
import torch

train_path = "./../../../data/MNIST/train"
test_path = "./../../../data/MNIST/test"
# add by zsh根据归一化结果，抽取bid，即bid与data size关联
def size_bid(size_normal):
    data = size_normalization(size_normal)
    size_normalization_bid = []
    np.random.seed(0)  # 随机种子，确保用户在整个过程中的报价不变。后续如果想要每一次迭代都想要重新选择clients,可以改变seed让其随round_id变化
    # 原来的定价规则
    for i in data:
        if 0<=i and i<= 0.3:
            bid = np.random.uniform(1, 3)
        elif 0.3<i and i<= 0.5:
            bid = np.random.uniform(2, 5)
        elif 0.5<=i and i<= 0.7:
            bid = np.random.uniform(3, 7)
        else:
            bid = np.random.uniform(4, 9)
        size_normalization_bid.append(bid)
    return size_normalization_bid

# data size归一化
def size_normalization(data):
    _range = np.max(data) - np.min(data)
    size_normal = (data - np.min(data)) / _range
    return size_normal

def read_data(train_data_dir, test_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    data_size_bid = []  # by zsh
    num_sample_data = []  # by zsh

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        # print("cdatas是什么", type(cdata), cdata.keys(), "num_samplesnum_samples",cdata['num_samples'],"求解",
        # sum(cdata['num_samples']), 'users:::::',cdata['users'] ) #'user_data'
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])   # 所有的训练数据
        data_size_bid = size_bid(cdata['num_samples'])  # 将每个人data size转换为bid
        print("data_size_bid的长度和类型是什么", len(data_size_bid), type(data_size_bid))
        num_sample_data = cdata['num_samples']
        print('num_sample_data的内容:', len(num_sample_data), type(num_sample_data))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    # data_size_bid and num_sample_data  add by zsh
    return clients, groups, train_data, test_data,  data_size_bid, num_sample_data


# users, groups, train_data, test_data, data_size_bid, num_sample_data = read_data(train_path, test_path)

# print("train_data大小", type(train_data))
print('-----------', str(100).rjust(5,'0'))



