import json
import logging
import os
from sklearn import preprocessing
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from fedml_api.data_preprocessing.mydatasplit import data_int


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
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


# 专门为了能用data_size_bid这个函数的
def read_data1(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    data_size_bid = []

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        # print("cdatas是什么", type(cdata), cdata.keys(), "num_samplesnum_samples",cdata['num_samples'],"求解", sum(cdata['num_samples']), 'users:::::',cdata['users'] ) #'user_data'
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])
        data_size_bid = size_bid(cdata['num_samples'])  # 将每个人data size转换为bid
        print("data_size_bid的长度和类型是什么", len(data_size_bid), type(data_size_bid),"数据", data_size_bid)

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    clients = sorted(cdata['users'])
    return clients, groups, train_data, test_data, data_size_bid  # data_size_bid add by zsh


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


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']  # lable

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_mnist_by_device_id(batch_size,
                                           device_id,
                                           train_path="MNIST_mobile",
                                           test_path="MNIST_mobile"):
    train_path += '/' + device_id + '/' + 'train'
    test_path += '/' + device_id + '/' + 'test'
    return load_partition_data_mnist(batch_size, train_path, test_path)

# 修改
import gzip

def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
            # imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 1, 784)
    return (x_train, y_train)


def getlist_num(user_num, train_true=True):
    list_num = []
    for i in range(user_num):
        np.random.seed(i)
        if train_true:
            a = np.random.randint(300, 900, 1).tolist()[0]  # 抽取范围，每个client对应划分的数据量
        else:
            a = np.random.randint(50, 140, 1).tolist()[0]
        list_num.append(a)
    return list_num

def  changedata(traindata,trainlabel,indexlist,usernum,train_true=True):
    # /获取用户数据量
    listnum=getlist_num(usernum, train_true)
    #构建数据
    userdict={}
    k=0
    user=0
    # print("indexlist-----------", len(indexlist))
    for num in listnum:
        # print('indexlist[k:k+num]----------',indexlist[k:k+num])
        traindatalist=traindata[indexlist[k:k+num]].tolist()
        trainlabellist = trainlabel[indexlist[k:k + num]].tolist()
        k+=num
        userdict["f_"+str(user).rjust(5, '0')] = {"y":trainlabellist,"x":traindatalist}
        user+=1
    return userdict, listnum

def read_datamy(args):
    train_data, train_lable = load_data(r'/hy-tmp/HFL-robust/fedml_api/data_preprocessing/data/MNIST/raw/', "train-images-idx3-ubyte.gz",
                                        "train-labels-idx1-ubyte.gz")
    test_data, test_lable = load_data(r'/hy-tmp/HFL-robust/fedml_api/data_preprocessing/data/MNIST/raw/', "t10k-images-idx3-ubyte.gz",
                                      "t10k-labels-idx1-ubyte.gz")
    # Fashion-MNIST
    # train_data, train_lable = load_data(r'D:\fashion', "train-images-idx3-ubyte.gz",
    #                                     "train-labels-idx1-ubyte.gz")
    # test_data, test_lable = load_data(r'D:\fashion', "t10k-images-idx3-ubyte.gz",
    #                                   "t10k-labels-idx1-ubyte.gz")
    data = np.r_[train_data, test_data]
    lable = np.r_[train_lable, test_lable]
    # print('-----data的类型', (data.reshape(70000, 784)).shape)
    data = size_normalization(data)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data.reshape(70000, 784))
    data = data.reshape(70000, 28, 28)
    # print('--------data', data.shape, data)
    # 打乱的索引序列
    np.random.seed(0)
    permutation = np.random.permutation(lable.shape[0])  # 7万都打乱
    # print('permutation-----',len(permutation))
    validate_datasets = 0.875  # 训练和测试划分比例
    train_indexs = permutation[:int(lable.shape[0] * validate_datasets)]  # 0-63000
    # print('train_indexs------', len(train_indexs))
    validate_indexs = permutation[int(lable.shape[0] * validate_datasets):]  # 63000-

    usernum = args.client_num_in_total
    train_dataset, train_sample_num = changedata(data, lable, train_indexs, usernum)
    test_dataset,test_sample_nums = changedata(data, lable, validate_indexs, usernum, train_true=False)
    data_size_bid = size_bid(train_sample_num)
    # groups = []
    return train_dataset.keys(), [], train_dataset, test_dataset, data_size_bid, train_sample_num  # data_size_bid add by zsh

def load_partition_data_mnist(args):
    print('args', args)
    batch_size = args.batch_size
    users, groups, train_data, test_data, data_size_bid, num_sample_data = read_datamy(args)  # 读数据函数return中加了data_size_bid,冲突了
    # print("users----------", users)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()  # 把用户个人的数据都放在这个
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]['x'])  # train_data[u]-----['f_00000', 'f_00001', 'f_00002'
        user_test_data_num = len(test_data[u]['x'])
        # print("user_train_data_num_daxiao",user_test_data_num, "user_test_data_num",user_test_data_num)
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num
        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        # print("train_batch------------------", train_batch)
        test_batch = batch_data(test_data[u], batch_size)
        # index using client index
        train_data_local_dict[client_idx] = train_batch  # 1000中的client_idx与1000
        # print("train_batch________",len(train_batch))
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10
    print("train_data_global+++++++", len(train_data_global),len(users))
    # print("groups----------", groups)

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, data_size_bid, num_sample_data

def load_partition_data_mnist_new(args):
    users, groups, train_data, test_data, data_size_bid, num_sample_data , test_sample_num = data_int(args)  #
    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = sum(num_sample_data)
    test_data_num = sum(test_sample_num)
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    for k in range(args.num_clients):
        train_data_local_dict[k] = train_data[k]
        test_data_local_dict[k] = test_data[k]
        train_data_local_num_dict[k] = num_sample_data[k]
    client_num = args.num_clients
    class_num = 10
    return client_num, train_data_num, test_data_num, num_sample_data, test_sample_num, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, data_size_bid, num_sample_data
