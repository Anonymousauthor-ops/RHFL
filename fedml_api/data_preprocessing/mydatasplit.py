import json
import logging
import os
from sklearn import preprocessing
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse

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


def iid_esize_split(dataset, args, kwargs, is_shuffle = True):  # iid
    """
    split the dataset to users
    Return:
        dict of the data_loaders
    """
    sum_samples = len(dataset)
    num_samples_per_client = int(sum_samples / args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace = False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)
    return data_loaders


def niid_esize_split(dataset, args, kwargs, is_shuffle = True):  # non-iid for two class
    data_loaders = [0] * args.num_clients
    # each client has only two classes of the network
    num_shards = 2* args.num_clients
    # the number of images in one shard
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    # is_shuffle is used to differentiate between train and test
    if is_shuffle:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # sort the data according to their label
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)

    #divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace= False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                    batch_size = args.batch_size,
                                    shuffle = is_shuffle, **kwargs)
    return data_loaders

def Dirichlet_non_iid(train_dataset, args):
    np.random.seed(0)
    torch.manual_seed(0)
    batch_size = 10

    bias_weight = 0.3 # control the non-iid

    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = args.num_clients / 10

    each_worker_idx = {i: [] for i in range(args.num_clients)}
    for idx, (_, label) in enumerate(train_dataset):
        y = label
        # assign a data point to a group
        upper_bound = y * (1 - bias_weight) / 9. + bias_weight
        lower_bound = y * (1 - bias_weight) / 9.
        rd = np.random.random_sample()

        if rd > upper_bound:
            worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y + 1)
        elif rd < lower_bound:
            worker_group = int(np.floor(rd / other_group_size))
        else:
            worker_group = y

        # assign a data point to a worker
        rd = np.random.random_sample()
        selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
        each_worker_idx[selected_worker].append(idx)

    data_loaders = []
    for i in range(args.num_clients):
        subset = DatasetSplit(train_dataset, each_worker_idx[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        data_loaders.append(loader)

    return data_loaders

def split_data(dataset, args, kwargs, is_shuffle = True):
    """
    return dataloaders
    """
    if args.iid == 0:
        data_loaders = iid_esize_split(dataset, args, kwargs, is_shuffle)
    elif args.iid == 1:
        data_loaders = niid_esize_split(dataset, args, kwargs, is_shuffle)  # two class
    elif args.iid == -1:
        data_loaders = Dirichlet_non_iid(dataset, args)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders

def data_int(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.FashionMNIST(os.path.join(r'data/'), train=True,
                           download=True, transform=transform)
    test = datasets.FashionMNIST(os.path.join(r'data/'), train=False,
                          download=True, transform=transform)

    users = [i for i in range(args.num_clients)]
    group = [None for i in users]
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)

    data_size_bid = [1 for i in range(args.num_clients)]
    train_sample_num_all = len(train)
    train_sample_num_per_client = train_sample_num_all / args.num_clients
    train_sample_num = [train_sample_num_per_client for i in range(args.num_clients)]

    test_sample_num = [len(test)/args.num_clients for _ in range(args.num_clients)]

    return users, group, train_loaders, test_loaders, data_size_bid, train_sample_num, test_sample_num



def data_int_cifar10(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(os.path.join(r'data/'), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(os.path.join(r'data/'), train=False, download=True, transform=transform)

    users = [i for i in range(args.num_clients)]
    group = [None for i in users]
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)

    data_size_bid = [1 for i in range(args.num_clients)]
    train_sample_num_all = len(train)
    train_sample_num_per_client = train_sample_num_all / args.num_clients
    train_sample_num = [train_sample_num_per_client for i in range(args.num_clients)]

    test_sample_num = [len(test) / args.num_clients for _ in range(args.num_clients)]

    return users, group, train_loaders, test_loaders, data_size_bid, train_sample_num, test_sample_num


