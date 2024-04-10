# CS,ES,clients, HFL, simple version
import copy
import logging
import random
from copy import deepcopy
import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
import torch.nn.functional as F
from collections import OrderedDict
import collections
import math

import torch
# import wandb
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

import os
import sys
from sklearn.metrics import accuracy_score
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.MNIST.data_loader import  load_partition_data_mnist_new, load_partition_data_mnist   # 加载测试数据

from fedml_api.standalone.fedavg.client import Client
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.cnn import CNN_DropOut, CNN_DropOut1, CNN_DropOut2
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_OriginalFedAvg1, RNN_StackOverFlow
from fedml_api.model.cv.resnet_gn import resnet18, resnet18mnist
# from fedml_experiments.standalone.fedavg.main_fedavg import create_model    #  循环调用不行的
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.losssum = []
        self.device = device
        self.args = args
        self.templosslist = [0] * (self.args.client_num_in_total + 1)
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
         data_size_bid, num_sample_data] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # self.data_size_bid = data_size_bid  # zsh
        self.datasize = []  # add zsh
        # self.client_indexes =   # add zsh
        self.data_size_bid = data_size_bid  # zsh
        self.num_sample_data = num_sample_data  # zsh
        self.model_trainer = model_trainer
        self.client_indexes = []  #
        # self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.test_acc_round = []
        # self.B = args.buget  # 预算为100，后续可以写在运行参数中

    # 根据抽样方法初始化参加的index
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer, modelstr, lista):
        logging.info("############setup_clients (START)#############")
        self.client_list = []  # 置空
        # self.client_indexes = self._client_sampling_AAAI(self.args.client_num_in_total, modelstr, lista)
        # self.client_indexes = self._client_sample_HFL(modelstr)
        # for round_idx in range(self.args.client_num_in_total):
        self.client_indexes = self.allocate_clients_distance(modelstr, self.args.client_num_in_total)
        print('输出是什么？', self.client_indexes)
        for client_idx in self.client_indexes:  #
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)

                self.client_list.append(c)
        logging.info("############setup_clients (END)#############")
        print('self.client_list------', len(self.client_list))
        # print('self.client_indexes----',self.client_indexes)

    def data_change_lable(self, malicious_clients, malicious_m_rate_list, train_data_local_dict, client_idx):
        data_combine = []
        if client_idx not in malicious_clients:
            return train_data_local_dict
        for batch_idx, (x, labels) in enumerate(train_data_local_dict):
            data_combine.append((x, self.deal_label1(labels, malicious_m_rate_list[client_idx])))
        return data_combine

    def deal_label(self, labels, malicious_m_rate):
        labels = np.array(labels)
        label_len = labels.shape[0]
        label_len_num = int(np.ceil(label_len * malicious_m_rate))  # 向下取整，每个用户需要加噪的标签个数
        jiazao_index = np.random.choice(label_len, label_len_num, replace=False)  # 不取相同的client
        # 对index的标签进行翻转
        labels[jiazao_index] = (labels[jiazao_index] + 1) % 10
        return torch.from_numpy(labels)

    def deal_label1(self, labels, malicious_m_rate):  # 恶意用户的标签随机赋值
        labels = np.array(labels)
        label_len = labels.shape[0]
        # label_len_num = int(np.floor(label_len * malicious_m_rate))  # 向上取整,每个用户需要加噪的标签个数
        label_len_num = int(np.ceil(label_len * malicious_m_rate))  # 向下取整,每个用户需要加噪的标签个数
        # jiazao_index = np.random.randint(0, label_len, label_len_num)  #
        jiazao_index = np.random.choice(label_len, label_len_num, replace=False)  # 不取相同的client
        # 对index的标签加噪
        labels[jiazao_index] = np.random.randint(0, 9, label_len_num)
        return torch.from_numpy(labels)

    # 获得，每个client的加噪比/信誉比列表
    def get_malicious_m_rate(self, client_num_in_total):
        malicious_m_rate_list = []
        for i in range(client_num_in_total):
            np.random.seed(i)
            malicious_m_rate_list.append(np.random.uniform(1, 1))
        return malicious_m_rate_list

    def train(self):
        # #  获取测试集，只加载一次
        # result = load_partition_data_mnist(self.args)
        result = load_partition_data_mnist_new(self.args)
        print('result--------------------', result[5])
        test_data_global = result[4]  # test_data_global是list类型
        El_result = []
        # 对指定数量---m个用户进行加噪
        np.random.seed(0)
        malicious_clients = np.random.choice(self.args.client_num_in_total, self.args.malicious_m, replace=False)
        # malicious_clients = [0, 1, 2, 3, 4, 5]  # 认为控制m的列表
        print('malicious_clients恶意用户：', malicious_clients)

        CS_global = OrderedDict()
        for round_idx in range(self.args.comm_round):
            lista = [1 for _ in range(self.args.client_num_in_total)]
            k_array = []
            for i in range(self.args.k):
                k_array.append(self.args.task_name+'_'+str(i))
            self.args.task_array = k_array
            ES_global = []   # 保存ESs的全局模型
            for model_index, model_name in enumerate(self.args.task_array):
                if CS_global:
                    self.model_trainer.set_model_params(CS_global)
                else:
                    self.model_trainer = self.create_model(model_name, 10)
                self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict,
                                    self.test_data_local_dict, self.model_trainer, model_name, lista)   # 添加lr cnn rnn  shican
                w_global = self.model_trainer.get_model_params()  # 初始化全局模型11-12

                # print('w_global的类型是？',type(w_global))
                # print('w_global是什么样子', w_global)
                logging.info("################Communication round : {}".format(round_idx))
                w_locals = []
                for idx, client in enumerate(self.client_list):
                    # update dataset
                    client_idx = self.client_indexes[idx]
                    malicious_m_rate_list = self.get_malicious_m_rate(self.args.client_num_in_total)
                    train_data = self.data_change_lable(malicious_clients, malicious_m_rate_list,
                                                        self.train_data_local_dict[client_idx], client_idx)
                    client.update_local_dataset(client_idx, train_data,
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])
                    # LF加噪
                    w = client.train(copy.deepcopy(w_global), self.model_trainer)
                    # logging.info('----------loggg', w)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

                    # 高斯加噪
                    # if client_idx in malicious_clients:
                    #     w = self.reset_local_params(copy.deepcopy(w_global))
                    # else:
                    #     w = client.train(copy.deepcopy(w_global), self.model_trainer)
                    # w_locals.append((client.get_sample_number(), copy.deepcopy(w)))


#  #我们的=================start================
                # # 计算全局模型和每个参与者的模型的KL散度
                # calculate_kl_diverg = self.compute_kl_divergence(w_global, w_locals)
                # # if round_idx > 0:
                # #     calculate_kl_diverg = self.compute_kl_divergence(w_global, w_locals)
                # # else:  # 预训练
                # #     loadmodel = torch.load("\\2023code\\HFL-robust\\fedml_api\\standalone\\fedavg\\premod\\_model.mod")
                # #     w_global1 = loadmodel.cpu().state_dict()
                # #     calculate_kl_diverg = self.compute_kl_divergence(w_global1, w_locals)
                # print('calculate_kl_diverg----', calculate_kl_diverg)
                # print('self.client_indexes',self.client_indexes)
                # # 设置阈值，排除可能恶意的
                # # threshold = 0.00135  # 固定值
                # threshold = self.compute_threshold(calculate_kl_diverg, self.client_indexes)  # 动态阈值
                # result_dict = dict(zip(self.client_indexes, calculate_kl_diverg))
                # remian_clients = [key for key, value in result_dict.items() if value[1] < threshold]
                # print('remian_clients-----', remian_clients)
                # w_locals_id = dict()
                # for index, w_local in zip(self.client_indexes, w_locals):
                #     w_locals_id[index] = w_local
                # # print('w_locals_id-----', w_locals_id)
                # # 把对应的人挑出来
                # w_locals_id1 = {key: value for key, value in w_locals_id.items() if key in remian_clients}
                # # print('w_locals_id1----',w_locals_id1)
                # w_locals_id2 = [(value[0], value[1]) for value in w_locals_id1.values()]
                # # update global weights
                # w_global = self._aggregate(w_locals_id2)  # ours
                #  #我们的=================end ================

                w_global = self._aggregate(w_locals)  # HFL的
                # 简单平均聚合策略 CS端
                ES_global.append(w_global)

            #  #我们的=================start================
            # # 计算每个ES的测试acc，并加权权重。
            # weight = self.compute_acc_weight(round_idx, ES_global)
            # # print('ES_global长度------', ES_global)
            # # 计算每个ES端全局模型的测试acc,并处理为聚合的权重
            # CS_global = self._aggregate_CS_acc_weight(ES_global, weight)  # acc加权聚合

            CS_global = self._aggregate_CS(ES_global)   # HFL的简单平均聚合
            self.model_trainer.set_model_params(CS_global)
            # 测试CS端的全局模型
            # at last round
            if round_idx == self.args.comm_round - 1:
                acc_test = self._local_test_on_all_clients(round_idx, self.model_trainer)
                El_result.append(acc_test)
            # per {frequency_of_the_test} round
            # elif round_idx % self.args.frequency_of_the_test == 0:  # 取余，5轮打一次
            else:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    acc_test = self._local_test_on_all_clients(round_idx, self.model_trainer)
                    El_result.append(acc_test)
        print('El_result-------', El_result)
        self.save_test_acc(El_result)


    def compute_kl_divergence(self, w_global, w_locals):   # 通，但有负值
        # 对w_locals处理，将数据类型与w_global相同的部分分离出来放在新的列表w_locals_pram中
        w_locals_pram = []
        for client_data, client_model_update in w_locals:
            if isinstance(client_model_update,
                          collections.OrderedDict) and client_model_update.keys() == w_global.keys():
                w_locals_pram.append((client_data, client_model_update))
        # 计算global与w_locals_pram中的每个client的模型更新的KL散度
        w_locals_KL = []
        for client_data, client_model_update in w_locals_pram:
            kl_divergence = 0.0
            for global_param, client_param in zip(w_global.values(), client_model_update.values()):
                global_param_flatten = global_param.flatten()
                client_param_flatten = client_param.flatten()
                kl_divergence += torch.nn.functional.kl_div(torch.log_softmax(client_param_flatten, dim=0),
                                                            torch.softmax(global_param_flatten, dim=0),
                                                            reduction='batchmean')
            # w_locals_KL.append((client_data, torch.exp(torch.tensor(kl_divergence)).item()))
            w_locals_KL.append((client_data, kl_divergence.item()))  #原来的计算方式
        return w_locals_KL

    def compute_threshold(self, calculate_kl_diverg, client_indexes):
        calculate_client_kl = {}
        for i in range(len(client_indexes)):
            calculate_client_kl[client_indexes[i]] = calculate_kl_diverg[i][1]
        calculate_client_kl1_sorted = sorted(calculate_client_kl.items(), key=lambda x: x[1], reverse=True)

        first_magnitude = math.floor(math.log10(abs(calculate_client_kl1_sorted[0][1])))
        threshold = calculate_client_kl1_sorted[0][1]

        if all(math.floor(math.log10(abs(value))) == first_magnitude for _, value in calculate_client_kl1_sorted):
            # 所有客户端对应的值的量级相等
            threshold = calculate_client_kl1_sorted[0][1]
        else:
            # 客户端的值的量级不相同
            # threshold = 10 ** (first_magnitude - 1) # 比它小一个量级
            threshold = 10 ** first_magnitude
        return threshold

    def compute_acc_weight(self, round_idx, ES_global):
        ES_global_acc = []
        for model_param in ES_global:
            self.model_trainer.set_model_params(model_param)
            acc_test1 = self._local_test_on_all_clients(round_idx, self.model_trainer)
            ES_global_acc.append(acc_test1)
            # 使用softmax处理
        exp_acc = np.exp(ES_global_acc)
        normalized_acc = exp_acc / np.sum(exp_acc)
        return normalized_acc

    def save_list_to_file(self, lst, filename):
        with open(filename, 'w') as file:
            for item in lst:
                file.write(str(item) + '\n')


    def _aggregate_CS(self, w_locals):  # CS简单平均聚合
        avg = 1/len(w_locals)
        from collections import OrderedDict
        averaged_params = OrderedDict()
        for i in range(len(w_locals)):
            local_model_params = w_locals[i]
            for k in local_model_params.keys():
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * avg
                    else:
                        averaged_params[k] += local_model_params[k] * avg
        return averaged_params

    # def _aggregate_CS_acc_weight(self, w_locals, weight):  # CS采用acc加权平均聚合
    #     avg = 1/len(w_locals)
    #     from collections import OrderedDict
    #     averaged_params = OrderedDict()
    #     for i in range(len(w_locals)):
    #         local_model_params = w_locals[i]
    #         for k in local_model_params.keys():
    #                 if i == 0:
    #                     averaged_params[k] = local_model_params[k] * avg
    #                 else:
    #                     averaged_params[k] += local_model_params[k] * avg
    #     return averaged_params
    def _aggregate_CS_acc_weight(self, w_locals, weight): # CS采用acc加权平均聚合
        from collections import OrderedDict
        averaged_params = OrderedDict()
        for i, local_model_params in enumerate(w_locals):
            ES_weight = weight[i]
            for k, v in local_model_params.items():
                if i == 0:
                    averaged_params[k] = v * ES_weight
                else:
                    averaged_params[k] += v * ES_weight
        return averaged_params


    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    #  三个模型的预测结果  for hard
    def test_predict_hard(self, test_data, model_zz):  # model_zz形参
        y_predict = []
        target_list = []
        model_zz.eval()
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                # print("batch_idx",batch_idx, 'x:',x, 'target:', target)
                pred = model_zz(x)
                _, predict = torch.max(pred, 1)
                y_predict.extend(predict.tolist())
                target_list.extend(target.tolist())
        return y_predict, target_list

    def save_test_acc(self, test_acc):
        le = len(test_acc)
        index = [i for i in range(le)]
        data = {'round_id': index, 'test_acc': test_acc}
        df = pd.DataFrame(data)

        m = self.args.malicious_m
        df.to_csv('./csv/HFL-n50-simpleavg_k_3_' + '_M_' + str(m) + 'mnist_.csv', index=None)
        # df.to_csv(self.args.model + '_file.csv', index=None)
        # df.to_csv('./csv/_file_EL.csv', index=None)   # 只保存一个就可以，方便集成学习

    # 1000个人中随机抽人，提前假设所有人都能可以参加
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # def allocate_clients_distance(self, model, num_clients, round_id):
    def allocate_clients_distance(self, model, num_clients):
        client_indexes = []
        distance_range = [1, 5]   # 距离的范围
        num_servers = 3  # 边缘服务器的数量，即cnn_0, cnn_1, cnn_2
        task_comb = [f'cnn_{i}' for i in range(num_servers)]
        dis_dict = {}
        allocation_result = {}
        # 设置随机种子
        # random.seed(round_id)
        # 为每个客户端生成不重复的随机距离列表
        for client in range(num_clients):
            random.seed(client+1)
            dis_list = random.sample(range(distance_range[0], distance_range[1] + 1), num_servers)
            dis_dict[client] = dis_list
        # 初始化边缘服务器的客户端索引列表
        for server in range(num_servers):
            allocation_result[task_comb[server]] = []
        # 根据最近距离原则将客户端分配给边缘服务器
        for client, distances in dis_dict.items():
            min_distance = min(distances)
            closest_servers = [task_comb[i] for i, distance in enumerate(distances) if distance == min_distance]
            assigned_server = random.choice(closest_servers)
            allocation_result[assigned_server].append(client)
        print(allocation_result)
        for m in allocation_result.keys():  # 得到参加某个任务对应的clients
            if m == model:
                client_indexes = allocation_result[m]
        return client_indexes



    def _client_sample_HFL(self, model):  # 适用于HFL的随机分配方案
        client_indexes = []
        task_preference_client_all = self.single_minded()  # 分配偏好
        allocation_result = defaultdict(list)
        # 遍历任务偏好列表
        for i, task_preference in enumerate(task_preference_client_all):
            task = task_preference[0]  # 获取任务名称
            allocation_result[task].append(i)  # 将参与者索引添加到对应的任务列表中
        # 将字典转换为普通的字典类型
        allocation_result = dict(allocation_result)
        print("-------111111allocation_result:", allocation_result)
        for m in allocation_result.keys():  # 得到参加某个任务对应的clients
            if m == model:
                client_indexes = allocation_result[m]
        return client_indexes

    def _client_sampling_AAAI(self, client_num_in_total, model, templosslist):    # B下随机选择
            bid_each_client = []  # # 新的bid规则，从[1,3]中抽取,获取bid list
            for i in range(client_num_in_total):
                np.random.seed(i)
                a = np.random.uniform(1, 3)  # bid form [1,3]
                bid_each_client.append(a)
            client_indexes = []
            task_preference_client_all = self.single_minded()  # 分配偏好
            print('task_preference_client_all---',task_preference_client_all)
            input_dict = {}  # 拼成字典
            for k, v, t, r in zip(range(client_num_in_total), bid_each_client, task_preference_client_all,
                                  templosslist):
                input_dict[k] = [v, t, r]
            task_preference_client_all1 = self.single_minded1()  # 分配偏好------按照INforcom那个作者的思路来写，分开的
            # 别人的背包方案
            winners, allocation, payment = self.getmultresult_AAAI(input_dict, task_preference_client_all1, self.args.budget)
            allocation_result = allocation
            print("-------111111allocation_result:", allocation_result)
            for m in allocation_result.keys():  # 得到参加某个任务对应的clients
                if m == model:
                    client_indexes = allocation_result[m]
            # print('client_indexes-----------', client_indexes)
            return client_indexes


    def getmultresult_AAAI(self,hdict, task_preference_client_all, B):
        W = []
        X = {}
        P = []
        for k,v in B.items():
            v1=task_preference_client_all[k]
            mid = dict((key,value) for key,value in hdict.items() if key in v1)
            w1,x1,p1=self.get_result_AAAI(mid,v)
            W.extend(w1)
            X.update(x1)
            P.extend(p1)
        return W,X,P

    # for AAAI
    def get_result_AAAI(self, hdict, B):
        W = []
        X = {}
        P = []
        result = sorted(hdict.items(), key=lambda x: x[1][0] / (len(x[1][1]) * x[1][2]), reverse=True)
        # result = hdict
        for i in range(len(result) - 1):
            # cur = len(result[i][1][1]) * result[i][1][2]
            # next = result[i + 1][1][0] / (len(result[i + 1][1][1]) * result[i + 1][1][2])
            Pk = result[i][1][0]
            if Pk <= B:
                B -= Pk
                W.append(result[i][0])
                P.append(Pk)
                for k in result[i][1][1]:
                    if k not in X:
                        X[k] = [result[i][0]]
                    else:
                        X[k].append(result[i][0])
        return W, X, P

    #  任务列表，得到任务的全组合
    def taskcom(self, task_array):
        if len(task_array) == 0:
            return []
        from itertools import combinations
        res = []
        for i in range(1, len(task_array) + 1):
            res.extend(list(combinations(task_array, i)))
        return res

    def single_minded(self):  # 封装成函数，便于各个方法调用相同的结果，便于实验同维度对比
        task_comb = ['cnn_0', 'cnn_1', 'cnn_2']  # 限制每个用户最多能做1个
        task_preference_client_all = []
        for i in range(self.args.client_num_in_total):
            np.random.seed(i)
            task_preference_per_client = np.random.choice(task_comb, 1)
            task_preference_client_all.append(task_preference_per_client.tolist())
        # print('task_preference_client_all的-----', task_preference_client_all)
        return task_preference_client_all

    def single_minded1(self):  # 封装成函数，便于各个方法调用相同的结果，便于实验同维度对比
        # task_comb = self.taskcom(self.args.task_array)  # 原来随机抽1-3个模型的
        task_comb = ['cnn_0', 'cnn_1', 'cnn_2']  # 限制每个用户最多能做1个
        task_preference_client_all1 = dict()
        for i in range(self.args.client_num_in_total):
            # 固定single-minded
            np.random.seed()
            task_preference_per_client = np.random.choice(task_comb, 1)
            model=task_preference_per_client.tolist()[0]
            if model not in task_preference_client_all1:
                task_preference_client_all1[model] = [i]
            else:
                task_preference_client_all1[model].append(i)
        # print('task_preference_client_all1的-----', task_preference_client_all1)

        return task_preference_client_all1

    # test数据集获取  zsh
    def get_test_dataset(self, ):
        x_test = []
        y_test = []
        for k in self.client_indexes:
            # print('---------------------------len', len(self.test_data_local_dict[k][0]))
            for (date, lable) in self.test_data_local_dict[k][0]:
                x_test.append(date)
                y_test.append(lable)
        return x_test, y_test

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def reset_local_params(self, w_lo):
        averaged_params1 = w_lo
        for k in averaged_params1.keys():
            # averaged_params1[k] = torch.randn(averaged_params1[k].shape)   加上shape是3*5
            averaged_params1[k] = torch.nn.init.normal_(averaged_params1[k], mean=0, std=200)
            # averaged_params1[k] = torch.randn(averaged_params1[k].shape) * 5
        # print('--------梯度权重', averaged_params1)
        return averaged_params1

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx, model):  # 测试每一轮结束后的全部模型训练和测试的ACC
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        # global test_acc_round   # 记录每轮次的test ACC,只能保存当前轮的
        # test_acc_round = []

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False, model)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True, model)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        # print('============================test_metrics',test_metrics)
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        self.templosslist[self.args.client_num_in_total]=test_loss


        self.test_acc_round.append(test_acc)
        # print('测试test_acc_roundtest_acc_roundtest_acc_roundtest_acc_roundtest_acc_round', self.test_acc_round)

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        return test_acc

    def _local_test_on_validation_set(self, round_idx):  # 主要用于莎士比亚数据集

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Pre": test_pre, "round": round_idx})
            # wandb.log({"Test/Rec": test_rec, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)


    def create_model(self, model_name, output_dim):  # 将对应的模型创建进来，集成的是cnn,rnn,resnet18
        logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
        model = None
        # if model_name == "lr":
        #     logging.info("LogisticRegression + MNIST")
        #     model = LogisticRegression(28 * 28, output_dim)

        if "_" in model_name:
            model_name = model_name.split("_")[0]   # 位置0变为cnn


        if model_name == "cnn":  # by zsh
            torch.manual_seed(0)
            logging.info("CNN + MNIST")
            model = CNN_DropOut2(True)

        elif model_name == "rnn":  # add by zsh
            logging.info("RNN + mnist")
            model = RNN_OriginalFedAvg1()

        elif model_name == "resnet18_gn1":
            logging.info("ResNet18_GN1 + mnist")
            model = resnet18mnist(n_class=10)

        return MyModelTrainerCLS(model)
        # return model