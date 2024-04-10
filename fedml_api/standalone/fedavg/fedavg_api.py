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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.MNIST.data_loader import  load_partition_data_mnist_new, load_partition_data_cifar10, load_partition_data_mnist

from fedml_api.standalone.fedavg.client import Client
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.cnn import CNN_DropOut, CNN_DropOut1, CNN_DropOut2
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_OriginalFedAvg1, RNN_StackOverFlow
from fedml_api.model.cv.resnet_gn import resnet18, resnet18mnist
from fedml_api.model.cv.resnet20 import resnet20cifar10
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
        self.datasize = []
        self.data_size_bid = data_size_bid
        self.num_sample_data = num_sample_data
        self.model_trainer = model_trainer
        self.client_indexes = []

        self.test_acc_round = []



    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer, modelstr, lista, round_id):
        logging.info("############setup_clients (START)#############")
        self.client_list = []
        self.client_indexes = self.allocate_clients_distance(modelstr, self.args.client_num_in_total, round_id)
        for client_idx in self.client_indexes:
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
        label_len_num = int(np.ceil(label_len * malicious_m_rate))
        jiazao_index = np.random.choice(label_len, label_len_num, replace=False)

        labels[jiazao_index] = (labels[jiazao_index] + 1) % 10
        return torch.from_numpy(labels)

    def deal_label1(self, labels, malicious_m_rate):
        labels = np.array(labels)
        label_len = labels.shape[0]
        label_len_num = int(np.ceil(label_len * malicious_m_rate))
        jiazao_index = np.random.choice(label_len, label_len_num, replace=False)

        labels[jiazao_index] = np.random.randint(0, 9, label_len_num)
        return torch.from_numpy(labels)


    def get_malicious_m_rate(self, client_num_in_total):
        malicious_m_rate_list = []
        for i in range(client_num_in_total):
            np.random.seed(i)
            malicious_m_rate_list.append(np.random.uniform(1, 1))
        return malicious_m_rate_list

    def train(self):
        if self.args.model == "mnist" or self.args.model == "fashionmnist":
            result = load_partition_data_mnist_new(self.args)
        elif self.args.model == "cifar10":
            result = load_partition_data_cifar10(self.args)
        El_result = []
        np.random.seed(0)
        malicious_clients = np.random.choice(self.args.client_num_in_total, self.args.malicious_m, replace=False)

        print('malicious_clients:', malicious_clients)

        CS_global = OrderedDict()
        
        for round_idx in range(self.args.comm_round):
            lista = [1 for _ in range(self.args.client_num_in_total)]
            k_array = []
            for i in range(self.args.k):
                k_array.append(self.args.task_name+'_'+str(i))
            self.args.task_array = k_array
            ES_global_list = []


            for model_index, model_name in enumerate(self.args.task_array):
                if CS_global:
                    self.model_trainer.set_model_params(CS_global)
                else:
                    self.model_trainer = self.create_model(model_name, 10)
                self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict,
                                    self.test_data_local_dict, self.model_trainer, model_name, lista, round_idx)
                ES_global = self.model_trainer.get_model_params()
                logging.info("################Communication round : {}".format(round_idx))
                user_global = copy.deepcopy(ES_global)

                Edge_epoch = 1
                for id_es in range(Edge_epoch):
                    w_users = []
                    param_lists = []
                    for idx, client in enumerate(self.client_list):
                        # update dataset
                        client_idx = self.client_indexes[idx]
                        #  Byzantine attacks
                        if self.args.attack_type == 'label' and self.args.method_name in ['JSHFL', 'HierHFL']:
                            train_data = self.label_fliiping_attack(malicious_clients,
                                                                    self.train_data_local_dict[client_idx], client_idx,
                                                                    client, user_global, w_users)
                            client.update_local_dataset(client_idx, train_data, self.test_data_local_dict[client_idx],
                                                        self.train_data_local_num_dict[client_idx])
                            w = client.train(copy.deepcopy(user_global), self.model_trainer)
                            if self.args.method_name in ['JSHFL', 'HierHFL']:
                                w_users.append((client.get_sample_number(), copy.deepcopy(w)))
                        else:
                            train_data = self.train_data_local_dict[client_idx]
                            client.update_local_dataset(client_idx, train_data, self.test_data_local_dict[client_idx],
                                                        self.train_data_local_num_dict[client_idx])
                            if self.args.attack_type == 'sign' and self.args.method_name in ['JSHFL','HierHFL']:
                                    client.train(copy.deepcopy(user_global), self.model_trainer)
                                    if client_idx in malicious_clients:
                                        param_client = self.model_trainer.local_grads_sum_epoch
                                        param_client = [-10 * x for x in param_client]
                                    else:
                                        param_client = self.model_trainer.local_grads_sum_epoch
                                    param_client = torch.concat([x.reshape((-1, 1)) for x in param_client], dim=0)
                                    w_local = self.param_to_w_local(param_client.clone().detach(), self.model_trainer, self.args.lr)
                                    w_users.append((torch.tensor(600).to(self.device), w_local))

                            elif self.args.attack_type == 'gaussian' and self.args.method_name in ['JSHFL', 'HierHFL']:
                                if client_idx in malicious_clients:
                                    w = self.reset_local_params(copy.deepcopy(user_global))
                                else:
                                    w = client.train(copy.deepcopy(user_global), self.model_trainer)
                                w_users.append((client.get_sample_number(), copy.deepcopy(w)))

                            # trim attack
                            elif self.args.attack_type in ['full_trim', 'krum_attack'] and self.args.method_name in ['JSHFL', 'HierHFL']:
                                client.train(copy.deepcopy(user_global), self.model_trainer)
                                param_client = self.model_trainer.local_grads_sum_epoch
                                param_client = torch.concat([x.reshape((-1, 1)) for x in param_client], dim=0)
                                param_lists.append(param_client)
                                print('param_lists:', len(param_lists))

                    # defense_method_
                    if self.args.method_name == 'JSHFL':
                        if self.args.attack_type == 'full_trim':
                            param_listsnew = self.full_trim(param_lists, self.args.malicious_m)
                            for i in range(len(param_listsnew)):
                                w_local = self.param_to_w_local(param_listsnew[i], self.model_trainer, self.args.lr)
                                w_users.append(w_local)
                            #defense
                            calculate_JS_diverg = self.compute_js_divergence(user_global, w_users)

                            threshold = self.compute_threshold_js(calculate_JS_diverg, self.client_indexes)  # threshold
                            result_dict = dict(zip(self.client_indexes, calculate_JS_diverg))
                            remian_clients = [key for key, value in result_dict.items() if
                                              value[1] < threshold]

                            print('remianing_clients-----', remian_clients)
                            w_users_id = dict()
                            for index, w_user in zip(self.client_indexes, w_users):
                                w_users_id[index] = w_user

                            w_users_id1 = {key: value for key, value in w_users_id.items() if key in remian_clients}
                            w_users_id2 = [(value[0], value[1]) for value in w_users_id1.values()]
                            # update global weights
                            user_global = self._aggregate(w_users_id2)
                            self.model_trainer.set_model_params(user_global)

                        elif self.args.attack_type == 'krum_attack':
                            b = int((len(malicious_clients) / self.args.client_num_in_total) * len(self.client_indexes))
                            param_listsnew = self.dir_partial_krum_lambda(param_lists, b, epsilon=0.01)
                            for i in range(len(param_listsnew)):
                                w_local = self.param_to_w_local(param_listsnew[i], self.model_trainer, self.args.lr)
                                w_users.append(w_local)
                            #defense
                            calculate_JS_diverg = self.compute_js_divergence(user_global, w_users)
                            threshold = self.compute_threshold_js(calculate_JS_diverg, self.client_indexes)  #
                            result_dict = dict(zip(self.client_indexes, calculate_JS_diverg))
                            remian_clients = [key for key, value in result_dict.items() if
                                              value[1] < threshold]

                            print('remian_clients-----', remian_clients)
                            w_users_id = dict()
                            for index, w_user in zip(self.client_indexes, w_users):
                                w_users_id[index] = w_user

                            w_users_id1 = {key: value for key, value in w_users_id.items() if key in remian_clients}
                            w_users_id2 = [(value[0], value[1]) for value in w_users_id1.values()]
                            # update global weights
                            user_global = self._aggregate(w_users_id2)
                            self.model_trainer.set_model_params(user_global)
                        else:
                            calculate_JS_diverg = self.compute_js_divergence(user_global, w_users)

                            threshold = self.compute_threshold_js(calculate_JS_diverg, self.client_indexes)

                            result_dict = dict(zip(self.client_indexes, calculate_JS_diverg))
                            remian_clients = [key for key, value in result_dict.items() if
                                              value[1] < threshold]

                            print('remian_clients-----', remian_clients)
                            w_users_id = dict()
                            for index, w_user in zip(self.client_indexes, w_users):
                                w_users_id[index] = w_user

                            w_users_id1 = {key: value for key, value in w_users_id.items() if key in remian_clients}
                            w_users_id2 = [(value[0], value[1]) for value in w_users_id1.values()]
                            # update global weights
                            user_global = self._aggregate(w_users_id2)
                            self.model_trainer.set_model_params(user_global)

                    elif self.args.method_name == 'HierHFL':
                        if self.args.attack_type == 'full_trim':
                            param_listsnew = self.full_trim(param_lists, self.args.malicious_m)
                            for i in range(len(param_listsnew)):
                                w_local = self.param_to_w_local(param_listsnew[i], self.model_trainer, self.args.lr)
                                w_users.append(w_local)
                            user_global = self._aggregate(w_users)
                            self.model_trainer.set_model_params(user_global)
                        elif self.args.attack_type == 'krum_attack':
                            b = int((len(malicious_clients) / self.args.client_num_in_total) * len(self.client_indexes))
                            param_listsnew = self.dir_partial_krum_lambda(param_lists, b, epsilon=0.01)
                            for i in range(len(param_listsnew)):
                                w_local = self.param_to_w_local(param_listsnew[i], self.model_trainer, self.args.lr)
                                w_users.append(w_local)
                            user_global = self._aggregate(w_users)
                            self.model_trainer.set_model_params(user_global)
                        else:
                            user_global = self._aggregate(w_users)
                            self.model_trainer.set_model_params(user_global)

                ES_global_list.append(user_global)
            if self.args.method_name == 'JSHFL':  # ours
                weight = self.compute_acc_weight(round_idx, ES_global_list)
                CS_global = self._aggregate_CS_acc_weight(ES_global_list, weight)
                # CS_global = self._aggregate_CS(ES_global)   # for JSHFL_avg
                self.model_trainer.set_model_params(CS_global)
            else:
                CS_global = self._aggregate_CS(ES_global_list)
                self.model_trainer.set_model_params(CS_global)
            
            # test at each cloud round
            if round_idx == self.args.comm_round - 1:
                acc_test = self._local_test_on_all_clients(round_idx, self.model_trainer)
                El_result.append(acc_test)
            else:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    acc_test = self._local_test_on_all_clients(round_idx, self.model_trainer)
                    El_result.append(acc_test)
            print('El_result-------', El_result)
            self.save_test_acc(El_result)
        

    def compute_kl_divergence(self, w_global, w_locals):
        w_locals_pram = []
        for client_data, client_model_update in w_locals:
            if isinstance(client_model_update,
                          collections.OrderedDict) and client_model_update.keys() == w_global.keys():
                w_locals_pram.append((client_data, client_model_update))
        w_locals_KL = []
        for client_data, client_model_update in w_locals_pram:
            kl_divergence = 0.0
            for global_param, client_param in zip(w_global.values(), client_model_update.values()):
                global_param_flatten = global_param.flatten()
                client_param_flatten = client_param.flatten()
                kl_divergence += torch.nn.functional.kl_div(torch.log_softmax(client_param_flatten, dim=0),
                                                            torch.softmax(global_param_flatten, dim=0),
                                                            reduction='batchmean').to(self.device)
            w_locals_KL.append((client_data, kl_divergence.item()))
        return w_locals_KL
    '''
    f: number of compromised worker devices
    b: trim parameter
    '''
    def score(self, gradient, v, f):
        num_neighbours = v.shape[1] - 2 - f
        sorted_distance = torch.square(v - gradient).sum(dim=0).sort()[0]
        return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


    def compute_js_divergence(self, w_global, w_locals):
        w_locals_js = []
        for client_data, client_model_update in w_locals:
            if isinstance(client_model_update,
                          collections.OrderedDict) and client_model_update.keys() == w_global.keys():
                js_divergence = 0.0
                for global_param, client_param in zip(w_global.values(), client_model_update.values()):
                    global_param_flatten = global_param.flatten()
                    client_param_flatten = client_param.flatten()
                    m_param_flatten = 0.5 * (
                                torch.softmax(global_param_flatten.float(), dim=0).to(self.device) + torch.softmax(client_param_flatten.float().to(self.device), dim=0))

                    kl_divergence_pm = torch.nn.functional.kl_div(torch.log_softmax(client_param_flatten.float().to(self.device), dim=0),
                                                                  m_param_flatten,
                                                                  reduction='batchmean')
                    kl_divergence_mp = torch.nn.functional.kl_div(torch.log_softmax(global_param_flatten.float().to(self.device) , dim=0),
                                                                  m_param_flatten,
                                                                  reduction='batchmean')
                    js_divergence += 0.5 * (kl_divergence_pm + kl_divergence_mp)

                w_locals_js.append((client_data, js_divergence.item()))
        return w_locals_js


    def compute_threshold(self, calculate_kl_diverg, client_indexes):
        calculate_client_kl = {}
        for i in range(len(client_indexes)):
            calculate_client_kl[client_indexes[i]] = calculate_kl_diverg[i][1]
        calculate_client_kl1_sorted = sorted(calculate_client_kl.items(), key=lambda x: x[1], reverse=True)

        first_magnitude = math.floor(math.log10(abs(calculate_client_kl1_sorted[0][1])))
        threshold = calculate_client_kl1_sorted[0][1]

        if all(math.floor(math.log10(abs(value))) == first_magnitude for _, value in calculate_client_kl1_sorted):
            threshold = calculate_client_kl1_sorted[0][1]
        else:
            threshold = 10 ** first_magnitude
        return threshold

    def compute_threshold_js(self, calculate_js_diverg, client_indexes):
        js_values = [calculate_js_diverg[i][1] for i in range(len(client_indexes))]
        median_value = np.median(js_values)
        mean_value = np.mean(js_values)
        threshold = median_value + (mean_value - median_value) * 0.5  # parameter 0.5 is used to balance the median and mean

        return threshold



    def param_to_w_local(self, param_client, net, lr):  # change param to w_local

        averaged_params = copy.deepcopy(net.get_model_params())
        idx = 0
        for k in averaged_params.keys():
            if averaged_params[k].numel() == 0 or 'running_mean' in k or  'running_var' in k or 'num_batches_tracked' in k:
                continue
            averaged_params[k] = averaged_params[k].to(self.device) - lr * param_client[
                                                                           idx:(idx + averaged_params[
                                                                               k].numel())].reshape(
                averaged_params[k].shape).to(self.device)
            idx += averaged_params[k].numel()

        return averaged_params

    def compute_acc_weight(self, round_idx, ES_global):
        ES_global_acc = []
        for model_param in ES_global:
            self.model_trainer.set_model_params(model_param)
            acc_test1 = self._local_test_on_all_clients(round_idx, self.model_trainer)
            ES_global_acc.append(acc_test1)

        exp_acc = np.exp(ES_global_acc)
        normalized_acc = exp_acc / np.sum(exp_acc)
        return normalized_acc

    def compute_acc_weight_SFW(self, round_idx, ES_global):
        ES_global_acc = []
        for model_param in ES_global:
            self.model_trainer.set_model_params(model_param)
            acc_test1 = self._local_test_on_all_clients(round_idx, self.model_trainer)
            ES_global_acc.append(acc_test1)

        exp_acc = np.exp(ES_global_acc)

        excluded_idx = [idx for idx, acc in enumerate(ES_global_acc) if acc < 0.1]

        if len(excluded_idx) == len(ES_global_acc):

            normalized_acc = exp_acc / np.sum(exp_acc)
        else:

            for idx in excluded_idx:
                exp_acc[idx] = 0.0

            total_acc = np.sum(exp_acc)

            normalized_acc = exp_acc / total_acc
        return normalized_acc

    def save_list_to_file(self, lst, filename):
        with open(filename, 'w') as file:
            for item in lst:
                file.write(str(item) + '\n')


    def _aggregate_CS(self, w_locals):
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


    def _aggregate_CS_acc_weight(self, w_locals, weight):
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
        
    def compute_log_norm(self, w_locals):
        for w_local in w_locals:
            for key, value in w_local[1].items():
                norm_weight = torch.log(1 + torch.norm(value, p=2)) * value / torch.norm(value, p=2)
                w_local[1][key] = norm_weight
        return w_locals

    def compute_log_norm_g(self, w_locals):
        for w_local in w_locals:
            for key, value in w_local.items():
                norm_weight = torch.log(1 + torch.norm(value, p=2)) * value / torch.norm(value, p=2)
                w_local[key] = norm_weight
        return w_locals
        
    def compute_log_norm_param(self, w_local):
        epsilon = 1e-8
        norm = torch.norm(w_local) + epsilon
        norm_weight = torch.log(1 + norm) * w_local / norm
        return norm_weight


    def compute_trustscore(self, simi_client, round_id):
        global cumulative_scores
        if round_id == 0:
            cumulative_scores = simi_client
        else:
            cumulative_scores = [cumulative + current for cumulative, current in
                                 zip(cumulative_scores, simi_client)]
        trust_scores = [cumulative / (round_id + 1) for cumulative in cumulative_scores]
        return trust_scores

    def com_reputation(self, trust, round_id):
        global reputation_new
        reputation_new = [0] * len(trust)
        if round_id == 0:
            for i in range(len(trust)):
                repu = []
                reputation_new[i] = (1 / math.tanh(2)) * 1 * math.tanh(1 + trust[i])
            return reputation_new
        repu = self.com_reputation(trust, round_id - 1)
        for i in range(len(trust)):
            reputation_new[i] = (1 / math.tanh(2)) * repu[i] * math.tanh(1 + trust[i])
        return reputation_new


    def save_test_acc(self, test_acc):
        le = len(test_acc)
        index = [i for i in range(le)]
        data = {'round_id': index, 'test_acc': test_acc}
        df = pd.DataFrame(data)
        method_name = self.args.method_name
        attack_type = self.args.attack_type
        m = self.args.malicious_m
        df.to_csv('./csv/'+ str(method_name) + '_' + str(attack_type) + '_Moving_iid_' + '_M_' + str(m) + 'fashionmnist_.csv', index=None)


    def allocate_clients_distance(self, model, num_clients, round_id):
        client_indexes = []
        distance_range = [1, 5]   # distance
        num_servers = 3  # number of edge servers
        #task_comb = [f'resnet20_{i}' for i in range(num_servers)]  # for CIFAR10 dataset
        task_comb = [f'cnn_{i}' for i in range(num_servers)]
        dis_dict = {}
        allocation_result = {}
        random.seed(round_id)
        for client in range(num_clients):
            dis_list = random.sample(range(distance_range[0], distance_range[1] + 1), num_servers)
            dis_dict[client] = dis_list
        for server in range(num_servers):
            allocation_result[task_comb[server]] = []
        for client, distances in dis_dict.items():
            min_distance = min(distances)
            closest_servers = [task_comb[i] for i, distance in enumerate(distances) if distance == min_distance]
            assigned_server = random.choice(closest_servers)
            allocation_result[assigned_server].append(client)
        print(allocation_result)
        for m in allocation_result.keys():
            if m == model:
                client_indexes = allocation_result[m]
        return client_indexes


    def single_minded(self):
        #task_comb = ['resnet_0', 'resnet_1', 'resnet_2'] # for CIDAR10
        task_comb = ['cnn_0', 'cnn_1', 'cnn_2']  # for mnist and fashion-mnist, cnn
        task_preference_client_all = []
        for i in range(self.args.client_num_in_total):
            np.random.seed(i)
            task_preference_per_client = np.random.choice(task_comb, 1)
            task_preference_client_all.append(task_preference_per_client.tolist())
        return task_preference_client_all

    def single_minded1(self):
        # task_comb = ['resnet_0', 'resnet_1', 'resnet_2'] # for CIDAR10
        task_comb = ['cnn_0', 'cnn_1', 'cnn_2']  # for mnist and fashion-mnist, cnn
        task_preference_client_all1 = dict()
        for i in range(self.args.client_num_in_total):
            np.random.seed()
            task_preference_per_client = np.random.choice(task_comb, 1)
            model=task_preference_per_client.tolist()[0]
            if model not in task_preference_client_all1:
                task_preference_client_all1[model] = [i]
            else:
                task_preference_client_all1[model].append(i)

        return task_preference_client_all1

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

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
                    averaged_params[k] = local_model_params[k].to(self.device) * w
                else:
                    averaged_params[k] += local_model_params[k].to(self.device) * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx, model):
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

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
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

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats)
        return test_acc

    def _local_test_on_validation_set(self, round_idx):

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
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)


    def create_model(self, model_name, output_dim):
        logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
        model = None
        if "_" in model_name:
            model_name = model_name.split("_")[0]
        if model_name == "cnn":
            torch.manual_seed(0)
            logging.info("CNN + MNIST")
            model = CNN_DropOut2(True)

        elif model_name == "resnet20":
            logging.info("resnet20+cifar10")
            model = resnet20cifar10()

        elif model_name == "rnn":
            logging.info("RNN + mnist")
            model = RNN_OriginalFedAvg1()

        elif model_name == "resnet18_gn1":
            logging.info("ResNet18_GN1 + mnist")
            model = resnet18mnist(n_class=10)

        return MyModelTrainerCLS(model)

    # ==========LF=================
    def lable_fliiping_attack(self, malicious_clients, train_data_local_dict, client_idx, client, user_global, w_users):
        malicious_m_rate_list = self.get_malicious_m_rate(self.args.client_num_in_total)
        data_combine = []
        if client_idx not in malicious_clients:
            return train_data_local_dict
        for batch_idx, (x, labels) in enumerate(train_data_local_dict):
            data_combine.append((x, self.deal_label1(labels, malicious_m_rate_list[client_idx])))
        train_data = data_combine
        client.update_local_dataset(client_idx, train_data, self.test_data_local_dict[client_idx],
                                    self.train_data_local_num_dict[client_idx])
        w = client.train(copy.deepcopy(user_global), self.model_trainer)
        w_users.append((client.get_sample_number(), copy.deepcopy(w)))
        return w_users

    def label_fliiping_attack(self, malicious_clients, train_data_local_dict, client_idx, client, user_global, w_users):
        malicious_m_rate_list = self.get_malicious_m_rate(self.args.client_num_in_total)
        data_combine = []
        if client_idx not in malicious_clients:
            return train_data_local_dict
        for batch_idx, (x, labels) in enumerate(train_data_local_dict):
            data_combine.append((x, self.deal_label1(labels, malicious_m_rate_list[client_idx])))
        train_data = data_combine
        return data_combine

    def reset_param_client_Gaussian(self, w_lo):
        w_list = []
        averaged_params1 = w_lo
        for k in averaged_params1.keys():
            y = averaged_params1[k].shape
            w_list.append(torch.normal(0, 200, size=y).to(self.device))
        return w_list

    def reset_local_params(self, w_lo):
        averaged_params1 = w_lo
        for k in averaged_params1.keys():
            averaged_params1[k] = averaged_params1[k].float()
            averaged_params1[k] = torch.nn.init.normal_(averaged_params1[k], mean=0, std=200)
        return averaged_params1

    def full_trim(self, v, f):

        # first compute the statistics
        vi_shape = v[0].shape
        v_tran = torch.cat(v, dim=1)
        maximum_dim = torch.max(v_tran, axis=1).values.reshape(vi_shape)
        minimum_dim = torch.min(v_tran, axis=1).values.reshape(vi_shape)
        direction = torch.sign(torch.sum(torch.cat(v, dim=1), axis=-1, keepdim=True))
        directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

        for i in range(f):
            # apply attack to compromised worker devices with randomness
            random_12 = 2
            v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
        return v

    def dir_full_krum_lambda(self, v, f, epsilon=0.01):
        if len(v) <= 2:
            for i in range(f):
                v[i] = -v[i]
            return v
        vi_shape = v[0].shape
        v_tran = torch.transpose(torch.cat(v, dim=1), 0, 1).clone()
        _, original_dir = self.krum_minid_para(v, f)
        original_dir = original_dir.reshape(vi_shape)

        lamda = 1.
        f = len(v)
        for i in range(f):
            v[i] = -lamda * torch.sign(original_dir)
        min_idx, _ = self.krum_minid_para(v, f)
        stop_threshold = 1e-5
        while min_idx >= f and lamda > stop_threshold:
            lamda = lamda / 2
            for i in range(f):
                v[i] = -lamda * torch.sign(original_dir)
            min_idx, _ = self.krum_minid_para(v, f)

        print('chosen lambda:', lamda)
        v[0] = -lamda * torch.sign(original_dir)
        for i in range(1, f):
            random_raw = torch.rand(vi_shape) - 0.5
            random_norm = torch.rand(1).item() * epsilon
            randomness = random_raw * random_norm / torch.norm(random_raw)
            v[i] = -lamda * torch.sign(original_dir).to(self.device) + randomness.to(self.device)
        return v

    def dir_partial_krum_lambda(self, v, f, epsilon=0.01):
        vi_shape = v[0].shape

        v_tran = torch.transpose(torch.cat(v, dim=1), 0, 1)[:f].clone()
        original_dir = torch.mean(v_tran, dim=0).view(vi_shape)
        v_attack_number = 1

        while (v_attack_number < f):
            lamda = 1.0
            v_simulation = [each_v.clone() for each_v in v[:f]]

            for i in range(v_attack_number):
                v_simulation.append(-lamda * torch.sign(original_dir))

            min_idx, _ = self.krum_minid_para(v_simulation, v_attack_number)

            stop_threshold = 0.00002
            while (min_idx < f and lamda > stop_threshold):
                lamda = lamda / 2
                for i in range(f, f + v_attack_number):
                    v_simulation[i] = -lamda * torch.sign(original_dir)
                min_idx, _ = self.krum_minid_para(v_simulation, v_attack_number)
            v_attack_number += 1

            if min_idx >= f:
                break
        print('chosen lambda:', lamda)
        v[0] = -lamda * torch.sign(original_dir)
        for i in range(1, f):
            random_raw = torch.rand(vi_shape) - 0.5
            random_norm = torch.rand(1).item() * epsilon
            randomness = random_raw * random_norm / torch.norm(random_raw)
            v[i] = -lamda * torch.sign(original_dir)
        return v

    def krum_minid_para(self, v, f):
        if len(v) - f - 2 <= 0:
            f = len(v) - 3
        if len(v[0].shape) > 1:
            v_tran = torch.cat(v, dim=1)
        else:
            v_tran = torch.stack(v)
        scores = torch.tensor([self.score_krum(gradient, v_tran, f) for gradient in v])
        # Find the index of the minimum score
        min_idx = torch.argmin(scores, dim=0).item()
        # Reshape the selected tensor to 1D
        krum_tensor = v[min_idx].reshape(-1)
        return min_idx, krum_tensor

    def score_krum(self, gradient, v, f):
        num_neighbours = v.shape[1] - 2 - f
        sorted_distance = torch.square(v - gradient).sum(dim=0).sort()[0]
        return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


