import logging

import numpy as np
import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.dictloss={}
        self.local_grads_sum_epoch = []

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_norm_gradient(self, model):
        grads = []
        idx = 0

        for k in model.parameters():
            grads.append(k.grad)
        return grads

    def sum_grad_client(self, w_zong):
        summed_tensors = []
        for t1 in zip(*w_zong):
            summed_tensors.append(sum(t1))
        return summed_tensors

    def train(self, train_data, device, args):
        lossdict = {}
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        local_grads_sum_epoch = []
        for epoch in range(args.epochs):
            batch_loss = []
            local_grads_sum = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                loss.backward()
                local_grads = self.get_model_norm_gradient(model)
                local_grads_sum.append(local_grads)

                optimizer.step()

                batch_loss.append(loss.item())
            local_grads_sum_epoch.append(self.sum_grad_client(local_grads_sum))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            lossdict[self.id] = sum(epoch_loss) / len(epoch_loss)

        self.dictloss=lossdict

        self.local_grads_sum_epoch = self.sum_grad_client(local_grads_sum_epoch)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

