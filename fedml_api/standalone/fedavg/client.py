import logging


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number  #

        # logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self. data_size_bid = []  # add by zsh

    # def read_data_size(self):

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data

        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):  # 数组的话，调用进来，然后？？？？
        return self.local_sample_number

    # # 原始的
    # def train(self, w_global):
    #     self.model_trainer.set_model_params(w_global)
    #     self.model_trainer.train(self.local_training_data, self.device, self.args)
    #     weights = self.model_trainer.get_model_params()
    #     return weights

    # 集成学习时的train
    def train(self, w_global, model_trainer):
        model_trainer.set_model_params(w_global)
        model_trainer.train(self.local_training_data, self.device, self.args)
        weights = model_trainer.get_model_params()
        return weights

    # 原始的
    # def local_test(self, b_use_test_dataset):
    #     if b_use_test_dataset:
    #         test_data = self.local_test_data
    #     else:
    #         test_data = self.local_training_data
    #     metrics = self.model_trainer.test(test_data, self.device, self.args)
    #     return metrics

    # 集成学习时local_test
    def local_test(self, b_use_test_dataset, model_trainer):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = model_trainer.test(test_data, self.device, self.args)
        return metrics
