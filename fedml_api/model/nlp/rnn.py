import torch.nn as nn
import torch
import torch.nn.functional as F

class RNN_OriginalFedAvg(nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
      This replicates the model structure in the paper:
      Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629
      This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_OriginalFedAvg, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        # For fed_shakespeare
        # output = self.fc(lstm_out[:,:])
        # output = torch.transpose(output, 1, 2)
        return output


class RNN_OriginalFedAvg1(nn.Module):
    def __init__(self, INPUT_SIZE=28, HIDDEN_SIZE=128, NUM_CLASSES = 10):
        super(RNN_OriginalFedAvg1, self).__init__()

        self.rnn = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=INPUT_SIZE,  # 图片每行的数据像素点
            hidden_size=HIDDEN_SIZE,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(HIDDEN_SIZE*28, NUM_CLASSES)  # 输出层，全连接

    def forward(self, x):
        x = x.view((-1, 28, 28))
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        hidden2one_res = []
        for i in range(28):
            hidden2one_res.append(r_out[:, i, :])
        hidden2one_res = torch.cat(hidden2one_res, dim=1)
        res = self.out(hidden2one_res)
        res = F.softmax(res, dim=1)
        # print('res------------------------------', res)
        return res

        # 这个地方选择lstm_output[-1]，也就是相当于最后一个输出，因为其实每一个cell（相当于图中的A）都会有输出，但是我们只关心最后一个
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        # out = self.out(r_out[:, -1, :])  # torch.Size([64, 28, 64])-> torch.Size([64, 10])
        # return out

# class RNN_OriginalFedAvg1(nn.Module):
#     def __init__(self, input_size: int = 28, hidden_size: int = 64, num_layers: int = 4, output_size: int = 10):
#         super(RNN_OriginalFedAvg1, self).__init__()
#         self.main = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#         )
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = x.view((-1, 28, 28))
#         output, (hn, cn) = self.main(x, None)
#         result = self.fc(output[:, -1, :])
#         return result


class RNN_StackOverFlow(nn.Module):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).
      This replicates the model structure in the paper:
      "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Table 9
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, vocab_size=10000,
                 num_oov_buckets=1,
                 embedding_size=96,
                 latent_size=670,
                 num_layers=1):
        super(RNN_StackOverFlow, self).__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size, embedding_dim=embedding_size,
                                            padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers)
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    def forward(self, input_seq, hidden_state = None):
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out[:,:])
        output = self.fc2(fc1_output)
        output = torch.transpose(output, 1, 2)
        return output
