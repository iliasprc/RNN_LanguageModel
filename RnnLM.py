import torch.nn as nn


class RNNModel(nn.Module):

    def __init__(self, hidden_size, n_layers, dropout, classes):
        super().__init__()
        self.hidden_size = 2*hidden_size
        self.n_layers = n_layers
        self.classes = classes
        self.embeddings = nn.Embedding(self.classes, self.hidden_size)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=False
            )

        self.final_linear = nn.Linear(self.hidden_size, self.classes)

        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.drop(self.embeddings(input))
        output, _ = self.rnn(emb)
        decoded = self.final_linear(output[-1, :, :].view(output.size(1), output.size(2)))
        return decoded, output[-1, :, :]
