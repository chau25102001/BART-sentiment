import torch
import torch.nn as nn
from utils.utils import count_parameters


class LSTMSentimentAnalysis(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers=3, dropout=0.2, bi=True,
                 last_state=True, padding_idx=None):
        """
        A biLSTM model for Sentiment Analysis. The input will be passed through a LSTM to make a hidden feature. Then
        it will be passed through a Linear classifier.
        :param vocab_size: The vocabulary size.
        :param embedding_size: The embedding size of each word.
        :param hidden_size: The size of the LSTM hidden state.
        :param output_size: The size of the output tensor (the number of sentiment classes).
        :param num_layers: The number of LSTM layers.
        :param dropout: The dropout rate of LSTM layers.
        :param bi: To use biLSTM or not.
        :param last_state: If True, only the output of the last LSTM cell is used. Else, the output of LSTM cells will
                           be averaged.
        :param padding_idx: The padding_idx to be passed into nn.Embedding. This is the index of <pad> token in the
                            vocabulary.
        """
        super(LSTMSentimentAnalysis, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx,
                                      max_norm=1)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bi,
                            batch_first=True)
        self.classifier = nn.Linear(in_features=(1+bi)*hidden_size,
                                    out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.last_state = last_state

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        embeddings = self.embedding(x)
        lstm_out, lstm_hidden = self.lstm(embeddings)
        if self.last_state:
            feature = lstm_out[:, -1, :]
        else:
            feature = torch.mean(lstm_out, dim=-2)
        out = self.classifier(feature)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    model = LSTMSentimentAnalysis(vocab_size=10,
                                  embedding_size=128,
                                  hidden_size=256,
                                  output_size=2)
    total_params, table = count_parameters(model)
    print(table)
    print(f"Total Trainable Params: {total_params}")
    test = torch.tensor([[1, 2, 4, 5, 6], [0, 2, 5, 6, 9]])
    output = model(test)
    print(torch.exp(output))
    print(output.shape)
