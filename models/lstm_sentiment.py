import torch
import torch.nn as nn
import json
from utils.utils import count_parameters, split_punctuation
import os


class LSTMSentimentAnalysis(nn.Module):

    def __init__(self, wl_vocab_size, cl_vocab_size, wl_embedding_size, cl_embedding_size, hidden_size, output_size,
                 num_layers=3, dropout=0, bi=True, last_state=True, padding_idx=None):
        """
        A biLSTM model for Sentiment Analysis. The input will be passed through a LSTM to make a hidden feature. Then
        it will be passed through a Linear classifier.
        :param wl_vocab_size: The size of the word level vocabulary.
        :param cl_vocab_size: The size of the character level vocabulary.
        :param wl_embedding_size: The size of the word level embedding.
        :param cl_embedding_size: The size of the character level embedding.
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
        self.wl_embedding = nn.Embedding(num_embeddings=wl_vocab_size,
                                         embedding_dim=wl_embedding_size,
                                         padding_idx=padding_idx,
                                         max_norm=1)
        self.cl_embedding = CharLevelEmbedding(vocab_size=cl_vocab_size,
                                               embedding_dim=cl_embedding_size)
        self.lstm = nn.LSTM(input_size=wl_embedding_size + cl_embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bi,
                            batch_first=True)
        self.classifier = nn.Linear(in_features=(1+bi)*hidden_size,
                                    out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.last_state = last_state
        self.bi = bi

    def forward(self, words, chars):
        if len(words.shape) == 1:
            words = words.unsqueeze(0)
        if len(chars.shape) == 2:
            chars = chars.unsqueeze(0)
        wl_embeddings = self.wl_embedding(words)
        cl_embeddings = self.cl_embedding(chars)
        embeddings = torch.cat([wl_embeddings, cl_embeddings], -1)
        lstm_out, lstm_hidden = self.lstm(embeddings)
        if self.last_state:
            if self.bi:
                # Get the last state of the forward output and the "first" state of the backward output
                embedding_dim = lstm_out.shape[2] // 2
                forward_feature = lstm_out[:, -1, :embedding_dim].clone()
                backward_feature = lstm_out[:, 0, embedding_dim:].clone()
                feature = torch.cat([forward_feature, backward_feature], dim=-1)
            else:
                feature = lstm_out[:, -1, :]
        else:
            feature = torch.mean(lstm_out, dim=-2)
        out = self.classifier(feature)
        # out = self.softmax(out)
        return out


class CharLevelEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout=0, bi=False, padding_idx=0):
        super(CharLevelEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=padding_idx,
                                            max_norm=1)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=1,
                            dropout=dropout,
                            bidirectional=bi,
                            batch_first=True)
        self.linear = nn.Linear(in_features=(1 + bi) * embedding_dim,
                                out_features=embedding_dim)
        self.bi = bi

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        out = []
        for word_id in range(x.shape[-2]):
            word = x[:, word_id, :]
            input_embeddings = self.embedding_layer(word)
            lstm_out, lstm_hidden = self.lstm(input_embeddings)
            if self.bi:
                # Get the last state of the forward output and the "first" state of the backward output
                embedding_dim = lstm_out.shape[2] // 2
                forward_feature = lstm_out[:, -1, :embedding_dim].clone()
                backward_feature = lstm_out[:, 0, embedding_dim:].clone()
                feature = torch.cat([forward_feature, backward_feature], dim=-1)
                word_embedding = self.linear(feature)
            else:
                # feature = lstm_out[:, -1, :]
                word_embedding = lstm_out[:, -1, :]
            # out = self.linear(feature)
            word_embedding = word_embedding.unsqueeze(-2)
            out.append(word_embedding)
        out = torch.cat(out, -2)
        return out


# Unused
class JsonTokenizer:

    def __init__(self, tokenization_path):
        with open(tokenization_path, 'r') as f:
            tokenization = json.load(f)
        self.vocab_size = tokenization['vocab_size']
        self.vocabulary = tokenization['vocabulary']

    def tokenize(self, docs):
        if isinstance(docs, str):
            return self._tokenize_doc(docs)
        elif isinstance(docs, tuple) or isinstance(docs, list):
            tokenizations = []
            for doc in docs:
                tokenizations.append(self._tokenize_doc(doc))
            return torch.tensor(tokenizations, dtype=torch.long)

    def _tokenize_doc(self, doc):
        doc = split_punctuation(doc)
        doc = doc.split()
        tokens = []
        for word in doc:
            if word in self.vocabulary.keys():
                tokens.append(self.vocabulary[doc])
            else:
                tokens.append(self.vocabulary['<unk>'])
        return torch.tensor(tokens, dtype=torch.long)


if __name__ == '__main__':
    # biLSTM:
    # forward output: [:, :, :hidden_size]
    # backward output: [:, :, hidden_size:]
    model = LSTMSentimentAnalysis(wl_vocab_size=10,
                                  cl_vocab_size=10,
                                  wl_embedding_size=128,
                                  cl_embedding_size=100,
                                  hidden_size=256,
                                  output_size=2
                                  )
    # model = CharLevelEmbedding(vocab_size=10,
    #                            embedding_dim=100,
    #                            bi=True
    #                            )
    total_params, table = count_parameters(model)
    print(table)
    print(f"Total Trainable Params: {total_params}")
    words = torch.tensor([[1, 2, 4, 5, 6], [0, 2, 5, 6, 9], [0, 2, 5, 6, 9], [1, 2, 4, 5, 6]])
    chars = torch.tensor([
        [[1, 2, 4], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5]],
        [[1, 2, 4], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5]],
        [[1, 2, 4], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5]],
        [[1, 2, 4], [0, 2, 5], [0, 2, 5], [0, 2, 5], [0, 2, 5]]
    ])
    output = model(words, chars)
    print(torch.exp(output))
    print(output.shape)
