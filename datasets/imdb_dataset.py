import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer
from utils.utils import split_punctuation
import json


class IMDBDataset(Dataset):
    def __init__(self, csv_path):
        super(IMDBDataset, self).__init__()
        self.text = None
        self.label = None
        self.csv_path = csv_path
        self.from_csv(csv_path)
        self.max_seq_length = 1021
        self.from_csv(self.csv_path)
        print("data set size: ", len(self.text))

    def __len__(self):
        return len(self.text)

    def from_csv(self, csv_path):
        dataframe = pd.read_csv(csv_path)
        text = dataframe['clean']
        label = dataframe['label']
        self.text = text
        self.label = label

    def __getitem__(self, item):
        text = self.text[item]
        label = int(self.label[item])
        return text, label


class NaiveTokenizedDataset(IMDBDataset):

    def __init__(self, csv_path, vocabulary_path, max_seq_length):
        super(NaiveTokenizedDataset, self).__init__(csv_path)
        with open(vocabulary_path, 'w') as f:
            tokenization = json.load(f)
        self.vocab_size = tokenization['vocab_size']
        self.vocabulary = tokenization['vocabulary']
        self.max_seq_length = max_seq_length

    def __getitem__(self, item):
        text = self.text[item]
        label = int(self.label[item])
        text = split_punctuation(text)
        text = text.split()
        tokens = []
        for word in text:
            if word in self.vocabulary.keys():
                tokens.append(self.vocabulary[word])
            else:
                tokens.append(self.vocabulary['<unk>'])
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        while len(tokens) < self.max_seq_length:
            tokens.append(self.vocabulary['<pad>'])
        return torch.tensor(tokens, dtype=torch.int), torch.tensor(label, dtype=torch.long)
