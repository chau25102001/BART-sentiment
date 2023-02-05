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

    def __init__(self, csv_path, vocabulary_path, max_seq_length, max_word_length):
        super(NaiveTokenizedDataset, self).__init__(csv_path)
        with open(vocabulary_path, 'r') as f:
            tokenization = json.load(f)
        self.vocab_size = tokenization['vocab_size']
        self.cl_vocab_size = tokenization['char_level_vocab_size']
        self.vocabulary = tokenization['vocabulary']
        self.cl_vocabulary = tokenization['char_level_vocabulary']
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

    def __getitem__(self, item):
        text = self.text[item]
        label = int(self.label[item])
        text = split_punctuation(text)
        text = text.split()
        tokens = []
        characters = []
        for word in text:
            # word level token
            if word in self.vocabulary.keys():
                tokens.append(self.vocabulary[word])
            else:
                tokens.append(self.vocabulary['<unk>'])
            # character level token
            chars = list(word)
            chars_token = [self.cl_vocabulary[char]
                           if char in self.cl_vocabulary.keys() else self.cl_vocabulary['<unk>']
                           for char in chars]
            if len(chars_token) > self.max_word_length:
                chars_token = chars_token[:self.max_word_length]
            while len(chars_token) < self.max_word_length:
                chars_token.append(self.cl_vocabulary['<pad>'])
            characters.append(chars_token)
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            characters = characters[:self.max_seq_length]
        while len(tokens) < self.max_seq_length:
            tokens.append(self.vocabulary['<pad>'])
            characters.append([self.cl_vocabulary['<pad>'] for i in range(self.max_word_length)])
        return (
                (torch.tensor(tokens, dtype=torch.int),
                 torch.tensor(characters, dtype=torch.int)),
                torch.tensor(label, dtype=torch.long)
                )
