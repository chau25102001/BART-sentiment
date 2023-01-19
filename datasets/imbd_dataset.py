import pandas as pd
from torch.utils.data import Dataset
from transformers import BartTokenizer


class IMDBDataset(Dataset):
    def __init__(self, csv_path):
        super(IMDBDataset, self).__init__()
        self.text = None
        self.label = None
        self.csv_path = csv_path
        self.from_csv(csv_path)
        self.max_seq_length = 1021
        self.from_csv(self.csv_path)

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
        if len(text) > self.max_seq_length:
            text = text[:self.max_seq_length]
        label = int(self.label[item])
        return text, label
