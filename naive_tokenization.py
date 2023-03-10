import os
import json
import pandas as pd
from utils.utils import split_punctuation
from string import printable


def unique_tokens(doc):
    tokens = doc.split()
    return set(tokens)


def create_vocabulary(csv_path, save_folder='./datasets'):
    tokenization = {"vocabulary": {"<pad>": 0, "<unk>": 1, },
                    "char_level_vocabulary": {"<pad>": 0, "<unk>": 1, },
                    "vocab_size": 2,
                    "char_level_vocab_size": 2}
    dataframe = pd.read_csv(csv_path)
    text = dataframe['clean']
    # Add word tokens to vocabulary
    for sample in text:
        sample = split_punctuation(sample)
        tokens = unique_tokens(sample)
        for token in tokens:
            if token not in tokenization["vocabulary"].keys():
                tokenization["vocabulary"][token] = tokenization['vocab_size']
                tokenization['vocab_size'] += 1
    # Add character tokens to vocabulary
    for char in printable:
        if char not in tokenization['char_level_vocabulary'].keys():
            tokenization['char_level_vocabulary'][char] = tokenization['char_level_vocab_size']
            tokenization['char_level_vocab_size'] += 1
    # tokenization['inverse_vocabulary'] = {v: k for k, v in tokenization['vocabulary'].items()}
    file_name = os.path.split(csv_path)[1]
    file_name = os.path.splitext(file_name)[0]
    with open(os.path.join(save_folder, f'{file_name}_vocabulary.json'), 'w') as f:
        json.dump(tokenization, f, indent=4)
    return tokenization


if __name__ == '__main__':
    # print(punctuation)
    create_vocabulary('./datasets/train_clean.csv')
