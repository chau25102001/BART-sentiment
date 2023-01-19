import os

import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def transform_label(label):
    return 1 if label == 'pos' else 0


def rm_link(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


# handle case like "shut up okay?Im only 10 years old"
# become "shut up okay Im only 10 years old"
def rm_punct2(text):
    # return re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
    return re.sub(r'[\#\$\%\&\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)


def rm_html(text):
    return re.sub(r'<[^>]+>', '', text)


def rm_number(text):
    return re.sub(r'\d+', '', text)


def rm_whitespaces(text):
    return re.sub(r' +', ' ', text)


def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)


def rm_emoji(text):
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)


def spell_correction(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


def clean_pipeline(text):
    no_link = rm_link(text)
    no_html = rm_html(no_link)
    no_emoji = rm_emoji(no_html)
    no_punct = rm_punct2(no_emoji)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonasci = rm_nonascii(no_whitespaces)
    # no_emoji = rm_emoji(no_nonasci)
    spell_corrected = spell_correction(no_nonasci)
    return spell_corrected


def clean_dataframe(path):
    assert os.path.isfile(path), "data file does not exist, check it"
    root = os.path.dirname(path)
    name = path.split("/")[-1].split(".")[0]
    dataframe = pd.read_csv(path)
    dataframe['label'] = dataframe['sentiment'].apply(transform_label)
    dataframe['clean'] = dataframe['text'].apply(clean_pipeline)
    dataframe[['clean', 'label']].to_csv(os.path.join(root, name + "_clean.csv"), index=False, header=True)


# clean_dataframe('./datasets/train.csv')
# clean_dataframe('./datasets/test.csv')
data_cleaned = pd.read_csv("./datasets/train_clean.csv")
print(data_cleaned['clean'][0])