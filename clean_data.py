import pandas as pd
from transformers import BartTokenizer
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

dataframe = pd.read_csv('/home/chaunm/PycharmProjects/datasets/IMDB Dataset.csv')


def transform_label(label):
    return 1 if label == 'positive' else 0


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
    no_punct = rm_punct2(no_html)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonasci = rm_nonascii(no_whitespaces)
    no_emoji = rm_emoji(no_nonasci)
    spell_corrected = spell_correction(no_emoji)
    return spell_corrected


dataframe['label'] = dataframe['sentiment'].apply(transform_label)
dataframe['clean'] = dataframe['review'].apply(clean_pipeline)
dataframe_pos = dataframe[dataframe['label'] == 1]
dataframe_neg = dataframe[dataframe['label'] == 0]
split_ratio = 0.8
dataframe_pos_train = dataframe_pos.sample(frac=split_ratio, random_state=200)
dataframe_pos_test = dataframe_pos.drop(dataframe_pos_train.index)
dataframe_neg_train = dataframe_neg.sample(frac=split_ratio, random_state=200)
dataframe_neg_test = dataframe_neg.drop(dataframe_neg_train.index)

dataframe_train = pd.concat([dataframe_neg_train, dataframe_pos_train], ignore_index=True, axis=0)
dataframe_test = pd.concat([dataframe_neg_test, dataframe_pos_test], ignore_index=True, axis=0)
dataframe_train = dataframe_train.sample(frac=1)  # shuffle
dataframe_test = dataframe_test.sample(frac=1)  # shuffle
print(len(dataframe_train[dataframe_train['label'] == 1]))
print(len(dataframe_test[dataframe_test['label'] == 1]))
print(len(dataframe_train[dataframe_train['label'] == 0]))
print(len(dataframe_test[dataframe_test['label'] == 0]))
dataframe_train[['clean', 'label']].to_csv("./datasets/imdb_train.csv", index=False, header=True)
dataframe_test[['clean', 'label']].to_csv("./datasets/imdb_test.csv", index=False, header=True)
