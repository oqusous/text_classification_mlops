import json
import io
import pandas as pd
from sklearn import preprocessing
from datasets import Dataset as HFDS
from transformers import RobertaTokenizerFast
from tqdm import tqdm
import numpy as np
import spacy
import tensorflow as tf
from shutil import copyfile

def data_preprocess(df, dataset_type, le):
    df_orig_row_num=df.shape[0]
    if dataset_type=='val' or dataset_type=='train' or dataset_type=='test_target':
        df['text'].fillna(value='Empty' ,axis=0, inplace=True)
        df['label'].fillna(value=0 ,axis=0, inplace=True)
    elif dataset_type=='test':
        df['text'].fillna(value='Empty' ,axis=0, inplace=True)
    df_percent_na_dropped = str(round((df_orig_row_num-df.shape[0])/df_orig_row_num, 1)*100)+'%'
    print("percentage of rows dropped due to NaNs in text and label cols:", df_percent_na_dropped)
    # df.drop_duplicates(subset='id', keep='first', inplace=True, ignore_index=False)
    # df_percent_na_dropped = str(round((df_orig_row_num-df.shape[0])/df_orig_row_num, 1)*100)+'%'
    # print("percentage of rows dropped due to duplicates in id col:", df_percent_na_dropped)
    df['text'] = df['text'].astype(str)
    if dataset_type=='train':
        le_train = preprocessing.LabelEncoder()
        df['label']= le_train.fit_transform(list(df['label']))
        classes = dict(df['label'].value_counts())
        print("class count:\n", classes)
        print("number of classes detected in 'label' col:", str(len(set(df['label']))))
        return df, le_train
    elif dataset_type=='test_target' or dataset_type=='val':
        df['label']= le.transform(list(df['label']))
        return df
    else:
        return df

def X_y(df, dataset_type):
    X = df[['text']]
    if dataset_type != 'test':
        y = df['label']
        return X, y
    return X

def tokenizer_spacy(df, col):
    nlp = spacy.load('en_core_web_sm')
    train_tokenized = []
    train_list = df[col].values.tolist()
    for answer in tqdm(train_list):
        tokens_joined = ''
        try:
            nlpd = nlp(answer)
            for t in nlpd:
                if (t.is_alpha) and (not t.is_stop) and (not t.is_punct) and (not t.is_space):
                    tokens_joined += ' '+t.text.lower()
            train_tokenized.append(tokens_joined.strip())
        except:
            train_tokenized.append('Not processable')
#   pickle.dump(train_tokenized, open('data/'+filename+'.pkl', 'wb'))
    return train_tokenized

def tokenize_pad(X_train_tok, X_val_tok, X_test_tok):
    max_len=max([len(x.split(" ")) for x in X_train_tok])
    maxlen = max_len if max_len<=256 else 256
    tok = tf.keras.preprocessing.text.Tokenizer()
    tok.fit_on_texts(X_train_tok)
    tok_json = tok.to_json()
    with io.open('model/tf_nn_arch/tf_tok.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tok_json, ensure_ascii=False))
        f.close()
    word_index = tok.word_index
    train_seq = tok.texts_to_sequences(X_train_tok)
    val_seq = tok.texts_to_sequences(X_val_tok)
    test_seq =tok.texts_to_sequences(X_test_tok)
    X_train_tok_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen)
    X_val_tok_padded = tf.keras.preprocessing.sequence.pad_sequences(val_seq, maxlen=maxlen)
    X_test_tok_padded = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=maxlen)
    return X_train_tok_padded, X_val_tok_padded, X_test_tok_padded, word_index, maxlen

def encode_glove(word_index):
    word_emb_w2v = {}
    file_emb = open("glove/glove.6B.300d.txt", encoding="utf-8")
    for emb in tqdm(file_emb):
        array = emb.split()
        word = str(array[0])
        vector = np.asarray(array[1:], dtype=np.float32)
        word_emb_w2v[word] = vector
    file_emb.close()

    num_o_words = len(word_index)+1
    print(num_o_words)
    emb_dim = 300 # size of w2v vector
    emb_mtrx = np.zeros((num_o_words, emb_dim))
    for word, i in word_index.items():
        if i > num_o_words:
            continue
        emb_vector = word_emb_w2v.get(word)
        if emb_vector is not None:
            emb_mtrx[i] = emb_vector
    return num_o_words, emb_dim, emb_mtrx

def roberta_tokenize(dataset, dataset_type):
    # dataset = dataset.rename(columns={'label':'label'})
    if dataset_type == 'test':
        dataset = HFDS.from_pandas(dataset[['text']])
    else:
        dataset = HFDS.from_pandas(dataset[['text', 'label']])
    model_checkpoint = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_checkpoint)
    sentence1_key = 'text'
    def preprocess_function(examples,):
        return tokenizer(examples[sentence1_key], truncation=True)
    dataset_en = dataset.map(preprocess_function, batched=True)
    tokenizer.save_pretrained('model/roberta_tok')
    return dataset_en


if __name__ == '__main__':
    pass

# from torchtext.data import Field, TabularDataset, BucketIterator
# import torchtext.legacy
# import torch
# cuda = torch.device('cuda')
# cpu = torch.device('cpu')

# from tqdm import tqdm
# import numpy as np
# import spacy
# # import tensorflow as tf

# def rnn_lstm_tokenize_glove(df_train, df_val, df_test):
#     df_train = df_train[['text', 'label']]
#     df_val = df_val[['text', 'label']]
#     df_test = df_test[['text']]
#     df_train.to_csv('pytorch_tabular/df_train.csv', index=False)
#     df_val.to_csv('pytorch_tabular/df_val.csv', index=False)
#     df_test.to_csv('pytorch_tabular/df_test.csv', index=False)

#     label_field = torchtext.legacy.data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#     text_field = torchtext.legacy.data.Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)

#     fields = [('text', text_field), ('label', label_field)]

#     train, valid, test = torchtext.legacy.data.TabularDataset.splits(path='pytorch_tabular/', train='df_train.csv', validation='df_val.csv', test='df_test.csv',
#                                             format='CSV', fields=fields, skip_header=True)

#     train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)
#     valid_iter = torchtext.legacy.data.BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)
#     test_iter = torchtext.legacy.data.BucketIterator(test, batch_size=16, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)

#     text_field.build_vocab(train, min_freq=2, vectors = "glove.6B.300d")
#     return train_iter, valid_iter, test_iter, text_field


# def rnn_lstm_tokenize(df_train, df_val, df_test):
#     df_train = df_train[['text', 'label']]
#     df_val = df_val[['text', 'label']]
#     df_test = df_test[['text']]
#     df_train.to_csv('pytorch_tabular/df_train.csv', index=False)
#     df_val.to_csv('pytorch_tabular/df_val.csv', index=False)
#     df_test.to_csv('pytorch_tabular/df_test.csv', index=False)

#     label_field = torchtext.legacy.data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#     text_field = torchtext.legacy.data.Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)

#     fields = [('text', text_field), ('label', label_field)]

#     train, valid, test = torchtext.legacy.data.TabularDataset.splits(path='pytorch_tabular/', train='df_train.csv', validation='df_val.csv', test='df_test.csv',
#                                             format='CSV', fields=fields, skip_header=True)

#     train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)
#     valid_iter = torchtext.legacy.data.BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)
#     test_iter = torchtext.legacy.data.BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.text),
#                                 device=cuda, sort=True, sort_within_batch=True)

#     text_field.build_vocab(train, min_freq=2) # vectors = "glove.6B.300d"
#     return train_iter, valid_iter, test_iter, text_field