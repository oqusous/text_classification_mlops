import data_processing
import model_cls
import numpy as np
import pandas as pd
import pickle
from os import listdir
from os.path import join, isfile

def main(train, train_type, val, val_type, test, test_type, num_label, metric_name, epochs):
    df_train, le = data_processing.data_preprocess(train, train_type, None)
    df_val = data_processing.data_preprocess(val, val_type, le)
    df_test = data_processing.data_preprocess(test, test_type, le)
    pickle.dump(le, open('data/data_split/le.pkl', 'wb'))

    X_train, y_train = data_processing.X_y(df_train, train_type)
    X_val, y_val = data_processing.X_y(df_val, val_type)
    if test_type == 'test_target':
        X_test, y_test = data_processing.X_y(df_test, test_type)
    elif test_type == 'test':
        X_test = data_processing.X_y(df_test, test_type)

    X_train_tok = data_processing.tokenizer_spacy(X_train, 'text')
    X_val_tok = data_processing.tokenizer_spacy(X_val, 'text')
    X_test_tok = data_processing.tokenizer_spacy(X_test, 'text')

    X_train_tok_padded, X_val_tok_padded, X_test_tok_padded, word_index, maxlen = data_processing.tokenize_pad(X_train_tok, X_val_tok, X_test_tok)
    
    num_o_words, emb_dim, emb_mtrx = data_processing.encode_glove(word_index)

    pickle.dump(maxlen, open('model/tf_nn_arch/max_len.pkl', 'wb'))
    pickle.dump(emb_dim, open('model/tf_nn_arch/emb_dim.pkl', 'wb'))
    pickle.dump(emb_mtrx, open('model/tf_nn_arch/emb_mtrx.pkl', 'wb'))
    pickle.dump(num_o_words,open('model/tf_nn_arch/num_o_words.pkl', 'wb'))

    model_rnn, history = model_cls.bi_rnn_model(X_train_tok_padded, y_train, X_val_tok_padded, \
                                                y_val, num_o_words, emb_dim, maxlen, emb_mtrx, num_label, epochs, metric_name)
    model_crnn, history_crnn = model_cls.crnn_model(X_train_tok_padded, y_train, X_val_tok_padded, \
                                                y_val, num_o_words, emb_dim, maxlen, emb_mtrx, num_label, epochs, metric_name)
    
    predict_rnn = model_rnn.predict(X_test_tok_padded)
    pickle.dump(predict_rnn, open('test_preds/predict_rnn.pkl', 'wb'))
    predict_crnn = model_crnn.predict(X_test_tok_padded)
    pickle.dump(predict_crnn, open('test_preds/predict_crnn.pkl', 'wb'))
    pd.DataFrame(history.history).to_csv('plots/rnn_plot.csv')
    pd.DataFrame(history_crnn.history).to_csv('plots/crnn_plot.csv')

if __name__ == '__main__':
    onlyfiles = [f for f in listdir('data/') if isfile(join('data/', f))]
    if 'val.csv' not in onlyfiles and 'test.csv' not in onlyfiles and 'train.csv' in onlyfiles:
        all = pd.read_csv("data/train.csv")
        feat_target_train = pickle.load(open('data/data_split/feat_target_train.pkl', 'rb'))
        all.rename(columns={feat_target_train['label']:'label', feat_target_train['feat']:'text'}, inplace=True)
        df_test = all.sample(frac=0.2, random_state=81)
        df_train_val = all.drop(index=list(df_test.index))
        df_train = df_train_val.sample(frac=0.85, random_state=81)
        df_val = all.drop(index=list(df_train.index))

    elif 'val.csv' not in onlyfiles and 'test.csv' in onlyfiles and 'train.csv' in onlyfiles:
        df_train_val = pd.read_csv("data/train.csv")
        feat_target_train = pickle.load(open('data/data_split/feat_target_train.pkl', 'rb'))
        df_train_val.rename(columns={feat_target_train['label']:'label', feat_target_train['feat']:'text'}, inplace=True)
        df_test = pd.read_csv("data/test.csv")
        feat_target_test = pickle.load(open('data/data_split/feat_target_test.pkl', 'rb'))
        if 'label' in feat_target_test.keys():
            df_test.rename(columns={feat_target_test['label']:'label', feat_target_test['feat']:'text'}, inplace=True)
        else:
            df_test.rename(columns={feat_target_test['feat']:'text'}, inplace=True)
        df_train = df_train_val.sample(frac=0.85, random_state=81)
        df_val = df_train_val.drop(index=list(df_train.index))

    elif 'val.csv' in onlyfiles and 'test.csv' not in onlyfiles and 'train.csv' in onlyfiles:
        df_train_test = pd.read_csv("data/train.csv")
        feat_target_train = pickle.load(open('data/data_split/feat_target_train.pkl', 'rb'))
        df_train_test.rename(columns={feat_target_train['label']:'label', feat_target_train['feat']:'text'}, inplace=True)
        df_val = pd.read_csv("data/val.csv")
        feat_target_val = pickle.load(open('data/data_split/feat_target_val.pkl', 'rb'))
        df_val.rename(columns={feat_target_val['label']:'label', feat_target_val['feat']:'text'}, inplace=True)
        df_train = df_train_test.sample(frac=0.80, random_state=81)
        df_test = df_train_test.drop(index=list(df_train.index))

    elif 'val.csv' in onlyfiles and 'test.csv' in onlyfiles and 'train.csv' in onlyfiles:
        df_train = pd.read_csv("data/train.csv")
        feat_target_train = pickle.load(open('data/data_split/feat_target_train.pkl', 'rb'))
        all.rename(columns={feat_target_train['label']:'label', feat_target_train['feat']:'text'}, inplace=True)
        df_val = pd.read_csv("data/val.csv")
        feat_target_val = pickle.load(open('data/data_split/feat_target_val.pkl', 'rb'))
        df_val.rename(columns={feat_target_val['label']:'label', feat_target_val['feat']:'text'}, inplace=True)
        df_test = pd.read_csv("data/test.csv")
        feat_target_test = pickle.load(open('data/data_split/feat_target_test.pkl', 'rb'))
        if 'label' in feat_target_test.keys():
            df_test.rename(columns={feat_target_test['label']:'label', feat_target_test['feat']:'text'}, inplace=True)
        else:
            df_test.rename(columns={feat_target_test['feat']:'text'}, inplace=True)
    num_label= feat_target_train['num_labels']
    epochs= feat_target_train['epochs']
    metric_name= feat_target_train['metric_name']
    num_label = 1 if num_label==2 else num_label
    df_val.to_csv('data/data_split/val.csv', index=False)
    df_train.to_csv('data/data_split/train.csv', index=False)
    df_test.to_csv('data/data_split/test.csv', index=False)

    main(df_train, 'train', df_val, 'val', df_test, 'test', num_label=num_label, metric_name=metric_name, epochs=epochs)