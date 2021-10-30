import data_processing
import model_cls
import numpy as np
import pandas as pd
import pickle
import os

def main(num_label, test_type, metric_name, epochs):
    df_train = pd.read_csv("data/data_split/train.csv")
    df_test = pd.read_csv("data/data_split/test.csv")
    df_val = pd.read_csv("data/data_split/val.csv")

    le_transform = pickle.load(open('data/data_split/le.pkl', 'rb'))

    for df in [df_train, df_val]:
        df['text'].fillna(value="Empty", inplace=True)
        df['label'].fillna(value=0, inplace=True)
        df['label']=le_transform.transform(df['label'])
    if test_type == 'test_target':
        df_test['text'].fillna(value="Empty", inplace=True)
        df_test['label'].fillna(value=0, inplace=True)
        df_test['label']=le_transform.transform(df_test['label'])
    elif test_type == 'test':
        df_test['text'].fillna(value="Empty", inplace=True)
    
    train_ds_en = data_processing.roberta_tokenize(df_train, 'train')
    val_ds_en = data_processing.roberta_tokenize(df_val, 'val')
    test_ds_en = data_processing.roberta_tokenize(df_test, test_type)

    model_roberta, trainer_roberta = model_cls.roberta_model_cls(train_ds_en, val_ds_en, num_label, metric_name, epochs)
    trainer_roberta.evaluate()
    preds = trainer_roberta.predict(test_ds_en)
    pickle.dump(preds, open('test_preds/roberta_preds.pkl', 'wb'))

if __name__ == '__main__':
    files_and_dirs_data_split = [f for f in os.listdir('data/data_split/')]
    test_type = 'test_target'
    if 'feat_target_test.pkl' in files_and_dirs_data_split:
        test_pkl = pickle.load(open('data/data_split/feat_target_test.pkl', 'rb'))
        test_type = 'test_target' if 'label' in test_pkl.keys() else 'test'
    feat_target_train = pickle.load(open('data/data_split/feat_target_train.pkl', 'rb'))
    num_label= feat_target_train['num_labels']
    epochs= feat_target_train['epochs']
    metric_name = feat_target_train['metric_name']
    main(num_label=num_label, test_type=test_type, metric_name=metric_name, epochs=epochs)