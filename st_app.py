import streamlit as st
import nltk
import pandas as pd
import tensorflow as tf
from model.tf_nn_arch.create_models import create_crnn, create_rnn
from st_app_utils import *
import torch
import keras
import boto3
import sys
import subprocess
import json
from tqdm import tqdm
import numpy as np
import pickle
from os import listdir
import os
from os.path import isfile, exists
import base64
st.set_page_config(layout="wide")
device = torch.device('cpu')
st.markdown('<h1 style="text-align:center;color:white;font-weight:bolder;font-size:100px;">MLOps Text Classifier</h1>',unsafe_allow_html=True)
# st.title("MLOps Text Classifier")
st.title("\n")
num_of_labels = None
name_of_feat_opts_train = ['']
name_of_feat_opts_test = ['']
name_of_feat_opts_val = ['']
name_of_target_opts_train = ['']
name_of_target_opts_test = ['']
name_of_target_opts_val = ['']
can_train = False
name_of_target_val = None
name_of_feat_val = None
option_test_label = None
if exists("data/data_split/feat_target_train.pkl"):
    metric_train_eval = pickle.load(open("data/data_split/feat_target_train.pkl", "rb"))['metric_name']
    num_label = pickle.load(open("data/data_split/feat_target_train.pkl", "rb"))['num_labels']
st.sidebar.subheader("About")
st.sidebar.write("MLOps Text Classifier")
st.sidebar.write("Authored: Omar Q")
st.sidebar.write("Version 1.0")

st.header("Upload data files using the widgets below")

row0_1, row0_2, row0_3= st.columns([1,1,1])

with row0_1:
    train_csv = st.file_uploader(label="Upload training set here", type=["csv"], accept_multiple_files=False, \
                                        key=None, help=None, on_change=None, args=None, kwargs=None)
    st.write("\n")
    if train_csv is not None:
        df_train_ = pd.read_csv(train_csv)
        train_cols = list(df_train_.columns)
        df_train_.to_csv('data/train.csv', index=False)

with row0_2:
    val_csv = st.file_uploader(label="If validation set is available upload here", type=["csv"], accept_multiple_files=False, \
                                        key=None, help=None, on_change=None, args=None, kwargs=None)
    if val_csv is not None:
        df_val = pd.read_csv(val_csv)
        val_cols = list(df_val.columns)
        df_val.to_csv('data/val.csv', index=False)

with row0_3:
    test_csv = st.file_uploader(label="If test set is available upload here", type=["csv"], accept_multiple_files=False, \
                                        key=None, help=None, on_change=None, args=None, kwargs=None)
    st.write("\n")
    if test_csv is not None:
        df_test = pd.read_csv(test_csv)
        test_cols = list(df_test.columns)
        df_test.to_csv('data/test.csv', index=False)
        option_test_label = st.selectbox(label='Is the test set labeled?',options=('','Yes', 'No'))

if (train_csv is not None) and (test_csv is not None) and (val_csv is not None):
    name_of_target_opts_train.extend(train_cols)
    name_of_target_opts_test.extend(test_cols)
    name_of_target_opts_val.extend(val_cols)
    name_of_feat_opts_train.extend(train_cols)
    name_of_feat_opts_test.extend(val_cols)
    name_of_feat_opts_val.extend(val_cols)
    name_of_target_train = st.selectbox('Select name of target (label) column in training set:', options=name_of_target_opts_train)
    name_of_feat_train = st.selectbox('Select name of feature (text) column in training set:', options=name_of_feat_opts_train)
    if option_test_label == 'Yes':
        name_of_target_test = st.selectbox('Select name of target (label) column in test set:', options=name_of_target_opts_test)
    name_of_feat_test = st.selectbox('Select name of feature (text) column in test set:', options=name_of_feat_opts_test)
    name_of_target_val = st.selectbox('Select name of target (label) column in val set:', options=name_of_target_opts_val)
    name_of_feat_val = st.selectbox('Select name of feature (text) column in val set:', options=name_of_feat_opts_val)
    if option_test_label == 'Yes' and name_of_target_test and name_of_target_train and name_of_target_val and name_of_feat_train and name_of_feat_test and name_of_feat_val:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="train set count")
        num_label = len(name_of_labels_detected_train)
        name_of_labels_detected_val = df_val[name_of_target_val].value_counts().to_frame(name="val set count")
        name_of_labels_detected_test = df_test[name_of_target_test].value_counts().to_frame(name="test set count")
        st.write("Number of classes detected in training, validation and test sets are {}, {} and {} respectively. The names and counts of classes for each set are:".format(len(name_of_labels_detected_train), len(name_of_labels_detected_val), len(name_of_labels_detected_test)))
        st.table(pd.concat([name_of_labels_detected_train, name_of_labels_detected_val, name_of_labels_detected_test], axis=1))
        if np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_test.index.tolist().sort()) and np.array_equal(name_of_labels_detected_val.index.tolist().sort() , name_of_labels_detected_train.index.tolist().sort()):
            can_train = True
        elif np.array_equal(name_of_labels_detected_train.index.tolist().sort() != name_of_labels_detected_test.index.tolist().sort()) or np.array_equal(name_of_labels_detected_train.index.tolist().sort() != name_of_labels_detected_val.index.tolist().sort()):
            st.warning("WARNING: Number of classes in target columns have to match in order to be able to proceed to training.")
    elif option_test_label == 'No' and name_of_target_train and name_of_feat_train and name_of_feat_test and name_of_feat_val and name_of_target_val:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="train set count")
        num_label = len(name_of_labels_detected_train)
        name_of_labels_detected_val = df_val[name_of_target_val].value_counts().to_frame(name="val set count")
        st.write("Number of classes detected in training and validation sets are {} and {} respectively. The names and counts of classes for each set are:".format(len(name_of_labels_detected_train), len(name_of_labels_detected_val)))
        st.table(pd.concat([name_of_labels_detected_train, name_of_labels_detected_val], axis=1))
        if np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_val.index.tolist().sort()):
            can_train = True
        elif (name_of_labels_detected_train != name_of_labels_detected_val):
            st.warning("WARNING: Number of classes in target columns have to match in order to be able to proceed to training.")

elif (train_csv is not None) and (test_csv is not None):
    name_of_target_opts_train.extend(train_cols)
    name_of_target_opts_test.extend(test_cols)
    name_of_feat_opts_train.extend(train_cols)
    name_of_feat_opts_test.extend(test_cols)
    name_of_target_train = st.selectbox('Select name of target (label) column in training set:', options=name_of_target_opts_train)
    name_of_feat_train = st.selectbox('Select name of feature (text) column in training set:', options=name_of_feat_opts_train)
    if option_test_label == 'Yes':
        name_of_target_test = st.selectbox('Select name of target (label) column in test set:', options=name_of_target_opts_test)
    name_of_feat_test = st.selectbox('Select name of feature (text) column in test set:', options=name_of_feat_opts_test)
    if option_test_label == 'Yes' and name_of_target_test and name_of_target_train and name_of_feat_train and name_of_feat_test:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="train set count")
        num_label = len(name_of_labels_detected_train)
        name_of_labels_detected_test = df_test[name_of_target_test].value_counts().to_frame(name="test set count")
        st.write("Number of classes detected in training and test sets are {} and {} respectively. The names and counts of classes for each set are:".format(len(name_of_labels_detected_train), len(name_of_labels_detected_test)))
        st.table(pd.concat([name_of_labels_detected_train, name_of_labels_detected_test], axis=1))
        if option_test_label == 'Yes' and np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_test.index.tolist().sort()):
            can_train = True
        elif option_test_label == 'Yes' and not np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_test.index.tolist().sort()):
            st.warning("WARNING: Number of classes in target columns have to match in order to be able to proceed to training.")
    elif option_test_label == 'No' and name_of_target_train and name_of_feat_train and name_of_feat_test:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="count")
        st.write("Number of classes detected in training set is {}. The names and counts of classes for the set are:".format(len(name_of_labels_detected_train)))
        st.table(name_of_labels_detected_train)
        can_train = True

elif (train_csv is not None) and (val_csv is not None):
    name_of_target_opts_train.extend(train_cols)
    name_of_target_opts_val.extend(val_cols)
    name_of_feat_opts_train.extend(train_cols)
    name_of_feat_opts_val.extend(val_cols)
    name_of_target_train = st.selectbox('Select name of target (label) column in training set:', options=name_of_target_opts_train)
    name_of_feat_train = st.selectbox('Select name of feature (text) column in training set:', options=name_of_feat_opts_train)
    name_of_target_val = st.selectbox('Select name of target (label) column in validation set:', options=name_of_target_opts_val)
    name_of_feat_val = st.selectbox('Select name of feature (text) column in validation set:', options=name_of_feat_opts_val)
    if name_of_target_val and name_of_target_train and name_of_feat_train and name_of_feat_val:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="train set count")
        num_label = len(name_of_labels_detected_train)
        name_of_labels_detected_val = df_val[name_of_target_val].value_counts().to_frame(name="val set count")
        st.write("Number of classes detected in training and validation set are {} and {} respectively. The names and counts of classes for each set are:".format(len(name_of_labels_detected_train), len(name_of_labels_detected_val)))
        st.table(pd.concat([name_of_labels_detected_train, name_of_labels_detected_val], axis=1))
        if np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_val.index.tolist().sort()):
            can_train = True
        elif not np.array_equal(name_of_labels_detected_train.index.tolist().sort() , name_of_labels_detected_val.index.tolist().sort()):
            st.warning("WARNING: Number of classes in target columns have to match in order to be able to proceed to training.")

elif (train_csv is not None):
    name_of_target_opts_train.extend(train_cols)
    name_of_feat_opts_train.extend(train_cols)
    name_of_target_train = st.selectbox('Select name of target (label) column:', options=name_of_target_opts_train)
    name_of_feat_train = st.selectbox('Select name of feature (text) column:', options=name_of_feat_opts_train)
    if name_of_target_train and name_of_feat_train:
        name_of_labels_detected_train = df_train_[name_of_target_train].value_counts().to_frame(name="count")
        num_label = len(name_of_labels_detected_train)
        name_of_labels_detected_train.index.name = 'label'
        st.write("Number of classes detected in training set is {}. The names and counts of classes for the set are:".format(len(name_of_labels_detected_train)))
        st.table(name_of_labels_detected_train)
        can_train = True

if can_train==True:
    st.header("\n")
    st.header("Data Visualization")
    st.subheader("Histogram of most frequent words for each label")
    row1_1, row1_s1, row1_2= st.columns([2,0.2,1])
    with row1_1:
        num_most_common=st.slider('Max number of most common words to show for each label',min_value=5,max_value=25,value=5,step=1)
    with row1_2:
        num_cols=st.slider('Number of columns to show plot in',min_value=1,max_value=3,value=2,step=1)
    labels_to_show=st.multiselect("Select labels to plot the histograms",options=df_train_[name_of_target_train].unique(), default=df_train_[name_of_target_train].unique())
    
    fig_wf = plotly_wordfreq(df=df_train_, train_feat_col=name_of_feat_train, train_label_col=name_of_target_train, lables_selected=labels_to_show, num_cols = num_cols, num_most_common=num_most_common)
    st.plotly_chart(fig_wf, use_container_width=True)

    st.header("Training parameters")
    st.write("\n")
    row2_1, row2_s1, row2_2= st.columns([2,0.2,1])
    with row2_1:
        epochs = st.slider("Choose number of epochs to run each model:", min_value=2, max_value=15, value=8, step=1)
        st.write("\n")
    with row2_2:
        metric_train_eval=st.selectbox("Select evaluation meteric:", options=["Accuracy","F1-Score"], index=0)
        st.write("\n")
    metric_train_eval = 'accuracy' if metric_train_eval=="Accuracy" else 'f1'
    if st.button("Start Training", key=None, help=None, on_click=None, args=None, kwargs=None):
        pickle.dump( {'label': name_of_target_train, 'feat': name_of_feat_train, \
                      "num_labels":num_label, "epochs":int(epochs), \
                      "metric_name":metric_train_eval}, open('data/data_split/feat_target_train.pkl','wb'))
        if name_of_target_val and name_of_feat_val:
            pickle.dump( {'label': name_of_target_val, 'feat': name_of_feat_val} , open('data/data_split/feat_target_val.pkl','wb'))
        if option_test_label == 'Yes':
            pickle.dump( {'label': name_of_target_test, 'feat': name_of_feat_test} , open('data/data_split/feat_target_test.pkl','wb'))
        elif option_test_label == 'No':
            pickle.dump( {'feat': name_of_feat_test} , open('data/data_split/feat_target_test.pkl','wb'))
        with st.empty():
            st.warning(f"⏳ Model training in progress. WARNING: Do not refresh or hit the back button on the browser")
            run_and_display_stdout("sh", "./sh_scripts/run_all.sh")
            st.success("✔️ All models trained successfully!")
files_and_dirs_model = [f for f in listdir('model/')]
files_and_dirs_roberta = []
if exists('model/roberta_model/'):
    files_and_dirs_roberta = [f for f in listdir('model/roberta_model/')]
files_and_dirs_tf = [f for f in listdir('model/tf_nn_arch/')]
files_and_dirs_preds = [f for f in listdir('test_preds/')]

if 'pytorch_model.bin' in files_and_dirs_roberta and 'model_crnn.h5' in files_and_dirs_model and 'model_rnn.h5' in files_and_dirs_model and 'tf_tok.json' in files_and_dirs_tf:
    st.header('\n')
    st.header('Model training metrics and loss')
    row3_1, row3_2, row3_3= st.columns([1,1,1])
    # RNN plot
    df1 = pd.read_csv('plots/rnn_plot.csv')
    df1.rename(columns={'Unnamed: 0':'Epoch'}, inplace=True)
    df1['Epoch'] = df1['Epoch'].apply(lambda x: x+1)
    df1_cols = df1.columns.tolist()
    y_axis1 = "accuracy" if "accuracy" in df1_cols else "f1_score"
    with row3_1:
        fig1 = plotly_training_graphs(df1, "RNN training results", True, y_axis1)
        st.plotly_chart(fig1, use_container_width=True)
    # CRNN plot
    df2 = pd.read_csv('plots/crnn_plot.csv')
    df2.rename(columns={'Unnamed: 0':'Epoch'}, inplace=True)
    df2['Epoch'] = df2['Epoch'].apply(lambda x: x+1)
    with row3_2:
        fig2 = plotly_training_graphs(df2, "CNN+RNN Hybrid training results", True, y_axis1)
        st.plotly_chart(fig2, use_container_width=True)
    # Roberta plot
    path_to_rob_eval = "plots/trainer_state.json"
    with open(path_to_rob_eval, 'r', encoding='utf-8') as f:
        rob_eval =json.load(f)
    epoch_x = []
    eval_meteric=[]
    eval_loss=[]
    for epoch in rob_eval['log_history']:
        if epoch['epoch'].is_integer():
            epoch_x.append(int(epoch['epoch']))
            if y_axis1 == 'accuracy':
                eval_meteric.append(epoch['eval_accuracy'])
                y_axis2 = "val_accuracy"
            else:
                 eval_meteric.append(epoch['eval_f1'])
                 y_axis2 = "val_f1"
            eval_loss.append(epoch['eval_loss'])
    df3 = pd.DataFrame({"Epoch":epoch_x, "val_loss":eval_loss, y_axis2:eval_meteric})
    with row3_3:
        fig3 = plotly_training_graphs(df3, "RoBERTa training Results", False, y_axis2)
        st.plotly_chart(fig3, use_container_width=True)

    st.header("Test models with custom sentences")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    max_len = pickle.load(open('model/tf_nn_arch/max_len.pkl', 'rb'))
    emb_dim = pickle.load(open('model/tf_nn_arch/emb_dim.pkl', 'rb'))
    emb_mtrx = pickle.load(open('model/tf_nn_arch/emb_mtrx.pkl', 'rb'))
    num_o_words = pickle.load(open('model/tf_nn_arch/num_o_words.pkl', 'rb'))
    row4_1, row4_2 = st.columns([1,1])
    with row4_1:
        sentence_test = st.text_area('Type test sentence:',max_chars=1000,height=100)
    with row4_2:
        option_test = st.selectbox(label='Select classifier model',options=('','RNN', 'CNN+RNN', 'RoBERTa'))
    le_inv_transform = pickle.load(open("data/data_split/le.pkl", "rb"))
    if len(sentence_test) > 0 and (option_test == 'RNN'):
        model_rnn =create_rnn(max_len, emb_dim, emb_mtrx, num_o_words, num_label)
        tf_predict(sentence=option_test, max_len=max_len, model_fun=model_rnn, path='model/model_rnn.h5', le=le_inv_transform, num_label=num_label)
        tf.keras.backend.clear_session()
    elif len(sentence_test) > 0 and (option_test == 'CNN+RNN'):
        model_crnn =create_crnn(max_len, emb_dim, emb_mtrx, num_o_words, num_label)
        tf_predict(sentence=sentence_test, max_len=max_len, model_fun=model_crnn, path='model/model_crnn.h5', le=le_inv_transform, num_label=num_label)
        tf.keras.backend.clear_session()
    elif len(sentence_test) > 0 and (option_test == 'RoBERTa'):
        hf_predict(sentence_test, max_len, le=le_inv_transform)
    else:
        pass

if ("roberta_preds.pkl" in files_and_dirs_preds) and ("predict_rnn.pkl" in files_and_dirs_preds) and ("predict_crnn.pkl" in files_and_dirs_preds):
    st.header("\n")
    st.header("Testset Results")
    roberta_preds = pickle.load(open("test_preds/roberta_preds.pkl", "rb"))
    rnn_preds = pickle.load(open("test_preds/predict_rnn.pkl", "rb"))
    crnn_preds = pickle.load(open("test_preds/predict_crnn.pkl", "rb"))
    labels_test = pd.read_csv('data/data_split/test.csv')
    test_cols=list(labels_test.columns)
    roberta_preds_class = np.argmax(roberta_preds[0], axis=1)
    if num_label == 2:
        rnn_preds_class = [0 if x <0.5 else 1 for x in rnn_preds]
        crnn_preds_class = [0 if x <0.5 else 1 for x in crnn_preds]
    else:
        rnn_preds_class = np.argmax(rnn_preds, axis=1)
        crnn_preds_class = np.argmax(crnn_preds, axis=1)
        
    # le_classes = np.sort(le_inv_transform.classes_)
    # pred_classes = np.sort(np.unique( rnn_preds_class))
    # le_same = np.array_equal(le_classes, pred_classes)
    if 'label' not in test_cols:
        st.write("RNN model testset class count:")
        st.write(np.unique( rnn_preds_class , return_counts=True))
        st.write("RNN+CNN hybrid model testset class count:")
        st.write(np.unique( crnn_preds_class , return_counts=True))
        st.write("RoBERTa model testset class count:")
        st.write(np.unique( roberta_preds_class , return_counts=True))
    elif 'label' in test_cols:
        labels_test_ = le_inv_transform.transform(labels_test['label'].values)
        accuracy_rnn, f1_rnn, recall_rnn, precision_rnn = return_scores(labels_test_, rnn_preds_class, num_label)
        accuracy_crnn, f1_crnn, recall_crnn, precision_crnn = return_scores(labels_test_, crnn_preds_class, num_label)
        accuracy_rob, f1_rob, recall_rob, precision_rob = return_scores(labels_test_, roberta_preds_class, num_label)
        st.write("Model test dataset scores:")
        st.table(data=pd.DataFrame(data={"RNN":[accuracy_rnn, f1_rnn, recall_rnn, precision_rnn], \
                                         "RNN+CNN":[accuracy_crnn, f1_crnn, recall_crnn, precision_crnn], \
                                         "RoBERTa":[accuracy_rob, f1_rob, recall_rob, precision_rob]}, \
                                         index=['Accuracy', 'F1', 'Recall','Precision']))
    choice_preds_dict={"RNN":rnn_preds_class, "RNN+CNN":crnn_preds_class, "RoBERTa":roberta_preds_class}
    choice_preds_key = st.selectbox("Choose testset predictions to download.", options=list(choice_preds_dict.keys()))
    df_download = pd.concat([labels_test, pd.DataFrame(choice_preds_dict[choice_preds_key], columns=['pred_labels'],index=labels_test.index)], axis=1)
    csv_download = df_download.to_csv(index=False)
    if st.button("Download CSV"):
        b64 = base64.b64encode(csv_download.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')

confirmation = None
st.write("To rest the app and upload new data click on the Reset button below. This will delete trained models and data uploaded.")

if st.button("Reset", key=None, help=None, on_click=None, args=None, kwargs=None):
    subprocess.call(['sh', './sh_scripts/delete_data_models.sh'])
