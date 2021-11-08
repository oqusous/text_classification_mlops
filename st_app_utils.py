from numpy.lib.function_base import average
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# NN
import tensorflow as tf
from tensorflow.python.framework.ops import container
from transformers import AutoModelForSequenceClassification
from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import keras
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
stop_ws = set(nltk.corpus.stopwords.words('english'))
import re
# utils
import subprocess
import json
import numpy as np
import os

def f1_str_flt(df_col):
    if df_col.dtype == 'O':
        return df_col.apply(lambda x : x.replace('[','').replace(']','')).astype('float')
    return df_col

def return_scores(label, pred, num_label):
    if num_label == 2:
        accuracy = round(accuracy_score(label, pred), 2)
        f1 = round(f1_score(label, pred), 2)
        recall = round(recall_score(label, pred), 2)
        precision = round(precision_score(label, pred), 2)
    else:
        average='macro'
        accuracy = round(accuracy_score(label, pred), 2)
        f1 = round(f1_score(label, pred, average=average), 2)
        recall = round(recall_score(label, pred, average=average), 2)
        precision = round(precision_score(label, pred, average=average), 2)
    return accuracy, f1, recall, precision

def run_and_display_stdout(*cmd_with_args):
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE)
    with open("terminal_logs/log_f.txt", 'w', encoding="utf-8") as log_file:
        for line in iter(lambda: result.stdout.readline(), b""):
            log_file.write(line.decode("utf-8") + '\n')

def tokenize_tf(sentence, maxlen, spacy_nlp=nlp):
    with open('model/tf_nn_arch/tf_tok.json', 'r') as f:
        tok_tf_json = json.load(f)
    tok_tf = keras.preprocessing.text.tokenizer_from_json(tok_tf_json)
    tokens_joined = ''
    nlpd = spacy_nlp(sentence)
    for t in nlpd:
        if (t.is_alpha) and (not t.is_stop) and (not t.is_punct) and (not t.is_space):
            tokens_joined += ' '+t.lemma_.lower()
    train_seq = tok_tf.texts_to_sequences([tokens_joined])
    X_train_tok_padded = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=maxlen)
    return X_train_tok_padded

def tf_predict(sentence, max_len, model_fun, path, le, num_label):
    sentence_tok = tf.constant(tokenize_tf(sentence, max_len))
    model_fun.load_weights(path)
    output = model_fun(sentence_tok)
    if num_label != 2:
        pred = np.argmax(output, axis=1)
    else:
        pred = [0 if output < 0.5 else 1]
    le_inv_tran = le.inverse_transform(pred)
    st.write("Class/Category predicted: {}".format(le_inv_tran[0]))

def hf_predict(text, max_len, le):
    hf_roberta_tok = RobertaTokenizer.from_pretrained('model/roberta_tok')
    hf_roberta_model = AutoModelForSequenceClassification.from_pretrained('model/roberta_model')
    hf_roberta_model.to('cpu')
    hf_roberta_model.eval()
    tokenized_sequence = hf_roberta_tok(text, max_length=max_len, truncation=True, return_tensors='pt')
    prediction = hf_roberta_model(**tokenized_sequence)
    catrgory = int(str(np.argmax(prediction[0].detach().numpy())))
    le_inv_tran= le.inverse_transform([catrgory])
    st.write("Class/Category predicted: {}".format(le_inv_tran[0]))

@st.cache
def roc_plot(y_test, y_test_pred, num_labels, label_names, model):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    if num_labels == 2:
        if model != "RoBERTa":
            y_score = np.exp(y_test_pred)/(1+np.exp(y_test_pred))
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_score = roc_auc_score(y_test, y_score)
            name = f"(AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        else:
            y_score = np.exp(y_test_pred[:,1])/(1+np.exp(y_test_pred[:,1]))
            y_score = y_score.reshape(-1,1)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_score = roc_auc_score(y_test, y_score)
            name = f"(AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    else:
        enc = OneHotEncoder()
        enc.fit(np.arange(0,num_labels).reshape(-1,1))
        y_enc = enc.transform(y_test.reshape(-1,1))
        for i in range(num_labels):
            y_true = y_enc.toarray()[:, i]
            y_score = y_test_pred[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            name = f"{label_names[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      yaxis=dict(scaleanchor="x", scaleratio=1),
                      xaxis=dict(constrain='domain'),
                      legend=dict(y=0.02, x =0.75 ,xanchor="right",yanchor="bottom",traceorder='normal',font=dict(size=10,), orientation="v"),
                      showlegend=True,
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig

@st.cache
def cf_plot(y_test, y_test_pred, ax_labels):
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    cnf_matrix = np.round_(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis], 3)
    x=sorted(ax_labels)
    y=sorted(ax_labels)
    colorscale = [[0, 'rgb(255, 220, 130)'], [1, 'rgb(0, 168, 255)']]
    font_colors = ['blue', 'black']
    fig = ff.create_annotated_heatmap(cnf_matrix, x=x, y=y, colorscale=colorscale, font_colors=font_colors)
    fig["layout"]["xaxis"].update(side="bottom")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig

@st.cache
def plotly_training_graphs(df1, title_, training_avilable, y_axis1, y_axis2):
    fig1 = go.Figure(layout = {'xaxis': {'title': 'Epoch','visible': True,'showticklabels': True},\
                               'yaxis': {'title': 'Loss/Metric','visible': True,'showticklabels': True}})
    if training_avilable:
        fig1.add_trace(go.Scatter(name="loss", x=df1['Epoch'], y= f1_str_flt(df1['loss']), legendrank=4, mode='lines'))
        fig1.add_trace(go.Scatter(name=y_axis1, x=df1['Epoch'], y= f1_str_flt(df1[y_axis1]), legendrank=2, mode='lines'))
    fig1.add_trace(go.Scatter(name="val_loss", x=df1['Epoch'], y= f1_str_flt(df1['val_loss']), legendrank=1, mode='lines'))
    fig1.add_trace(go.Scatter(name=y_axis2, x=df1['Epoch'], y= f1_str_flt(df1[y_axis2]), legendrank=3, mode='lines'))
    fig1.update_layout(title_text=title_, 
                       legend=dict(orientation="h",yanchor="top",y=-0.15,xanchor="center",x=0.5,),
                       margin=dict(l=20, r=20, t=30, b=100)) # width=800, height=500, 
    return fig1

@st.cache
def plotly_wordfreq(df, train_feat_col, train_label_col, lables_selected, num_cols = 2, num_most_common=5):
    num_label=len(lables_selected)
    num_rows = num_label // num_cols 
    num_rows += num_label % num_cols
    labels=lables_selected
    labels.sort()
    labels_as_title = [str(x) for x in labels]
    fig1 = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=labels_as_title, vertical_spacing=0.12)
    label_idx = 0
    for r in  range(1,num_rows+1):
        for c in range(1,num_cols+1):
            if label_idx<len(labels):
                df_label = df[df[train_label_col]==labels[label_idx]]
                tokens_list=[]
                for sent in df_label[train_feat_col].tolist():
                    tokens_list.extend(re.sub(r'[^\w\s]','',str(sent)).split(' '))
                filter_tokens = [token.lower().strip() for token in tokens_list \
                                    if token.lower().strip() not in list(stop_ws) and token!=' ' and token !='' ]
                
                fd = nltk.FreqDist(filter_tokens)
                most_common_words = fd.most_common(num_most_common)
                # fig1 = go.Figure(layout = {'xaxis': {'title': 'Word','visible': True,'showticklabels': True},\
                #                         'yaxis': {'title': 'Frequency','visible': True,'showticklabels': True}})
                fig1.add_trace(go.Bar(x=[k[0] for k in most_common_words], y=[v[1] for v in most_common_words]), row=r,col=c)
                label_idx+=1
    height=min(1000, 200+(200*num_rows))
    return fig1.update_layout(showlegend=False, height=height, margin=dict(l=20, r=20, t=30, b=20))