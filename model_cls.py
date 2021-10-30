import pickle
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from datasets import load_metric
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, TrainingArguments, Trainer # , optuna, ray
import numpy as np
import pickle
import json
import glob
import os
from shutil import copyfile

def classifier_label(num_label):
    if num_label == 1:
        return tf.keras.layers.Dense(num_label)
    else:
        return tf.keras.layers.Dense(num_label, activation='softmax')

def loss_function(num_label):
    if num_label == 1:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.CategoricalCrossentropy()

def metric_for_tf(metric_name, num_label):
    if metric_name=='accuracy':
        return "accuracy"
    else:
        if num_label == 2:
            return tfa.metrics.F1Score(num_classes=num_label, average= None)
        else:
            return tfa.metrics.F1Score(num_classes=num_label, average= 'macro')

def y_for_categorical_crossentropy(y_le, num_label):
    if num_label != 2:
        enc = OneHotEncoder()
        enc.fit(np.arange(0,num_label).reshape(-1,1))
        y_enc = enc.transform(y_le.reshape(-1,1))
        return tf.convert_to_tensor(y_enc.toarray(), dtype=tf.int32)
    else:
        return y_le
    

def bi_rnn_model(X_train_tok_padded, y_train, X_val_tok_padded, y_val, \
                num_o_words, emb_dim, max_len, emb_mtrx, num_label, epochs, metric_name):
    
    model_rnn = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=num_o_words, output_dim=emb_dim, input_length=max_len, 
                                embeddings_initializer = tf.keras.initializers.Constant(emb_mtrx), trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(64, activation='relu'),
        classifier_label(num_label)
    ])
    print(model_rnn.summary())
    # ModelCheckpoint callback
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='model/model_rnn.h5', save_best_only=True,
                                                        save_weights_only=True)
    # EarlyStopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    # compile and run
    model_rnn.compile(tf.keras.optimizers.Adam(learning_rate=0.0001) , loss=loss_function(num_label), 
                metrics=[metric_for_tf(metric_name, num_label)])
    history = model_rnn.fit(X_train_tok_padded, y_for_categorical_crossentropy(y_train.values, num_label), \
                            validation_data=(X_val_tok_padded,y_for_categorical_crossentropy(y_val.values, num_label)), \
                            batch_size= 32, epochs=epochs,
                            callbacks=[model_checkpoint_cb, early_stopping_cb, reduce_lr_on_plateau_cb])
    return model_rnn, history

def pred_test(model, X_test_tok_padded):
    preds = model.predict_classes(X_test_tok_padded)
    return preds

def crnn_model(X_train_tok_padded, y_train, X_val_tok_padded, y_val, \
                num_o_words, emb_dim, max_len, emb_mtrx, num_label, epochs, metric_name):
    num_label = 1 if num_label==2 else num_label
    model_crnn = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=num_o_words, output_dim=emb_dim, input_length=max_len, 
                                embeddings_initializer = tf.keras.initializers.Constant(emb_mtrx), trainable=False),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        classifier_label(num_label)
    ])
    print(model_crnn.summary())
    # ModelCheckpoint callback
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='model/model_crnn.h5', save_best_only=True,
                                                        save_weights_only=True)
    # EarlyStopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    # ReduceLROnPlateau callback
    reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    # compile and run
    model_crnn.compile(tf.keras.optimizers.Adam(learning_rate=0.0001) , loss=loss_function(num_label), 
                    metrics=[metric_for_tf(metric_name, num_label)])
    history_crnn = model_crnn.fit(X_train_tok_padded, y_for_categorical_crossentropy(y_train.values, num_label), \
                            validation_data=(X_val_tok_padded,y_for_categorical_crossentropy(y_val.values, num_label)), \
                            batch_size= 32, epochs=epochs,
                            callbacks=[model_checkpoint_cb, early_stopping_cb, reduce_lr_on_plateau_cb])
    
    return model_crnn, history_crnn

def roberta_model_cls(train_ds_en, val_ds_en, num_labels, metric_name, epochs):
    tokenizer = RobertaTokenizerFast.from_pretrained("model/roberta_tok")
    model_checkpoint = "distilroberta-base"
    batch_size = 8
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    args = TrainingArguments(
        "model/roberta_text_cls",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=1.5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='plots/logging_dir',
        metric_for_best_model=metric_name)
    
    metric = load_metric('f1',  average='macro') if metric_name =='f1' else load_metric('accuracy')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        if metric_name == 'f1':
            calc_metric = metric.compute(predictions=predictions, references=labels, average='macro')
        else:
            calc_metric = metric.compute(predictions=predictions, references=labels)
        return calc_metric

    trainer = Trainer(model, args, \
        train_dataset=train_ds_en, eval_dataset=val_ds_en,
        tokenizer=tokenizer, compute_metrics=compute_metrics)

    trainer.train()
    model.save_pretrained('model/roberta_model')

    search_dir = "model/roberta_text_cls/"
    checkpoint_dir = list(filter(os.path.isdir, glob.glob(search_dir + "*")))
    checkpoint_dir.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    last_checkpoint = os.path.join(checkpoint_dir[0], 'trainer_state.json')
    copyfile(last_checkpoint, "./plots/trainer_state.json")

    return model, trainer

if __name__ == '__main__':
    pass