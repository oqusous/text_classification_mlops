import tensorflow as tf
import keras
import pickle


def classifier_label(num_label):
    if num_label == 1:
        return tf.keras.layers.Dense(num_label)
    else:
        return tf.keras.layers.Dense(num_label, activation='softmax')

def create_rnn(max_len, emb_dim, emb_mtrx, num_o_words, num_label):
    model_rnn = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=num_o_words, output_dim=emb_dim, input_length=max_len, 
                                embeddings_initializer = tf.keras.initializers.Constant(emb_mtrx), trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(64, activation='relu'),
        classifier_label(num_label)
    ])
    return model_rnn

def create_crnn(max_len, emb_dim, emb_mtrx, num_o_words, num_label):
    model_crnn = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=num_o_words, output_dim=emb_dim, input_length=max_len, 
                                embeddings_initializer = tf.keras.initializers.Constant(emb_mtrx), trainable=False),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        classifier_label(num_label)
    ])
    return model_crnn

if __name__ == '__main__':
    pass