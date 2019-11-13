from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from config import *

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import io
import re
import time
import pickle
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def build_model(vocab_size, embedding_dim, rnn_units, batch_size, max_length):
    model = keras.Sequential([
        layers.Embedding(vocab_size + 1, embedding_dim),
        layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(rnn_units // 2)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # https://www.kaggle.com/advaitsave/lstm-using-tensorflow-2-with-embeddings
    return model


def preprocess_text(w):
    w = re.sub(r"([?.:()!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.lower().strip()
    # w = '<start> ' + w + ' <end>'
    return w


def train():

    # Prepare data
    path_to_zip = keras.utils.get_file(
        'quora_duplicate_questions.zip', origin='https://yun.yusanshi.com/TF_datasets/quora_duplicate_questions.zip', extract=True)
    path_to_file = os.path.join(os.path.dirname(
        path_to_zip), 'quora_duplicate_questions.tsv')

    dataframe = pd.read_table(path_to_file)
    dataframe = dataframe.dropna(axis=0)
    dataframe = dataframe[:DATA_NUM]

    question_one = list(map(preprocess_text, dataframe['question1']))
    question_two = list(map(preprocess_text, dataframe['question2']))
    labels = list(dataframe['is_duplicate'])

    def tokenize(lang):
        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        return lang_tokenizer

    tokenizer = tokenize(question_one + question_two)
    question_one_seq = tokenizer.texts_to_sequences(question_one)
    question_two_seq = tokenizer.texts_to_sequences(question_two)
    max_length = max(len(t) for t in question_one_seq + question_two_seq)
    vocab_size = len(tokenizer.index_word)

    question_one_seq_padded = keras.preprocessing.sequence.pad_sequences(question_one_seq,
                                                                         maxlen=max_length,
                                                                         padding='post')
    question_two_seq_padded = keras.preprocessing.sequence.pad_sequences(question_two_seq,
                                                                         maxlen=max_length,
                                                                         padding='post')

    question_pairs = np.concatenate(
        (question_one_seq_padded, question_two_seq_padded), axis=-1)

    # for pair, label in list(zip(question_pairs, labels))[:3]:
    #     print('Q1:', end='')
    #     print(' '.join([tokenizer.index_word[x]
    #                     for x in pair[:max_length] if x != 0]))
    #     print('Q2:', end='')
    #     print(' '.join([tokenizer.index_word[x]
    #                     for x in pair[max_length:] if x != 0]))
    #     print('Duplicate? ', end='')
    #     print('Yes' if label == 1 else 'No')
    #     print()

    def split_to_three(features, labels, proportion):
        # train, validate, test
        assert len(proportion) == 3
        proportion = [x / sum(proportion) for x in proportion]
        features_train, features_tmp, labels_train, labels_tmp = train_test_split(
            features, labels, train_size=proportion[0])
        features_val, features_test, labels_val, labels_test = train_test_split(
            features_tmp, labels_tmp, train_size=proportion[1] / (proportion[1] + proportion[2]))
        return features_train, features_val, features_test, labels_train, labels_val, labels_test

    questions_train, questions_val, questions_test, labels_train, labels_val, labels_test = split_to_three(
        question_pairs, labels, SPLIT)

    # print(len(questions_train))
    # print(len(questions_val))
    # print(len(questions_test))

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (questions_train, labels_train)).shuffle(len(questions_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (questions_val, labels_val))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (questions_test, labels_test))

    # for question, label in train_dataset.take(3):
    #     print(question)
    #     print(label)
    #     print()

    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    validation_dataset = validation_dataset.batch(
        BATCH_SIZE, drop_remainder=True)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

    model = build_model(vocab_size, EMBEDDING_DIM,
                        RNN_UNITS, BATCH_SIZE, max_length)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    logdir = os.path.join(
        LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1)

    # print(train_dataset)
    model.summary()

    model.fit(train_dataset, validation_data=validation_dataset,
              epochs=EPOCHS, callbacks=[tensorboard_callback])

    result = model.evaluate(test_dataset)
    print('Finish trainning. Test loss: %.4f. Test accuracy: %.4f.' %
          (result[0], result[1]))

    def save_model(variables, models):
        for k, v in variables.items():
            with open(os.path.join(MODEL_PATH, k + '.pickle'), 'wb') as handle:
                pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for k, v in models.items():
            v.save_weights(os.path.join(MODEL_PATH, k))

        print('Model saved in %s.' % MODEL_PATH)

    save_model(
        {
            'vocab_size': vocab_size,
            'tokenizer': tokenizer,
            'max_length': max_length
        },
        {
            'model': model
        }
    )


if __name__ == '__main__':
    train()
