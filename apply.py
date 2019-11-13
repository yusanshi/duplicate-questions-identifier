import os
import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
from train import build_model, preprocess_text
from config import *


def apply(question_pairs):
    with open(os.path.join(MODEL_PATH, 'vocab_size.pickle'), 'rb') as handle:
        vocab_size = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'max_length.pickle'), 'rb') as handle:
        max_length = pickle.load(handle)

    model = build_model(vocab_size, EMBEDDING_DIM,
                        RNN_UNITS, 1, max_length)

    model.load_weights(os.path.join(MODEL_PATH, 'model'))
    print('Load model successfully.')
    model.summary()

    def process_pair(tokenizer, pair, max_length):
        pair = [x.split(' ') for x in map(preprocess_text, pair)]
        pair = [[tokenizer.word_index[y]
                 for y in x if y in tokenizer.word_index] for x in pair]
        pair = keras.preprocessing.sequence.pad_sequences(
            pair, maxlen=max_length, padding='post')

        return np.concatenate(pair)

    question_pairs = np.vstack(([process_pair(tokenizer, pair, max_length)
                                 for pair in question_pairs]))
    predictions = model.predict(question_pairs)
    return predictions
