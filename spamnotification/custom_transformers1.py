
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_sequence_length=100):
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None

    def fit(self, X, y=None):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        if self.tokenizer is None:
            raise RuntimeError("The tokenizer is not fitted yet. Please call fit before transform.")
        sequences = self.tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences

class LSTMFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.vocab_size,
                                    output_dim=self.embedding_dim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=False)(input_layer)
        lstm_layer = LSTM(128)(embedding_layer)
        model = Model(inputs=input_layer, outputs=lstm_layer)
        return model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.predict(X)