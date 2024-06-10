{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "97c0f57a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Input, LSTM, Embedding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f6881ca9",
   "metadata": {},
   "outputs": [],
   "source": [
    "class TextPreprocessor(BaseEstimator, TransformerMixin):\n",
    "    def __init__(self, max_sequence_length=100):\n",
    "        self.max_sequence_length = max_sequence_length\n",
    "        self.tokenizer = None\n",
    "\n",
    "    def fit(self, X, y=None):\n",
    "        self.tokenizer = Tokenizer()\n",
    "        self.tokenizer.fit_on_texts(X)\n",
    "        return self\n",
    "\n",
    "    def transform(self, X):\n",
    "        if self.tokenizer is None:\n",
    "            raise RuntimeError(\"The tokenizer is not fitted yet. Please call fit before transform.\")\n",
    "        sequences = self.tokenizer.texts_to_sequences(X)\n",
    "        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)\n",
    "        return padded_sequences\n",
    "\n",
    "class LSTMFeatureExtractor(BaseEstimator, TransformerMixin):\n",
    "    def __init__(self, vocab_size, embedding_dim, max_sequence_length, embedding_matrix):\n",
    "        self.vocab_size = vocab_size\n",
    "        self.embedding_dim = embedding_dim\n",
    "        self.max_sequence_length = max_sequence_length\n",
    "        self.embedding_matrix = embedding_matrix\n",
    "        self.model = self._build_model()\n",
    "\n",
    "    def _build_model(self):\n",
    "        input_layer = Input(shape=(self.max_sequence_length,))\n",
    "        embedding_layer = Embedding(input_dim=self.vocab_size,\n",
    "                                    output_dim=self.embedding_dim,\n",
    "                                    weights=[self.embedding_matrix],\n",
    "                                    input_length=self.max_sequence_length,\n",
    "                                    trainable=False)(input_layer)\n",
    "        lstm_layer = LSTM(128)(embedding_layer)\n",
    "        model = Model(inputs=input_layer, outputs=lstm_layer)\n",
    "        return model\n",
    "\n",
    "    def fit(self, X, y=None):\n",
    "        return self\n",
    "\n",
    "    def transform(self, X):\n",
    "        return self.model.predict(X)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed5606c6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
