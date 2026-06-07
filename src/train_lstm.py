"""
Step 7a: Train a compact LSTM for next-word prediction.
Uses NLTK's gutenberg corpus (built-in) so no external dataset needed.
Trains in a few minutes on CPU. Saves model + tokenizer.
"""
import os
import pickle
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Download a small built-in corpus
nltk.download("gutenberg")
from nltk.corpus import gutenberg

# Use one book as the training text (keeps it fast)
text = gutenberg.raw("austen-emma.txt").lower()
# Trim to keep training quick for a demo
text = text[:200_000]

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
print("Vocabulary size:", total_words)

# Build input sequences (n-gram style)
sequences = []
for line in text.split("."):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        sequences.append(tokens[: i + 1])

max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
X, y = sequences[:, :-1], sequences[:, -1]

print(f"Training samples: {len(X)}, max_len: {max_len}")

# Compact model
model = Sequential([
    Embedding(total_words, 64, input_length=max_len - 1),
    LSTM(128),
    Dense(total_words, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(X, y, epochs=20, batch_size=256, verbose=1)

# Save
model.save(f"{MODEL_DIR}/lstm_nextword.h5")
with open(f"{MODEL_DIR}/lstm_tokenizer.pkl", "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_len": max_len}, f)

print("\nSaved LSTM model and tokenizer to", MODEL_DIR)