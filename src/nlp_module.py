"""
Step 7b: NLP helpers - preprocessing, sentiment, next-word prediction.
"""
import re
import pickle
import numpy as np
import nltk
from textblob import TextBlob

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))


def preprocess(text):
    """Lowercase, strip punctuation, remove stopwords. Returns dict of steps."""
    lower = text.lower()
    no_punct = re.sub(r"[^a-z\s]", "", lower)
    tokens = no_punct.split()
    no_stop = [t for t in tokens if t not in STOP]
    return {
        "lowercased": lower,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "without_stopwords": no_stop,
    }


def sentiment(text):
    """Returns polarity (-1..1) and subjectivity (0..1)."""
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    label = "Positive" if pol > 0.05 else "Negative" if pol < -0.05 else "Neutral"
    return {"polarity": pol, "subjectivity": subj, "label": label}


_lstm = None
_tok = None
_maxlen = None


def _load_lstm():
    global _lstm, _tok, _maxlen
    if _lstm is None:
        import tensorflow as tf
        _lstm = tf.keras.models.load_model("models/lstm_nextword.h5")
        with open("models/lstm_tokenizer.pkl", "rb") as f:
            d = pickle.load(f)
        _tok = d["tokenizer"]
        _maxlen = d["max_len"]
    return _lstm, _tok, _maxlen


def predict_next_words(seed, n_words=5):
    """Generate n_words continuation from a seed string."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tok, maxlen = _load_lstm()
    result = seed
    for _ in range(n_words):
        seq = tok.texts_to_sequences([result])[0]
        seq = pad_sequences([seq], maxlen=maxlen - 1, padding="pre")
        pred = np.argmax(model.predict(seq, verbose=0), axis=-1)[0]
        word = next((w for w, i in tok.word_index.items() if i == pred), "")
        if not word:
            break
        result += " " + word
    return result