"""
utils.py — Shared text preprocessing utilities for the Fake News Detector.

This module provides a consistent preprocessing pipeline used by both
the training script (train_model.py) and the Streamlit app (app.py).
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


# Auto-download on import
ensure_nltk_data()

# Initialize once at module level for performance
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def preprocess_text(text: str, use_stemming: bool = False) -> str:
    """
    Clean and normalize a piece of text for NLP processing.

    Steps:
        1. Lowercase the entire text
        2. Remove URLs
        3. Remove punctuation and digits
        4. Remove English stopwords
        5. (Optional) Apply Porter stemming

    Args:
        text: Raw text string to preprocess.
        use_stemming: If True, apply Porter stemming to each token.

    Returns:
        A cleaned, space-separated string of tokens.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove punctuation and digits
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)

    # 4. Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in _stop_words and len(word) > 1]

    # 5. Optional stemming
    if use_stemming:
        tokens = [_stemmer.stem(word) for word in tokens]

    # 6. Collapse whitespace
    return " ".join(tokens)
