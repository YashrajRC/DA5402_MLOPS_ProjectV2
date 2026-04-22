"""Feature engineering: TF-IDF + hand-crafted numeric features."""
import re
import string

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Extracts text length, punctuation density, sentiment, first-person ratio."""

    def __init__(self):
        self.sia = None

    def fit(self, X, y=None):
        self.sia = SentimentIntensityAnalyzer()
        return self

    def _extract_one(self, text: str):
        text = str(text)
        tokens = text.split()
        n_tokens = max(len(tokens), 1)
        n_chars = max(len(text), 1)

        punct_count = sum(1 for c in text if c in string.punctuation)
        upper_chars = sum(1 for c in text if c.isupper())
        first_person = sum(1 for t in tokens if t.lower() in FIRST_PERSON)
        exclaim = text.count("!")
        question = text.count("?")
        vader = self.sia.polarity_scores(text) if self.sia else {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}

        return [
            len(text),
            n_tokens,
            np.mean([len(t) for t in tokens]) if tokens else 0,
            punct_count / n_chars,
            upper_chars / n_chars,
            first_person / n_tokens,
            exclaim,
            question,
            vader["compound"],
            vader["pos"],
            vader["neg"],
            vader["neu"],
        ]

    def transform(self, X):
        if self.sia is None:
            self.sia = SentimentIntensityAnalyzer()
        return np.array([self._extract_one(t) for t in X], dtype=np.float32)


def build_tfidf(max_features: int, ngram_range):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )


def combine_features(tfidf_matrix, handcrafted):
    """hstack sparse TF-IDF with dense handcrafted features."""
    return hstack([tfidf_matrix, csr_matrix(handcrafted)])
