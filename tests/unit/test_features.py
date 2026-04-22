"""Unit tests for feature engineering."""
import numpy as np

from src.training.features import HandcraftedFeatures, build_tfidf


def test_handcrafted_shape():
    hf = HandcraftedFeatures()
    hf.fit(["hello world"])
    out = hf.transform(["hello world", "another sample text!"])
    assert out.shape == (2, 12)
    assert out.dtype == np.float32


def test_handcrafted_finite():
    hf = HandcraftedFeatures()
    hf.fit([""])
    out = hf.transform(["", "short", "a much longer sample text with exclamation!"])
    assert np.all(np.isfinite(out))


def test_tfidf_shape():
    vec = build_tfidf(max_features=100, ngram_range=[1, 2])
    X = vec.fit_transform(["hello world", "hello there", "world is big"])
    assert X.shape[0] == 3
    assert X.shape[1] <= 100
