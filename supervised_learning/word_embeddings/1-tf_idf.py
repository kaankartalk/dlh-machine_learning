#!/usr/bin/env python3
"""TF-IDF embedding"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding

    sentences: list of sentences to analyze
    vocab: list of vocabulary words to use for the analysis
           If None, all words within sentences are used

    Returns: embeddings, features
        embeddings: numpy.ndarray of shape (s, f) with the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features: the features used for embeddings
    """
    tokens = []
    for sentence in sentences:
        cleaned = re.sub(r"'s\b", "", sentence.lower())
        tokens.append(re.findall(r"\w+", cleaned))

    if vocab is None:
        vocab = sorted({word for words in tokens for word in words})

    features = np.array(vocab)
    index = {word: i for i, word in enumerate(vocab)}

    s = len(sentences)
    f = len(vocab)

    # term frequency (raw counts)
    tf = np.zeros((s, f))
    for i, words in enumerate(tokens):
        for word in words:
            if word in index:
                tf[i, index[word]] += 1

    # smoothed inverse document frequency
    df = np.count_nonzero(tf, axis=0)
    idf = np.log((1 + s) / (1 + df)) + 1

    # tf-idf with l2 normalization of each row
    embeddings = tf * idf
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
