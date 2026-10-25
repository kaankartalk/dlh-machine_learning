#!/usr/bin/env python3
"""Bag of words embedding"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

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

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, words in enumerate(tokens):
        for word in words:
            if word in index:
                embeddings[i, index[word]] += 1

    return embeddings, features
