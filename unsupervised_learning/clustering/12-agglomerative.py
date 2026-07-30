#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                             criterion='distance')

    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
