#!/usr/bin/env python3
"""Symmetric P affinities of a data set for t-SNE."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) to be transformed by t-SNE.
        tol: maximum tolerance allowed for the difference in Shannon
            entropy from perplexity.
        perplexity: the perplexity all Gaussian distributions should have.

    Returns:
        P: numpy.ndarray of shape (n, n) with the symmetric P affinities.
    """
    n = X.shape[0]
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low, high = None, None
        Di = np.append(D[i, :i], D[i, i + 1:])
        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + low) / 2
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    return (P + P.T) / (2 * n)
