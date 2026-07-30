#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, l_prev = expectation(X, pi, m, S)
    if g is None or l_prev is None:
        return None, None, None, None, None

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            0, round(l_prev, 5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, l = expectation(X, pi, m, S)

        if verbose and (i % 10 == 0 or abs(l - l_prev) <= tol):
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(l, 5)))

        if abs(l - l_prev) <= tol:
            break

        l_prev = l

    return pi, m, S, g, l
