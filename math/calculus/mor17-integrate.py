#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    result = [C]

    for i, coef in enumerate(poly):
        new_coef = coef / (i + 1)
        if new_coef == int(new_coef):
            new_coef = int(new_coef)
        result.append(new_coef)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
