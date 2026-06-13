#!/usr/bin/env python3
"""Module that plots a stacked bar graph."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph of fruit quantities."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    width = 0.5

    plt.bar(
        people, apples, width=width,
        color='red', label='apples'
    )
    plt.bar(
        people, bananas, width=width,
        color='yellow', bottom=apples,
        label='bananas'
    )
    plt.bar(
        people, oranges, width=width,
        color='#ff8000', bottom=apples + bananas,
        label='oranges'
    )
    plt.bar(
        people, peaches, width=width,
        color='#ffe5b4', bottom=apples + bananas + oranges,
        label='peaches'
    )

    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend(loc='upper right')

    plt.show()
