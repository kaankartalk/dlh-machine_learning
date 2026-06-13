#!/usr/bin/env python3
"""The Module that plots a stacked bar graph."""
import numpy as np
import matplotlib.pyplot as plt

def bars():
    """ The function represents plotting a stacked bar graph."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # X-axis labels representing the people
    people = ['Farrah', 'Fred', 'Felicia']
    
    # Extracting fruit quantities from the matrix rows
    apples = fruit[0]   # 1st row: Apples
    bananas = fruit[1]  # 2nd row: Bananas
    oranges = fruit[2]  # 3rd row: Oranges
    peaches = fruit[3]  # 4th row: Peaches

    # Setting the bar width as requested
    width = 0.5

    # 1st LAYER: Apples (At the very bottom, no base needed)
    plt.bar(people, apples, width=width, color='red', label='apples')

    # 2nd LAYER: Bananas (Stacked directly on top of apples)
    plt.bar(people, bananas, width=width, color='yellow', bottom=apples, label='bananas')

    # 3rd LAYER: Oranges (Stacked on top of apples + bananas combined)
    plt.bar(people, oranges, width=width, color='#ff8000', bottom=apples + bananas, label='oranges')

    # 4th LAYER: Peaches (Stacked at the very top on top of all previous fruits)
    plt.bar(people, peaches, width=width, color='#ffe5b4', bottom=apples + bananas + oranges, label='peaches')

    # Setting y-axis range from 0 to 80 and defining ticks every 10 units
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Adding axis labels and plot title
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    
    # Adding the legend to indicate fruit colors
    plt.legend(loc='upper right')

    # Render the plot
    plt.show()
