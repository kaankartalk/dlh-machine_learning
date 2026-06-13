#!/usr/bin/env python
"""The Module that plots all 5 previous graphs in one figure."""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """The function represents plotting  graphs in one figure."""
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)


    font_size = "x-small"

    plt.figure(figsize=(10, 8))
    plt.suptitle('All in One')

    #Graph1
    plt.subplot2grid((3,2), (0,0))
    x = np.arange(0, 11)
    plt.plot(x, yO, color='red', linestyle='-')
    plt.xlim(0, 10)
    plt.tick_params(labelsize = font_size)

    #Graph2
    plt.subplot2grid((3,2), (0,1))
    plt.scatter(x1,y1, color='magenta')
    plt.xlabel("Height (in)", fontsize = font_size )
    plt.ylabel("Weight (lbs)", fontsize = font_size)
    plt.title("Men's Height vs Weight", fontsize = font_size)
    plt.tick_params(labelsize = font_size)

    #Graph3
    plt.subplot2grid((3,2), (1,0))
    plt.plot(x2, y2)
    plt.xlabel("Time (years)", fontsize = font_size)
    plt.ylabel("Fraction Remaining", fontsize = font_size)
    plt.title("Exponential Decay of C-14", fontsize = font_size)
    plt.xlim(0, 28650)
    plt.yscale('log')
    plt.tick_params(labelsize = font_size)

    #Graph4
    plt.subplot2grid((3,2), (1,1))
    plt.xlabel("Time (years)", fontsize = font_size)
    plt.ylabel("Fraction Remaining", fontsize = font_size)
    plt.title("Exponential Decay of Radioactive Elements", fontsize = font_size)
    plt.xlim(0, 20000)
    plt.ylim(0,1)
    plt.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    plt.plot(x3,y32, color='green', linestyle='solid', label='Ra-226')
    plt.legend(loc='upper right')
    plt.tick_params(labelsize = font_size)

    #Graph5
    plt.subplot2grid((3,2), (2,0), colspan=2)
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.title("Project A", fontsize = font_size)
    plt.xlabel("Grades", fontsize = font_size)
    plt.ylabel("Number of Students", fontsize = font_size)
    plt.xticks(bins)
    plt.tick_params(labelsize = font_size)
