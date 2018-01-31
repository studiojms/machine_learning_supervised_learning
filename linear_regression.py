#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#import train_test_split

movies = pd.read_csv('data/linear_regression.csv')

SAMPLE_SIZE = 200

sample = movies.sample(n=SAMPLE_SIZE)
sample_x = sample['Investiment']
sample_y = sample['Gross']

plt.scatter(sample_x, sample_y)
plt.show()
