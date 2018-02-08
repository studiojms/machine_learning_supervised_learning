#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Reading data
movies = pd.read_csv('data/linear_regression.csv')

#Exploring a sample of the data
SAMPLE_SIZE = 200

sample = movies.sample(n=SAMPLE_SIZE)
sample_x = sample['Investiment']
sample_y = sample['Gross']

#Show data in graph
plt.scatter(sample_x, sample_y)
plt.show()

#Splitting data (dependent and inpendent)
movies_investiment = movies['Investiment']
movies_gross = movies['Gross']

#Splitting data for train and test
TEST_SIZE=0.25
train, test, mark_train, mark_test = train_test_split(movies_investiment, movies_gross, test_size=TEST_SIZE)

train = np.array(train).reshape(len(train), 1)
test = np.array(test).reshape(len(test), 1)
mark_train = np.array(mark_train).reshape(len(mark_train), 1)
mark_test = np.array(mark_test).reshape(len(mark_test), 1)

#value_to_be_predicted = 27.74456356
value_to_be_predicted = 145.5170642

#Creating the model
model = LinearRegression()
model.fit(train, mark_train)

#r^2 - the score indicates how could the model explain the data
score = model.score(train, mark_train)
test_score = model.score(test, mark_test)

coeficient_val = model.coef_
intercept_val = model.intercept_


predicted_val = model.predict(value_to_be_predicted)

calculated_gross = coeficient_val * value_to_be_predicted + intercept_val

print("The aproximated gross is {0}".format(calculated_gross))


#predicting value

new_movie = [0,0,0,0,0,0,0,0,1,1,1,0,1,145.5170642,3.451632127]
model.predict([new_movie])