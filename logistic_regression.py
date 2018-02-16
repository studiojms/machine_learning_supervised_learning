#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

movies = pd.read_csv('data/users_rate.csv')

total_columns = len(movies.columns)

movies_characteristics = movies[movies.columns[1:total_columns - 1]]
movies_user_rates = movies[movies.columns[-1:]]

train, test, train_rates, test_rates = train_test_split(
    movies_characteristics, movies_user_rates)

train_columns_count = len(train.columns)

train = np.array(train).reshape(len(train), train_columns_count)
test = np.array(test).reshape(len(test), train_columns_count)

train_rates = train_rates.values.ravel()
test_rates = test_rates.values.ravel()

model = LogisticRegression()
model.fit(train, train_rates)
predictions = model.predict(test)

# calculates the predictions accuracy of logistic regression
predictions_accuracy = accuracy_score(test_rates, predictions)
print("Accuracy of Logistic Regression = {0}".format(predictions_accuracy))

# Calculates the accuracy of Multinomial NB, to compare with the logistic regression
model_NB = MultinomialNB()
model_NB.fit(train, train_rates)
predictions_NB = model_NB.predict(test)

predictions_accuracy_NB = accuracy_score(test_rates, predictions_NB)
print("Accuracy of Multinomial NB = {0}".format(predictions_accuracy_NB))


# predicting external data
movie_to_be_predicted = [0, 0, 0, 0, 0, 0,
                         0, 0, 1, 1, 1, 0, 1, 110, 27.74456356]

new_prediction = model.predict([movie_to_be_predicted])

print("The user's rate for the movie will be '{0}'".format(new_prediction[0]))
