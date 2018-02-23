#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score

movies = pd.read_csv('data/movies_multilinear_regression.csv')

total_columns = len(movies.columns)

movies_independent_variable = movies[movies.columns[2:total_columns - 1]]
movies_result_variable = movies[movies.columns[-1:]]

train, test, train_rates, test_rates = train_test_split(
    movies_independent_variable, movies_result_variable)

train_columns_count = len(train.columns)

train = np.array(train).reshape(len(train), train_columns_count)
test = np.array(test).reshape(len(test), train_columns_count)

train_rates = train_rates.values.ravel()
test_rates = test_rates.values.ravel()

regression_model = DecisionTreeRegressor(max_depth=5)
regression_model.fit(train, train_rates)

regression_model.score(train, train_rates)

regression_model.score(test, test_rates)


# Classification

classification_movies = pd.read_csv('data/users_rate.csv')

total_columns = len(classification_movies.columns)

movies_characteristics = classification_movies[classification_movies.columns[1:total_columns - 1]]
movies_user_rates = classification_movies[classification_movies.columns[-1:]]

class_train, class_test, class_train_rates, class_test_rates = train_test_split(
    movies_characteristics, movies_user_rates)

class_train_columns_count = len(class_train.columns)

class_train = np.array(class_train).reshape(
    len(class_train), class_train_columns_count)
class_test = np.array(class_test).reshape(
    len(class_test), class_train_columns_count)

class_train_rates = class_train_rates.values.ravel()
class_test_rates = class_test_rates.values.ravel()


classifier_model = DecisionTreeClassifier(max_depth=5)
classifier_model.fit(class_train, class_train_rates)

classifier_model.score(class_train, class_train_rates)

classifier_model.score(class_test, class_test_rates)
