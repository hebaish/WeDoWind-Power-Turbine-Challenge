# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 08:12:27 2023

@author: hebaish
"""

#%%
import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.utils import shuffle
import joblib  # Import joblib directly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
#%%
#Loading and transforming data
train_data = pd.read_csv("Pen_df1_training.csv")
val_data = pd.read_csv("Pen_df2_training.csv")
#Clean data by dropping rows of missing values
train_data = train_data.dropna()
val_data = val_data.dropna()
#Converting timestamp into datetimelike format
train_data['time'] = pd.to_datetime(train_data['time'], format='%m/%d/%Y %H:%M')
val_data['time'] = pd.to_datetime(val_data['time'], format='%m/%d/%Y %H:%M')

#Getting the month from the date as 1-12. 
train_data['month'] = train_data['time'].dt.month
val_data['month'] = val_data['time'].dt.month

#Getting day/night and coding them as 0 for day and s1 for night.
#Day is defined as 6 am to 6 pm, and night is defined as 6 pm to 6 am. 
train_data['Day.Night'] = train_data['time'].dt.hour.apply(lambda x: 1 if x<6 or x>=18 else 0)
val_data['Day.Night'] = val_data['time'].dt.hour.apply(lambda x: 1 if x<6 or x>=18 else 0)

train_data = train_data.drop('time', axis = 1)
val_data = val_data.drop('time', axis = 1)
train_data = shuffle(train_data)
val_data = shuffle(val_data)

test_data = pd.read_csv("Pen_df1_test.csv")
test_data = test_data.drop(['time', 'power'], axis = 1)

month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
             'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Use the map() method to convert the month_text column to numeric format
test_data['month'] = test_data['month'].map(month_map)

# Define a dictionary to map 'Day' to 0 and 'Night' to 1
day_night_map = {'Day': 0, 'Night': 1}

# Use the map() method to convert the day_night column to numeric format
test_data['Day.Night'] = test_data['Day.Night'].map(day_night_map)

train_pred = train_data.drop('power', axis=1)
train_target = train_data['power']

val_pred = val_data.drop('power', axis=1)
val_target = val_data['power']

#%%
model = keras.Sequential([
    layers.Dense(64, activation="sigmoid", input_shape=(train_pred.shape[1],)),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="mse")

history = model.fit(train_pred, train_target, epochs=100, batch_size=128, validation_data=(val_pred, val_target))

val_predictions = model.predict(val_pred)
train_rmse = mean_squared_error(train_target, model.predict(train_pred), squared=False)
val_rmse = mean_squared_error(val_target, val_predictions, squared=False)
print('Training RMSE: {:.5f}'.format(train_rmse))
print('Validation RMSE: {:.5f}'.format(val_rmse))

test_predictions = model.predict(test_data)

