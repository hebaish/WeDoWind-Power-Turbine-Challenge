# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:05:21 2023

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
#%%
#Loading and transforming data
train_data = pd.read_csv("Pen_df1_training.csv")
test_data = pd.read_csv("Pen_df2_training.csv")
#Clean data by dropping rows of missing values
train_data = train_data.dropna()
test_data = test_data.dropna()
#Converting timestamp into datetimelike format
train_data['time'] = pd.to_datetime(train_data['time'], format='%m/%d/%Y %H:%M')
test_data['time'] = pd.to_datetime(test_data['time'], format='%m/%d/%Y %H:%M')

#Getting the month from the date as 1-12. 
train_data['month'] = train_data['time'].dt.month
test_data['month'] = test_data['time'].dt.month

#Getting day/night and coding them as 0 for day and s1 for night.
#Day is defined as 6 am to 6 pm, and night is defined as 6 pm to 6 am. 
train_data['Day.Night'] = train_data['time'].dt.hour.apply(lambda x: 1 if x<6 or x>=18 else 0)
test_data['Day.Night'] = test_data['time'].dt.hour.apply(lambda x: 1 if x<6 or x>=18 else 0)

train_data = train_data.drop('time', axis = 1)
test_data = test_data.drop('time', axis = 1)
train_data = shuffle(train_data)
test_data = shuffle(test_data)

final_test_data = pd.read_csv("Pen_df1_test.csv")
final_test_data = final_test_data.drop(['time', 'power'], axis = 1)

month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
             'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

# Use the map() method to convert the month_text column to numeric format
final_test_data['month'] = final_test_data['month'].map(month_map)

# Define a dictionary to map 'Day' to 0 and 'Night' to 1
day_night_map = {'Day': 0, 'Night': 1}

# Use the map() method to convert the day_night column to numeric format
final_test_data['Day.Night'] = final_test_data['Day.Night'].map(day_night_map)

#%%
# Define predictors and target
X = train_data.drop('power', axis=1)
y = train_data['power']

train_X = X.iloc[:30000]
train_y = y.iloc[:30000]
val_X = X.iloc[30000:]
val_y = y.iloc[30000:]

# Define the number of folds for cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize a list to store the RMSE scores for each fold
rmse_scores = []

# Loop through each fold
for train_idx, test_idx in kfold.split(X):
    
    # Split the data into training and testing sets for the current fold
    X_train, X_test = train_X.iloc[train_idx], train_X.iloc[test_idx]
    y_train, y_test = train_y.iloc[train_idx], train_y.iloc[test_idx]
    
    # Initialize a random forest regressor♣
    rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf = 1)
    
    # Fit the regressor to the training data
    rf.fit(X_train, y_train)
    
    # Predict the target values for the testing data
    y_pred = rf.predict(X_test)
    
    # Calculate the RMSE score for the testing data
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Append the RMSE score to the list of scores
    rmse_scores.append(rmse)

# Calculate the mean RMSE score across all folds
mean_rmse = np.mean(rmse_scores)

print('Training RMSE:', mean_rmse)
y_pred_val = rf.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred_val))
print('Validation RMSE:', rmse)
# Save the trained model to a file
# joblib.dump(rf, 'rf2_model.joblib')

#%%
# Load the data
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop('power', axis=1), 
                                                  train_data['power'], 
                                                  test_size=0.03, 
                                                  random_state=42)

# Define the parameter space to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a RandomForest regressor object
rf = RandomForestRegressor(random_state=42)

# Create a RandomizedSearchCV object
rf_random = RandomizedSearchCV(estimator=rf, 
                               param_distributions=param_grid, 
                               n_iter=100, 
                               cv=10, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs=-1)

# Fit the RandomizedSearchCV object to the training data
rf_random.fit(X_train, y_train)

# Get the best hyperparameters found
best_params = rf_random.best_params_

# Train a new RandomForest regressor with the best hyperparameters on the entire training set
rf = RandomForestRegressor(**best_params, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = rf.predict(X_val)

# Calculate the RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE on validation set:", rmse_val)

# Make predictions on the test set (not used in hyperparameter tuning)
X_test = test_data.drop('power', axis=1)
y_test = test_data['power']
y_test_pred = rf.predict(X_test)

# Calculate the RMSE on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("RMSE on test set:", rmse_test)

# Make predictions on the final test set (not used in training or hyperparameter tuning)
X_final_test = final_test_data
final_test_pred = rf.predict(X_final_test)

# Add the predicted values to the final test dataset
final_test_data.insert(0, 'predicted_power', final_test_pred)

# Save the final test dataset to a CSV file
final_test_data.to_csv('final_test_results2.csv', index=False)
