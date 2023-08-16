#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:03:25 2023

@author: hebaish
"""
#%%
# import xgboost as xgb
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
import os
#%%
# Load datasets 
N_kel_train_path = r"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Datasets\Training\Kelmarsh_training_data"
N_pen_train_path = r"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Datasets\Training\Penmanshiel_training_data"
N_kel_test_path = r"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Datasets\Testing\Kelmarsh_test_data"
N_pen_test_path = r"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Datasets\Testing\Penmanshiel_test_data"


kel_train_path = '/Users/hebaish/Library/CloudStorage/OneDrive-TexasA&MUniversity/A&M/Academics/Spring 23/ISEN 619/Project/Datasets/Training/Kelmarsh_training_data'
pen_train_path = '/Users/hebaish/Library/CloudStorage/OneDrive-TexasA&MUniversity/A&M/Academics/Spring 23/ISEN 619/Project/Datasets/Training/Penmanshiel_training_data'
kel_test_path = '/Users/hebaish/Library/CloudStorage/OneDrive-TexasA&MUniversity/A&M/Academics/Spring 23/ISEN 619/Project/Datasets/Testing/Kelmarsh_test_data'
pen_test_path = '/Users/hebaish/Library/CloudStorage/OneDrive-TexasA&MUniversity/A&M/Academics/Spring 23/ISEN 619/Project/Datasets/Testing/Penmanshiel_test_data'

folder_paths = [kel_train_path, pen_train_path, kel_test_path, pen_test_path]
N_folder_paths = [N_kel_train_path, N_pen_train_path, N_kel_test_path, N_pen_test_path]

for folder_path in N_folder_paths:
    # loop through all files in folder_path
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            # load the CSV file as a pandas dataframe
            df = pd.read_csv(os.path.join(folder_path, file))
            
            if 'training' in file:
                df = df.dropna()
                df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')
                df['month'] = df['time'].dt.month
                df['Day.Night'] = df['time'].dt.hour.apply(lambda x: 1 if x<6 or x>=18 else 0)
                df = df.drop('time', axis = 1)
                df = shuffle(df)
                pass
            
            elif 'test' in file:
                df = df.drop('time', axis = 1)
                month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,
                             'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

                # Use the map() method to convert the month_text column to numeric format
                df['month'] = df['month'].map(month_map)

                # Define a dictionary to map 'Day' to 0 and 'Night' to 1
                day_night_map = {'Day': 0, 'Night': 1}

                # Use the map() method to convert the day_night column to numeric format
                df['Day.Night'] = df['Day.Night'].map(day_night_map)
                pass
            
            else:
                pass
            
            
            # use the filename (without the ".csv" extension) to name the dataframe variable
            df_name = file[:-4] 
            globals()[df_name] = df # store the dataframe in a variable with the same name as the filename

#%%
X = Pen_df6_training.drop('power', axis=1)
y = Pen_df6_training['power']
train_X = X.iloc[:43000]
train_y = y.iloc[:43000]
val_X = X.iloc[43000:]
val_y = y.iloc[43000:]
rf = RandomForestRegressor(n_estimators=800, random_state=42, min_samples_leaf = 1, oob_score = True)
rf.fit(train_X, train_y)
y_pred = rf.predict(val_X)
rmse = np.sqrt(mean_squared_error(val_y, y_pred))
print(f'Validation RMSE:', rmse)
#%%
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
err = []
for n in n_estimators:
    rf = RandomForestRegressor(n_estimators=800, random_state=42, min_samples_leaf = 1, oob_score = True)
    rf.fit(train_X, train_y)
    y_pred = rf.predict(val_X)
    rmse = np.sqrt(mean_squared_error(val_y, y_pred))
    err.append(rmse)
    print(f'Validation RMSE with n_estimators = {n}:', rmse)
    
plt.plot(n_estimators, err)
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.show()

#%%
# Define a list of all training datasets
training_sets = [Pen_df1_training, Pen_df2_training, Pen_df4_training, Pen_df5_training, Pen_df6_training,
                 Pen_df7_training, Pen_df8_training, Pen_df9_training, Pen_df10_training, Pen_df11_training,
                 Pen_df12_training, Pen_df13_training, Pen_df14_training, Pen_df15_training, Kel_df1_training,
                 Kel_df2_training, Kel_df4_training, Kel_df5_training, Kel_df6_training]

# Define a list of all test datasets
test_sets = [Pen_df1_test, Pen_df2_test, Pen_df4_test, Pen_df5_test, Pen_df6_test,
             Pen_df7_test, Pen_df8_test, Pen_df9_test, Pen_df10_test, Pen_df11_test,
             Pen_df12_test, Pen_df13_test, Pen_df14_test, Pen_df15_test, Kel_df1_test,
             Kel_df2_test, Kel_df4_test, Kel_df5_test, Kel_df6_test]

# Train a model for each training set and use it to predict each corresponding test set
for i in range(len(training_sets)):
    X = training_sets[i].drop('power', axis=1)
    y = training_sets[i]['power']

    # Train a model on the i-th training set
    rf = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf = 1, oob_score = True)    # Use the trained model to predict on the i-th test set
    rf.fit(X, y)
    test_sets[i] = test_sets[i].drop('power', axis=1)
    y_pred = rf.predict(test_sets[i])
    # Add an empty 'time' column to the beginning of the DataFrame
    test_sets[i].insert(0, 'power', y_pred)

    # Add the 'power' column to the beginning of the DataFrame with the array of values
    test_sets[i].insert(0, 'time', np.nan)


    test_sets[i].to_csv(r"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Submissions\RandomForrest\test_sets[i].csv", index = False)
    joblib.dump(rf, f'rf{i}_model.joblib')
#%%
xx=0
Pen_training = [Pen_df1_training, Pen_df2_training, xx, Pen_df4_training,
                Pen_df5_training, Pen_df6_training, Pen_df7_training, 
                Pen_df8_training, Pen_df9_training, Pen_df10_training, 
                Pen_df11_training, Pen_df12_training, Pen_df13_training, 
                Pen_df14_training, Pen_df15_training]

Pen_test =  [Pen_df1_test, Pen_df2_test, xx, Pen_df4_test, Pen_df5_test,
             Pen_df6_test, Pen_df7_test, Pen_df8_test, Pen_df9_test, 
             Pen_df10_test, Pen_df11_test, Pen_df12_test, Pen_df13_test, 
             Pen_df14_test, Pen_df15_test]


# Prediction for Pen
for i in range(15):
    if i != 2:
        X = Pen_training[i].drop('power', axis=1)
        y = Pen_training[i]['power']
        # Train a model on the i-th training set
        rf = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf = 1, oob_score = True)    # Use the trained model to predict on the i-th test set
        rf.fit(X, y)
        Pen_test[i] = Pen_test[i].drop('power', axis=1)
        y_pred = rf.predict(Pen_test[i])
        # Add an empty 'time' column to the beginning of the DataFrame
        Pen_test[i].insert(0, 'power', y_pred)

        # Add the 'power' column to the beginning of the DataFrame with the array of values
        Pen_test[i].insert(0, 'time', np.nan)


        Pen_test[i].to_csv(fr"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Submissions\RandomForrest\Pen_df{i+1}_test.csv", index = False)
        joblib.dump(rf, f'Pen_df{i+1}_model.joblib')
        
#%%     

Kel_training = [Kel_df1_training, Kel_df2_training, Kel_df3_training, Kel_df4_training, 
                Kel_df5_training, Kel_df6_training]

Kel_test = [Kel_df1_test, Kel_df2_test, Kel_df3_test, Kel_df4_test, Kel_df5_test, 
            Kel_df6_test]

for i in range(6):
    X = Kel_training[i].drop('power', axis=1)
    y = Kel_training[i]['power']
    # Train a model on the i-th training set
    rf = RandomForestRegressor(n_estimators=800, random_state=42, min_samples_leaf = 1, oob_score = True)    # Use the trained model to predict on the i-th test set
    rf.fit(X, y)
    Kel_test[i] = Kel_test[i].drop('power', axis=1)
    y_pred = rf.predict(Kel_test[i])
    # Add an empty 'time' column to the beginning of the DataFrame
    Kel_test[i].insert(0, 'power', y_pred)

    # Add the 'power' column to the beginning of the DataFrame with the array of values
    Kel_test[i].insert(0, 'time', np.nan)


    Kel_test[i].to_csv(fr"C:\Users\hebaish\OneDrive - Texas A&M University\A&M\Academics\Spring 23\ISEN 619\Project\Submissions\RandomForrest\Kel_df{i+1}_test.csv", index = False)
    joblib.dump(rf, f'Kel_df{i+1}_model.joblib')
    
#%%