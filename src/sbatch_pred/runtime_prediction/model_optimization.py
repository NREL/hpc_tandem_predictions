import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import datetime
import os

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from category_encoders.glmm import GLMMEncoder

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from eagle_jobs.operation_support import train_test_split
from eagle_jobs.operation_support import normalize_columns

from eagle_jobs.data_preprocessing import label_encode_columns
from eagle_jobs.data_preprocessing import onehot_with_other


import warnings
warnings.filterwarnings('ignore')

def optimize_training_window(df, split_times, model_type='XGBoost'):
    """
    This function optimizes the training window for a given model type by evaluating its performance on different training window sizes. The model types supported are XGBoost, Neural Networks (NN), and Term Frequency-Inverse Document Frequency (TFIDF). The function calculates the R2 score and Root Mean Squared Error (RMSE) for each combination of split time and training window size, and returns dictionaries containing these values.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for training and testing the model.
    - split_times (list): A list of split times to be used for splitting the data into training and testing sets.
    - model_type (str, optional): The type of model to be used for training and evaluation. Supported values are 'XGBoost', 'NN', and 'TFIDF'. Defaults to 'XGBoost'.

    Returns:
    - tuple: A tuple containing two dictionaries, the first with the R2 score for each combination of split time and training window size, and the second with the corresponding RMSE values.
    """
    
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 30
    training_windows = [1] + list(range(5,181,5))
    for training_window in training_windows:
        r2_dict[training_window] = list()
        rmse_dict[training_window] = list()
    
    for split_time in split_times:
        for training_window in training_windows:
            train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
            if len(train_df) < 2 or len(test_df) < 2:
                continue
            train_features = train_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            test_features = test_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            if model_type == 'XGBoost':
                model = xgb.XGBRegressor()
                model.fit(train_features, train_target)
            elif model_type == 'NN':
                model = keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=[train_features.shape[1]]),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(1)
                ])
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
                history = model.fit(train_features, train_target, batch_size=10000, epochs=10,\
                                validation_data=(test_features, test_target), callbacks=[early_stopping], verbose=0)
            elif model_type == 'TFIDF':
                features = ['wallclock_req_XSD_duration','nodes_req','processors_req','gpus_req','mem_req','user','account','partition','qos','work_dir','name']
                X_train = ''

                for feature in features:
                    X_train += train_df[feature].astype('str')
                y_train = train_df.run_time.values
                vect = TfidfVectorizer(max_features=600)
                X_train_vec = vect.fit_transform(X_train.values)
                
                model = LinearRegression(n_jobs=-1)
                model.fit(X_train_vec, y_train)
                
                X_test = ''
                for feature in features:
                    X_test += test_df[feature].astype('str')
                X_test = X_test.values
                test_features = vect.transform(X_test)
                test_target = test_df.run_time.values
                
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[training_window].append(r2)
            rmse_dict[training_window].append(rmse)
            
            print(f'Split time: {split_time}, training window: {training_window}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_testing_window(df, split_times, training_window):
    """
    This function optimizes the testing window for an XGBoost model by evaluating its performance on different testing window sizes. The function calculates the R2 score and Root Mean Squared Error (RMSE) for each combination of split time and testing window size, and returns dictionaries containing these values.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for training and testing the model.
    - split_times (list): A list of split times to be used for splitting the data into training and testing sets.
    - training_window (int): The size of the training window to be used for training the model.

    Returns:
    - tuple: A tuple containing two dictionaries, the first with the R2 score for each combination of split time and testing window size, and the second with the corresponding RMSE values.
    """
    
    r2_dict = dict()
    rmse_dict = dict()
    testing_windows = range(1,61)
    for testing_window in testing_windows:
        r2_dict[testing_window] = list()
        rmse_dict[testing_window] = list()
    
    for split_time in split_times:
        train_df = df[df.end_time.between(split_time - datetime.timedelta(training_window),\
                                          split_time, inclusive='left')]
        if len(train_df) < 2:
            continue
        train_features = train_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
        train_target = train_df['run_time']
        
        model = xgb.XGBRegressor()
        model.fit(train_features, train_target)
            
        for testing_window in testing_windows:
            test_df = df[df.submit_time.between(split_time + datetime.timedelta(testing_window - 1),\
                                                split_time + datetime.timedelta(testing_window))]
            if len(test_df) < 2:
                continue
            test_features = test_df[['wallclock_req','nodes_req','processors_req','gpus_req']]
            test_target = test_df['run_time']
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[testing_window].append(r2)
            rmse_dict[testing_window].append(rmse)
            
            print(f'Split time: {split_time}, testing window: {testing_window}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_numerical_features(df, split_times, training_window, testing_window):
    """
    This function optimizes the numerical features used in an XGBoost model by evaluating its performance on different combinations of numerical features. The function calculates the R2 score and Root Mean Squared Error (RMSE) for each combination of split time and feature set, and returns dictionaries containing these values.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for training and testing the model.
    - split_times (list): A list of split times to be used for splitting the data into training and testing sets.
    - training_window (int): The size of the training window to be used for training the model.
    - testing_window (int): The size of the testing window to be used for testing the model.

    Returns:
    - tuple: A tuple containing two dictionaries, the first with the R2 score for each combination of split time and feature set, and the second with the corresponding RMSE values.
    """
    
    features = ['nodes_req', 'processors_req', 'gpus_req', 'mem_req']
    feature_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(features, r) for r in range(0, len(features) + 1)))
    feature_combinations = [('wallclock_req',) + f for f in feature_combinations]
    
    r2_dict = dict()
    rmse_dict = dict()
    for f in feature_combinations:
        r2_dict[f] = list()
        rmse_dict[f] = list()
    
    for split_time in split_times:
        train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
        if len(train_df) < 2 or len(test_df) < 2:
            continue
        
        for f in feature_combinations:
            train_features = train_df[list(f)]
            test_features = test_df[list(f)]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            model = xgb.XGBRegressor()
            model.fit(train_features, train_target)
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[f].append(r2)
            rmse_dict[f].append(rmse)
            
            print(f'Split time: {split_time}, features: {f}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict



def optimize_categorical_features(df, split_times, training_window, testing_window, numerical_features, encoding):
    """
    This function optimizes the categorical features used in an XGBoost model by evaluating its performance on different combinations of categorical features and encoding techniques. The function calculates the R2 score and Root Mean Squared Error (RMSE) for each combination of split time, feature set, and encoding type, and returns dictionaries containing these values.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for training and testing the model.
    - split_times (list): A list of split times to be used for splitting the data into training and testing sets.
    - training_window (int): The size of the training window to be used for training the model.
    - testing_window (int): The size of the testing window to be used for testing the model.
    - numerical_features (list): A list of numerical features to be included in the model.
    - encoding (str): The encoding technique to be used for categorical features. Supported values are 'label', 'onehot', and 'target'.

    Returns:
    - tuple: A tuple containing two dictionaries, the first with the R2 score for each combination of split time, feature set, and encoding type, and the second with the corresponding RMSE values.
    """
    
    features = ['user','account','partition','qos','work_dir','name']
    
    if encoding == 'label':
        label_encode_columns(df, features)
    elif encoding == 'onehot':
        n_values = [20,15,6,3,20,20]
        encoded_column_names = onehot_with_other(df, features, n_values)
    
    feature_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(features, r) for r in range(0, len(features) + 1)))
    feature_combinations = [numerical_features + f for f in feature_combinations]
    
    r2_dict = dict()
    rmse_dict = dict()
    for fc in feature_combinations:
        r2_dict[fc] = list()
        rmse_dict[fc] = list()
    
    for split_time in split_times:
        train_df, test_df = train_test_split(df, split_time, training_window, testing_window)
        if len(train_df) < 2 or len(test_df) < 2:
            continue
                
        if encoding == 'target':
            encoder = GLMMEncoder(cols=features, random_state=42)
            train_encoded = encoder.fit_transform(train_df[list(numerical_features)+features], train_df.run_time)
            test_encoded = encoder.transform(test_df[list(numerical_features)+features])
            train_df = pd.concat([train_encoded, train_df.run_time], axis=1)
            test_df = pd.concat([test_encoded, test_df.run_time], axis=1)
            
        for fc in feature_combinations:
            if encoding == 'onehot':
                categorical_features = list(set(fc) & set(features))
                categorical_columns = list()
                for cf in categorical_features:
                    categorical_columns += encoded_column_names[cf]
                numerical_features = list(set(fc) - set(features))
                final_features = numerical_features + categorical_columns
            else:
                final_features = fc
            
            train_features = train_df[list(final_features)]
            test_features = test_df[list(final_features)]
            train_target = train_df['run_time']
            test_target = test_df['run_time']
            
            model = xgb.XGBRegressor()
            model.fit(train_features, train_target)
            
            y_pred = model.predict(test_features)
            r2 = r2_score(test_target, y_pred)
            rmse = mean_squared_error(test_target, y_pred, squared=False)
            
            r2_dict[fc].append(r2)
            rmse_dict[fc].append(rmse)
            
            print(f'Encoding: {encoding}, Split time: {split_time}, features: {fc}, r2: {r2:.3f}, rmse: {rmse:.0f}')
    
    return r2_dict, rmse_dict

def average_runtime_algorithm(train_df, user_df, n):
    """
    This function calculates the average runtime of jobs for a given user based on the last 'n' jobs. If no user-specific jobs are available, the function returns the mean runtime of all jobs in the training dataset.

    Parameters:
    - train_df (pd.DataFrame): The input dataframe containing the data to be used for calculating the average runtime.
    - user_df (pd.DataFrame or None): A dataframe containing user-specific data. If None or empty, the function uses the overall dataset for calculations.
    - n (int): The number of last jobs to consider for calculating the average runtime.

    Returns:
    - float: The average runtime of the last 'n' jobs, either for a specific user or for the overall dataset. If the user-specific dataframe is empty or None, it returns a default value of 600.
    """
    
    if user_df is None or len(user_df) == 0:
        return train_df.run_time.tail(n).mean()
    avg_runtime = user_df.run_time.tail(n).mean()
    return avg_runtime

def optimize_recent_jobs(df, split_time, n_max):
    """
    This function optimizes the number of recent jobs considered for predicting the job runtime using the average runtime algorithm. The function calculates the R2 score and Root Mean Squared Error (RMSE) for each value of 'n' (number of recent jobs) up to 'n_max' and returns dictionaries containing these values.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for training and testing the average runtime algorithm.
    - split_time (str): The timestamp used for splitting the data into training and testing sets.
    - n_max (int): The maximum number of recent jobs to consider for optimizing the average runtime algorithm.

    Returns:
    - tuple: A tuple containing two dictionaries, the first with the R2 score for each value of 'n' (number of recent jobs) up to 'n_max', and the second with the corresponding RMSE values.
    """
    
    split_time = pd.Timestamp(split_time)
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 1
    training_window = 100

    train_df, test_df = train_test_split(df, split_time, training_window, testing_window)

    user_dataframes = dict()
    for user in train_df.user.unique():
        user_dataframes[user] = train_df[train_df.user == user]
    for user in test_df.user.unique():
        if user not in user_dataframes:
            user_dataframes[user] = None
            
    for n in range(1,n_max+1):
        y_test = test_df.run_time.values
        y_pred = test_df.apply(lambda x: average_runtime_algorithm(train_df, user_dataframes[x['user']], n), axis=1).values
       
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2_dict[n] = r2
        rmse_dict[n] = rmse
        print(f'Split time: {split_time}, n: {n}, r2: {r2}, rmse: {rmse}')

    return r2_dict, rmse_dict



def similar_jobs_algorithm(train_df, user_jobs, job_row, n):
    """
    This function calculates the average runtime of 'n' most similar jobs for a given job using the Euclidean distance as a similarity measure. If there are less than 'n' similar jobs available, the function considers all available jobs. If no user-specific jobs are available, the function returns the mean runtime of all jobs in the training dataset.

    Parameters:
    - train_df (pd.DataFrame): The input dataframe containing the data to be used for calculating the average runtime of similar jobs.
    - user_jobs (tuple or None): A tuple containing two arrays, the first with feature values of the user-specific jobs, and the second with the corresponding job indices. If None, the function uses the overall dataset for calculations.
    - job_row (np.array): A numpy array containing the feature values of the job for which the average runtime of similar jobs is to be calculated.
    - n (int): The number of most similar jobs to consider for calculating the average runtime.

    Returns:
    - float: The average runtime of 'n' most similar jobs for the given job for a specific user. If no user-specific jobs are available, it returns the mean runtime of all jobs in the training dataset.
    """
    
    if user_jobs is None:
        return train_df.run_time.mean()
    if len(user_jobs[0]) < n + 2:
        n = len(user_jobs[0]) - 2
        if n < 1:
            return train_df.run_time.mean()
    similarity = np.linalg.norm(user_jobs[0] - job_row, axis=-1)
    similar_job_indices = np.argpartition(similarity, n+1)[:n+1]
    ind = user_jobs[1]
    return train_df.loc[ind[similar_job_indices]].run_time.mean()

def optimize_similar_jobs(df, split_time, n_max):
    """
    This function optimizes the number of most similar jobs to consider for the similar_jobs_algorithm. It calculates the r2 score and RMSE for different values of 'n' up to 'n_max', considering a single testing window and a fixed training window.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data to be used for optimizing the number of similar jobs.
    - split_time (str or pd.Timestamp): The time at which the dataset is split into training and testing sets.
    - n_max (int): The maximum number of similar jobs to consider for optimization.

    Returns:
    - r2_dict (dict): A dictionary with keys as the number of similar jobs and values as the corresponding r2 scores.
    - rmse_dict (dict): A dictionary with keys as the number of similar jobs and values as the corresponding RMSE values.
    """
    
    df['submit_time_seconds'] = (df['submit_time'] - np.datetime64('1970-01-01T00:00:00')) // np.timedelta64(1, 's')
    df['wallclock_req_normalized'] = df['wallclock_req']
    df['submit_time_normalized'] = df['submit_time_seconds']
    features = ['nodes_req', 'processors_req', 'gpus_req', 'mem_req', 'wallclock_req_normalized', 'submit_time_normalized']
    df, scaler = normalize_columns(df, features)
    
    split_time = pd.Timestamp(split_time)
    r2_dict = dict()
    rmse_dict = dict()
    testing_window = 1
    training_window = 100

    train_df, test_df = train_test_split(df, split_time, training_window, testing_window)

    user_dataframes = dict()
    for user in train_df.user.unique():
        user_dataframes[user] = list()
        user_dataframes[user].append(train_df[train_df.user == user][features])
        user_dataframes[user].append(user_dataframes[user][0].index)
    for user in test_df.user.unique():
        if user not in user_dataframes:
            user_dataframes[user] = None
            
    for n in range(1,n_max+1):
        y_test = test_df.run_time.values
        y_pred = test_df.apply(lambda x: similar_jobs_algorithm(train_df, user_dataframes[x['user']], x[features].to_numpy().astype(float), n), axis=1)
        
        if len(y_test) < 2 or len(y_pred) < 2:
            continue
            
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2_dict[n] = r2
        rmse_dict[n] = rmse
        print(f'Split time: {split_time}, n: {n}, r2: {r2}, rmse: {rmse}')

    return r2_dict, rmse_dict