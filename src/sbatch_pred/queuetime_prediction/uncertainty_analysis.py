import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from sbatch_pred.queuetime_prediction.model_training import get_model_data, partition_params

from scipy.stats import iqr

def train_model(model_data_df, train_window, split_time, features, target, partition, params=None, verbose=False):
    train_condition = model_data_df.start_time.between(split_time - pd.Timedelta(days=train_window), split_time, inclusive='left')
    partition_condition = model_data_df.partition == partition
    train_condition &= partition_condition
        
    train_df = model_data_df[train_condition].copy()

    X_train = train_df[features].copy()
    y_train = train_df[target]

    if params:
        #print('in train', partition, params)
        model = xgb.XGBRegressor(**params)
    else:
        model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    return model


def test_model(model_data_df, model, test_window, split_time, features, target, partition, noise_magnitude=0, verbose=False):
    test_condition = model_data_df.submit_time.between(split_time, split_time + pd.Timedelta(days=test_window))
    partition_condition = model_data_df.partition == partition
    test_condition &= partition_condition
        
    test_df = model_data_df[test_condition].copy()
    
    if len(test_df) == 0:
        return None

    X_test = test_df[features].copy()
    y_test = test_df[target]

    results_df = test_df.copy()
    results_df['split_time'] = split_time

    
    if noise_magnitude == 0:
        num_perturbations = 1
    else:
        num_perturbations = 100
    predictions = np.zeros((test_df.shape[0], num_perturbations))

    for i in range(num_perturbations):
        perturbation = np.random.uniform(-noise_magnitude, noise_magnitude, X_test.shape) * X_test       
        X_test_perturbed = X_test + perturbation
        predictions[:, i] = model.predict(X_test_perturbed)
        
    results_df['queue_wait_pred'] = model.predict(X_test)
    if target == 'queue_wait_log':
        results_df['queue_wait_pred'] = np.exp(results_df['queue_wait_pred'])
    
    if noise_magnitude > 0:
        cv, iqr_med, mad_med, qcd = calculate_stats(np.exp(predictions))
        results_df['cv'] = cv
        results_df['iqr_med'] = iqr_med
        results_df['mad_med'] = mad_med
        results_df['qcd'] = qcd

    return results_df


def calculate_stats(predictions):
    cv = np.zeros(predictions.shape[0])
    iqr_med = np.zeros(predictions.shape[0])
    mad_med = np.zeros(predictions.shape[0])
    qcd = np.zeros(predictions.shape[0])
    
    for index, row in enumerate(predictions):
        mean = np.mean(row)
        std_dev = np.std(row)
        if mean != 0:
            cv[index] = std_dev / mean
        else:
            cv[index] = np.nan

        median = np.median(row)
        if median != 0:
            iqr_med[index] = iqr(row) / median
            mad_med[index] = np.median(np.abs((row - median))) / median
        else:
            iqr_med[index] = np.nan
            mad_med[index] = np.nan

        q1 = np.percentile(row, 25)
        q3 = np.percentile(row, 75)
        if q3 - q1 != 0:
            qcd[index] = (q3 - q1) / (q3 + q1)
        else:
            qcd[index] = np.nan

    return cv, iqr_med, mad_med, qcd



def get_partition_results(model_data_df, train_window, test_window, start_date, features, target, partition, n_days=120, noise_magnitude=0, verbose=False):
    results = {'train_window' : [],
               'test_window' : [],
               'split_time' : [],
               'results_df': []}

    try:
        params = partition_params[partition]
    except:
        params = None
    
    for days in range(n_days):
        split_time = pd.to_datetime(start_date) + pd.Timedelta(days=days)
        if verbose: print(split_time.date(),end=',')
        model = train_model(model_data_df, train_window, split_time, features, target, partition, params, verbose)
        results_df = test_model(model_data_df, model, test_window, split_time, features, target, partition, noise_magnitude, verbose)
    
        if results_df is not None:
            results['train_window'].append(train_window)
            results['test_window'].append(test_window)
            results['split_time'].append(split_time)
            results['results_df'].append(results_df)
    return results


def combine_partition_results(results):
    return pd.concat(results['results_df'], ignore_index=True).reset_index(drop=True)


def get_results_df(partition_results, noise_magnitude=0):
    results_dfs = []
    for partition in partition_results:
        if partition == 'off3':
            continue # No node-level information for csc partition
        if len(partition_results[partition]['results_df']) == 0:
            continue
        
        results_dfs.append(combine_partition_results(partition_results[partition]))

    results_df = pd.concat(results_dfs, ignore_index=True).reset_index(drop=True)

    return results_df