import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Tuple, Optional
import datetime

def select_features(df: pd.DataFrame, features: List[str], hashed: bool = False, encodings: Optional[List[str]] = None) -> pd.DataFrame:
    '''
    Filter a pandas DataFrame based on selected features and optional hashing or encoding.
    
    Encodings are provided as a list of options (label encoding, one-hot encoding, and/or embedding).
    
    Parameters:
    - df (pandas.DataFrame): The pandas DataFrame to be filtered.
    - features (List[str]): A list of the selected features.
    - hashed (Boolean): Whether to retrieve hashed features. Default is False.
    - encodings (List[str]): A list of encoding types to retrieve. Default is None.
    
    Returns:
    - pandas.DataFrame: The filtered pandas DataFrame.
    '''
    filtered_df = pd.DataFrame()
    encoded_columns = ['partition','qos','accounting_qos','state','flags','reason']
    hashed_columns = ['user','account','name','work_dir','submit_line']   
    for feature in features:
        if encodings and feature in encoded_columns:
            for encoding in encodings:
                if encoding == 'embedding' and feature not in ['partition','qos']:
                    continue
                filtered_df[feature + '_' + encoding] = df[feature + '_' + encoding]
        elif hashed and feature in hashed_columns:
            filtered_df[feature + '_hash'] = eagle_data_preprocessed_df[feature + '_hash']
        else:
            filtered_df[feature] = df[feature]
    return selected_df

def group_data_by_feature(df: pd.DataFrame, feature: str) -> dict:
    '''
    Group a pandas DataFrame by a feature.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to be grouped.
    - feature (str): The name of the column to group the DataFrame by.
    
    Returns:
    - dict: A dictionary of DataFrames, where the keys are the unique values in the 
    specified column and the values are the corresponding DataFrames grouped by those values.
    '''
    
    grouped_df = df.groupby(feature)
    return {key: grouped_df.get_group(key) for key in grouped_df.groups.keys()}

def select_time_window(df: pd.DataFrame, start_time, end_time) -> pd.DataFrame:
    '''
    Select the rows in a pandas DataFrame where submit_time is between start_time and end_time.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame where the rows need to be selected.
    - start_time: The start time used to filter the DataFrame.
    - end_time: The end time used to filter the DataFrame.
    
    Returns:
    - pandas.DataFrame: A DataFrame that contains only the rows where the value in the
    'submit_time' column is greater than or equal to start_time and less than or equal to
    end_time.
    '''
    df_time_window = df[df['submit_time'].between(start_time, end_time)]
    return df_time_window

def get_recent_jobs(df: pd.DataFrame, feature: str, feature_value: str, t, n: int) -> pd.DataFrame:
    """
    Get the n most recent jobs for a pandas DataFrame, filtered by a feature, submitted before time t.
    
    This function takes a pandas DataFrame, a feature, a feature value, a time t, and a 
    number of jobs n as input and returns the most recent n jobs with the feature value 
    submitted before time t. For this function to work correctly, the DataFrame
    must be sorted by submit time.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the jobs.
    - feature (str): The feature (user/account/group) we are filtering with.
    - feature_value (str): The user/account/group whose recent jobs need to be selected.
    - t: The time before which the jobs were submitted.
    - n (int): The number of recent jobs that need to be selected.
    
    Returns:
    - pandas.DataFrame: A DataFrame that contains the most recent n jobs with the feature value,
    submitted before time t.
    """
    return df[(df[feature].values == feature_value) & (df.end_time.values < t)].tail(n)

def normalize_columns(df: pd.DataFrame, columns: List[str], method: str = "StandardScaler") -> Tuple[pd.DataFrame, StandardScaler, MinMaxScaler or RobustScaler]:
    """
    Normalize a set of columns in a Pandas DataFrame using sklearn StandardScaler, MinMaxScaler, or RobustScaler.
    
    Parameters:
    - df (pd.DataFrame): The pandas DataFrame.
    - columns (List[str]): List of columns to normalize.
    - method (str): Method of normalization. Can be either "StandardScaler", "MinMaxScaler", or "RobustScaler". 
    Default is "StandardScaler".
    
    Returns:
    - Tuple[pd.DataFrame, StandardScaler, MinMaxScaler, or RobustScaler]: Tuple of DataFrame with normalized columns and the scaler 
    object used for normalization.
    """
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid normalization method. Please choose either 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'")
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def get_similar_jobs(df: pd.DataFrame, job_index: int, features: List[str], n: int = 1, weights: Optional[List[float]] = None, filter_feature: Optional[str] = None, filter_feature_val: Optional[str] = None) -> pd.DataFrame:
    """
    As compared to a specific job in a pandas DataFrame, retrieve the n most similar jobs.
    
    For best results, the features should be normalized. Results can be filtered by a feature value 
    (e.g. for a single user). Features can be weighted to adjust their relative importance in the 
    similarity calculation. The order of the returned jobs is indeterminate.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing all the jobs.
    - job_index (int): The index of the job to compare with other jobs.
    - features (List[str]): A list of the features to use in the similarity calculation.
    - n (int): The number of similar jobs to return. Default is 1.
    - weights (Optional[List[float]]): A list of weights to apply to each feature. Default is None.
    - filter_feature (Optional[str]): The feature used to filter the results. Default is None.
    - filter_feature_val (Optional[str]): The value of the filter feature. Default is None.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the similarity scores between the specified job and
    all other jobs in the DataFrame.
    """
    if filter_feature:
        filtered_df = df[df[filter_feature].values == filter_feature_val][features]
    else:
        filtered_df = df[features]
    indices = filtered_df.index
    job_row = filtered_df.loc[[job_index]].to_numpy()[0]
    filtered_df_array = filtered_df.to_numpy()
    if weights:
        filtered_df_array *= weights
        similarity = np.linalg.norm(filtered_df_array - job_row*weights, axis=-1)
    else:
        similarity = np.linalg.norm(filtered_df_array - job_row, axis=-1)
    similar_job_indices = np.argpartition(similarity, n+1)[:n+1]
    return df.loc[indices[similar_job_indices]].drop(job_index)


def train_test_split(df: pd.DataFrame, split_time: float, training_window: int, testing_window: int):
    '''
    Split a pandas DataFrame by a specific split time.
    
    This functioon takes a pandas DataFrame and a split time as input parameters.
    It splits the DataFrame into two DataFrames, where one DataFrame has all rows
    where the value in the 'end_time' column is less than or equal to the split time
    and one DataFrame where the value in the 'submit_time' column is greater than
    the split time. This simulates a real situation, where we can know the run time
    of jobs that have already ended by the split time, and use this knowledge to
    predict the runtime of jobs that are submitted after the split time.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to be split.
    - split_time (int or float): The time used to split the DataFrame. It is a float
    if it has been normalized.
    
    Returns:
    - Tuple (pandas.DataFrame, pandas.DataFrame): A tuple of the train and test dataframes
    '''
    
    train_df = df[df.end_time.between(split_time - datetime.timedelta(training_window), split_time, inclusive='left')].dropna()
    test_df = df[df.submit_time.between(split_time, split_time + datetime.timedelta(testing_window))].dropna()
    return train_df, test_df