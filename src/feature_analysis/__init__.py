"""
Module for exploratory data analysis (EDA).

This module provides utilities for analyzing and visualizing data.

Functions:
- log_descriptive_stats: Function for logging descriptive statistics of a DataFrame.

Submodules:
- correlations: Module for calculating and visualizing correlations between variables.
- distributions: Module for analyzing and visualizing data distributions for the original and transformed data.
"""

import pandas as pd
from src.utils.IOHandler import IOHandler
import src.utils.logger_setup as log
from src.utils.monitoring import log_errors_and_warnings
import os
import sys
import io as pyio
import contextlib



@log_errors_and_warnings
def log_descriptive_stats(path: str, delimiter=',', id_field: str = None):
    """
    Log descriptive statistics, pandas DataFrame info, duplicates, missing values,
    object type features, and descriptive statistics of a DataFrame.

    A log file with the same file name and extension .log is created in the same file directory.
    If a log file already exists when the method is called, the results will be appended to the existing log file.

    Args:
    - path (str): Path to the data file.
    - delimiter (str, optional): Delimiter used in the data file. Default is ','.
      Supports CSV, TSV, and other delimiter-separated data files.
    - id_field (str, optional): Field name to output IDs for duplicate rows. If not provided,
      prints the duplicate row indices to the log.

    Returns:
    - None
    """
    
    file_name = path.split("/")[-1].split(".")[0]
    out_dir_name = IOHandler.get_data_file_path(os.path.dirname(path))
    os.makedirs(out_dir_name, exist_ok=True)
    logger = log.get_logger(f'descriptive_stats_{file_name}', os.path.join(out_dir_name, f'{file_name}.log'))
    
    
    df = pd.read_csv(IOHandler.get_data_file_path(path), sep=delimiter)
    
    # Log DataFrame Info
    with pyio.StringIO() as logger_stdout:
        # Redirect standard output temporarily
        # Since info() method in pandas DataFrame prints result directly to the console
        with contextlib.redirect_stdout(logger_stdout):
            df.info()
        # Get the printed output
        info_output = logger_stdout.getvalue()
        # Log the serialized output
        out_str = '\n' + '-' * 25 + 'DataFrame Info' + '-' * 25 + '\n'
        logger.info(out_str + info_output)


    # Check for duplicate rows
    duplicate_rows = df[df.duplicated()]
    out_str = '\n' + '-' * 25 + f'{len(duplicate_rows)} ROW DUPLICATES DETECTED!' + '-' * 25 + '\n'
    # Print duplicate row id's when id_field passed
    if id_field:
        if not duplicate_rows.empty:
            for index, value in duplicate_rows[id_field].items():
                out_str += f'Index: {index} | Column Name: {id_field} | Value: {value} \n'
            logger.warning(out_str)
    # Print duplicate row indices only when id_field not provided
    else:
        if not duplicate_rows.empty:
            indices = duplicate_rows.index.tolist()
            out_str += f'Duplicate row indices: {indices}\n'
            logger.warning(out_str)
        
        
    # Check missing values
    df.isna().sum()
    na_sum = df.isna().sum()
    na_sum_present = na_sum[na_sum > 0]
    if len(na_sum_present) > 0:
        out_str = '\n' + '-' * 25 + f'MISSING VALUES DETECTED!' + '-' * 25 + '\n'
        logger.warning(out_str + na_sum_present.to_json(indent=4))
        
    
    # Object type feature logging
    object_columns = df.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        out_str = '\n' + '-' * 25 + f'Object type features' + '-' * 25 + '\n'
        for column in object_columns:
            unique_values = df[column].unique()
            out_str += f'{column} ({len(unique_values)}): {unique_values} \n'
        logger.info(out_str)
    
    
    # Descriptive statistic logging
    descr_stats_df = df.describe()
    # Serialize DataFrame to JSON
    json_str = descr_stats_df.to_json(indent=4)
    out_str = '\n' + '-' * 25 + f'Descriptive statistics' + '-' * 25 + '\n'
    logger.info(out_str + json_str)

def is_numerical(data: pd.core.series.Series) -> bool:
    """
    Check if the data in a pandas Series is quantitative (numerical).

    Parameters:
    data (pandas.core.series.Series): The pandas Series to be checked.

    Returns:
    bool: True if the data is quantitative (numerical), False otherwise.
    """
    dtype = data.dtype
    if dtype == 'object' or dtype == 'datetime64' or dtype == 'bool' or isinstance(dtype, pd.CategoricalDtype):
        return False
    else:
        return True

def get_categorical_features(data: pd.DataFrame) -> list:
    """
    Extract the names of categorical features from a pandas DataFrame.

    Parameters:
    data (pandas.DataFrame): The pandas DataFrame.

    Returns:
    list: List of categorical feature names.
    """
    categorical_features = []
    for column in data.columns:
        if not is_numerical(data[column]):
            categorical_features.append(column)
    return categorical_features

def get_categorical_features_info(data: pd.DataFrame, feature_values_to_drop=None) -> pd.DataFrame:
    """
    Extract the names of categorical features, their unique value counts, and the actual unique values 
    from a pandas DataFrame.

    Parameters:
    data (pandas.DataFrame): The pandas DataFrame.

    Returns:
    pandas.DataFrame: DataFrame containing feature names, their unique value counts, and actual unique values.
    """
    feature_info = []
    for column in data.columns:
        if not is_numerical(data[column]):
            unique_values = data[column].dropna().unique()
            if feature_values_to_drop:
                if column in feature_values_to_drop:
                    values_to_drop = feature_values_to_drop[column]
                    print(f'values_to_drop: {values_to_drop}')
                    unique_values = [x for x in unique_values if x not in values_to_drop]
            unique_values_count = len(unique_values)
            feature_info.append([column, unique_values_count, unique_values])
    return pd.DataFrame(feature_info, columns=['Feature', 'Unique Values Count', 'Unique Values'])

def get_binary_features(df: pd.DataFrame):
    """
    Identify binary features (columns) in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - binary_features (list): List of column names representing binary features.

    Notes:
    - This method identifies binary features in a DataFrame by checking if each column has exactly two unique values, 0 and 1.
    - Only columns with exactly two unique values of 0 and 1 are considered binary features.
    """
    return [col for col in df.columns if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}]
