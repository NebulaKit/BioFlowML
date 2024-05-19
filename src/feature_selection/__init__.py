from src.feature_analysis import get_binary_features
from src.utils import serialize_list
from src.utils.logger_setup import get_main_logger
import pandas as pd
import numpy as np


def apply_zeros_filter(df: pd.DataFrame, exclude_features: list, filter_records = False, threshold = 0.2):
    """
    Apply a zeros filter to a feature matrix, excluding specified columns and binary data columns from filtering.

    Parameters:
    - df (pd.DataFrame): Input feature matrix.
    - exclude_features (list): List of columns to exclude from filtering.
    - threshold (float): Threshold for the proportion of zeros in a column or row. Default is 0.2 (i.e., 20%).

    Returns:
    - filtered_matrix (pd.DataFrame): Filtered feature matrix with excluded columns added back after filtering on the target features has been performed.

    Notes:
    - This method applies a zeros filter to remove columns and rows with a high proportion of zero values from the input feature matrix.
    - Binary columns (containing only 0s and 1s) are automatically excluded from filtering.
    - Any columns specified in the `exclude_features` list are also excluded from filtering.
    - The threshold parameter determines the proportion of zeros (relative to the total number of columns or rows) above which columns or rows are filtered out.
    """
    logger = get_main_logger()
    # Initialize exclude_features if it's None
    exclude_features = exclude_features or []
    # Add binary columns to exclude_features if they are not already present
    exclude_features.extend(col for col in get_binary_features(df) if col not in exclude_features)
    logger.debug(f'Features excluded from zeros (>{threshold*100}) filtering: {serialize_list(exclude_features)}')

    # Calculate the number of zeros in each column and row
    num_zeros_col = df.loc[:, ~df.columns.isin(exclude_features)].eq(0).sum(axis=0)
    num_zeros_row = df.loc[:, ~df.columns.isin(exclude_features)].eq(0).sum(axis=1)

    # Calculate the total number of columns and rows included in filtering
    total_rows = len(df)
    total_cols = (len(df.columns)-len(exclude_features))

    # Calculate percentage threshold values
    threshold_for_rows = threshold * total_cols
    threshold_for_cols = threshold * total_rows

    # Filter columns based on the threshold
    filtered_features = num_zeros_col[num_zeros_col >= threshold_for_cols].index
    logger.debug(f'Dropped {len(filtered_features)}/{len(df.columns)} features: {serialize_list(filtered_features)}')
    logger.debug(f'{len(df.columns)-len(filtered_features)} features left')
    df = df.drop(columns=filtered_features)
    
    # Filter rows based on the threshold
    if filter_records:
        filtered_records = num_zeros_row[num_zeros_row >= threshold_for_rows].index
        df = df.drop(index=filtered_records)
        logger.debug(f'Dropped {len(filtered_records)}/{total_rows} features: {serialize_list(filtered_records)}')
        logger.debug(f'{total_rows-len(filtered_records)} records left')

    return df

def apply_minimum_abundance_filter(df: pd.DataFrame, exclude_features: list = None, min_pct = 0.1):
    """
    Apply a minimum abundance filter to remove low-abundance features from a DataFrame containing compositional data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing abundance data.
    - exclude_features (list, optional): List of feature names to exclude from the filtering process. Default is None.
    - min_pct (float, optional): Minimum percentage abundance threshold below which features are filtered out. 
                                 Default is 0.1 (i.e., 0.1%).

    Returns:
    - filtered_df (pd.DataFrame): Filtered DataFrame with low-abundance features removed.

    Notes:
    - This method applies a minimum abundance filter to remove low-abundance features from the input DataFrame.
    - The abundance of each feature is calculated by summing its values across all samples.
    - Features with total abundance below the specified minimum percentage threshold are filtered out.
    - Binary features (containing only 0s and 1s) are automatically excluded from filtering.
    - Any features specified in the `exclude_features` list are also excluded from filtering.
    """
    logger = get_main_logger()
    exclude_features = exclude_features or []
    exclude_features.extend(col for col in get_binary_features(df) if col not in exclude_features)
    logger.debug(f'Features excluded from minimum abundance ({min_pct}%) filtering: {serialize_list(exclude_features)}')
    
    df_copy = df.copy()
    total_counts = df_copy.drop(columns=exclude_features).sum()
    threshold = (min_pct/100) * total_counts.sum()
    filtered_features = total_counts[total_counts >= threshold].index
    removed_features = total_counts[total_counts < threshold].index
    logger.debug(f'Dropped {len(removed_features)}/{len(df.columns)} features: {serialize_list(removed_features)}')
    
    filtered_df = df[filtered_features].copy()
    filtered_df[exclude_features] = df[exclude_features]
    logger.debug(f'{len(filtered_df.columns)} features left')
    
    return filtered_df

def calculate_shannon_diversity_index(abundances):
    # the Shannon-Wiener index or Shannon entropy
    total_count = np.sum(abundances)
    proportions = abundances / total_count
    nonzero_proportions = proportions[proportions > 0]  # Exclude zero proportions
    return round(-np.sum(nonzero_proportions * np.log(nonzero_proportions)), 2)

def add_shannon_alpha_diversity_feature(df: pd.DataFrame, exclude_features: list = None):
    # Initialize the list to store Shannon diversity values
    shannon_diversity_values = []

    # Iterate over rows of the DataFrame
    for _, abundances in df.iterrows():
        # Exclude values of features specified in exclude_features
        abundances_excluded = abundances.drop(labels=exclude_features)
        
        # Calculate Shannon diversity index
        shannon_index = calculate_shannon_diversity_index(abundances_excluded)
        
        # Append Shannon diversity index to the list
        shannon_diversity_values.append(shannon_index)

    # Add Shannon diversity as a new feature column
    df['Shannon_diversity'] = shannon_diversity_values
    
    # Add Shannon_diversity to the exclude_features list
    if exclude_features is None:
        exclude_features = []
    exclude_features.append('Shannon_diversity')
    logger = get_main_logger()
    logger.debug(f'Shannon_diversity feature added, total features {len(df.columns)}')
    
    return df

def add_simple_alpha_diversity_feature(df: pd.DataFrame, exclude_features: list = None):
    # Initialize a list to store counts for each row
    counts_per_row = []
    
    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        # Exclude specified features if provided
        if exclude_features is not None:
            row = row.drop(labels=exclude_features)
        
        # Count instances when feature value is greater than 0
        count_positive_instances = (row > 0).sum()
        
        # Append the count to the list
        counts_per_row.append(count_positive_instances)
    
    df['Alpha_diversity'] = counts_per_row
    # Add Alpha_diversity to the exclude_features list
    if exclude_features is None:
        exclude_features = []
    exclude_features.append('Alpha_diversity')
    logger = get_main_logger()
    logger.debug(f'Alpha_diversity feature added, total features {len(df.columns)}')

    return df
