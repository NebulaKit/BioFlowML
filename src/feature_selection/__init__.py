from src.BioFlowMLClass import BioFlowMLClass
from src.preprocessing.microbiome_preprocessing import trim_taxa_names, trim_dup_taxa_names
from src.feature_analysis import get_binary_features
from src.utils.monitoring import log_errors_and_warnings
from src.utils.logger_setup import get_main_logger
from sklearn.feature_selection import VarianceThreshold
from src.utils import serialize_list
import pandas as pd
import numpy as np
import copy


@log_errors_and_warnings
def remove_low_variance_features(obj: BioFlowMLClass, threshold=0.05):
    """
    Remove low variance features from a BioFlowMLClass object Pandas DataFrame using scikit-learn's VarianceThreshold.

    Parameters:
        obj (BioFlowMLClass): BioFlowMLClass object containing features.
        threshold (float): Variance threshold below which features will be removed.
                           Default is 0.05.

    Returns:
        obj_new: BioFlowMLClass object with low variance features removed.
    """
    
    # Create a deep copy of the original object
    obj_copy = copy.deepcopy(obj)
    
    # Initialize VarianceThreshold with the specified threshold
    selector = VarianceThreshold(threshold=threshold)

    # Fit the selector to the data
    selector.fit(obj_copy.df)

    # Get boolean mask of features with high variance
    mask = selector.get_support()
    
    # Get names of removed features
    removed_features = obj_copy.df.columns[~mask]
    if len(removed_features) > 0:
        logger = get_main_logger()
        logger.info(f'Low varience features removed: {serialize_list(list(removed_features))}')

    # Filter DataFrame based on the mask
    obj_copy.df = obj_copy.df.loc[:, mask]

    return obj_copy

@log_errors_and_warnings
def aggregate_taxa_by_level(obj: BioFlowMLClass, level, drop_unclassified=False, trim_taxa=False):
    """
    Aggregates taxa in the dataframe based on the specified taxonomic level.

    Parameters:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing a DataFrame with taxa names.
        level (str): The taxonomic level to aggregate by (e.g., 'd', 'p', 'c', 'o', 'f', 'g', 's').
        drop_unclassified (bool): Whether to drop unclassified taxa. Defaults to True.

    Returns:
        BioFlowMLClass: The modified BioFlowMLClass object with taxa aggregated by the specified level in the pandas DataFrame.
    
    Raises:
        ValueError: If the provided level is not supported.
    
    Example:
        ```python
        # Assuming obj is an instance of BioFlowMLClass with a DataFrame containing taxa names
        obj = aggregate_taxa_by_level(obj, 'f')
        ```
    """
    
    # Define valid taxonomic level indicators
    level_indicators = ['d', 'p', 'c', 'o', 'f', 'g', 's']
    level_names = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # Check if the provided level is valid
    if level not in level_indicators:
        raise ValueError(f'Provided level ({level}) not supported! Possible options: {level_indicators}')

    # Create the indication string for the specified level (e.g. 'g__')
    # previous and next level
    index = level_indicators.index(level)
    prev_index = index - 1
    next_index = index + 1

    indication_str = level_indicators[index] + '__'
    prev_indication_str = level_indicators[prev_index] + '__' if prev_index >= 0 else None
    next_indication_str = level_indicators[next_index] + '__' if next_index < len(level_indicators) else None
    
    if not any(indication_str in col for col in obj.df.columns):
        raise ValueError(f"The provided taxonomic profile has not been classified to the selected level ({level_names[index]})!")
    
    # Remove duplicating information from taxa names e.g.
    # d__Archaea;p__Thermoplasmatota;c__Thermoplasmata;o__Methanomassiliicoccales;f__Methanomethylophilaceae;g__uncultured__f__Methanomethylophilaceae -->
    # d__Archaea;p__Thermoplasmatota;c__Thermoplasmata;o__Methanomassiliicoccales;f__Methanomethylophilaceae;g__uncultured
    trim_dup_taxa_names(obj, level)
    
    unclassified_indications = ['norank', 'unclassified']
    aggregate_taxa = {'unclassified': []}

    for c in obj.df.columns:
        if indication_str not in c and '__' in c:
            aggregate_taxa['unclassified'].append(c)
        elif indication_str in c:
            level_name = c.split(indication_str)[-1].split(next_indication_str)[0] if next_indication_str else c.split(indication_str)[-1]
            
            if any(substring in level_name for substring in unclassified_indications):
                aggregate_taxa['unclassified'].append(c)
            elif 'uncultured' in level_name:
                if prev_indication_str:
                    prev_level_name = c.split(indication_str)[0].split(prev_indication_str)[-1].replace('_', ' ').replace(';', '')
                    if 'uncultured' in prev_level_name:
                        aggregate_taxa['unclassified'].append(c)
                    else:
                        aggregate_name = c.split(next_indication_str)[0].rstrip(';') if next_indication_str else c
                        aggregate_name = aggregate_name.rstrip(';__')
                        aggregate_taxa.setdefault(aggregate_name, []).append(c)
                else:
                    aggregate_taxa['unclassified'].append(c)
            else:
                aggregate_name = c.split(next_indication_str)[0].rstrip(';') if next_indication_str else c
                aggregate_name = aggregate_name.rstrip(';__')
                aggregate_taxa.setdefault(aggregate_name, []).append(c)    
    
    
    # Initialize lists to store columns to sum and drop
    sum_columns = []
    drop_columns = []

    # Iterate over the aggregated taxa dictionary
    for key, value in aggregate_taxa.items():
        if value:
            # Select the columns to aggregate
            selected_columns = obj.df[value]
            
            # Sum the selected columns across rows
            sum_column = selected_columns.sum(axis=1)
            sum_column.name = key
            sum_columns.append(sum_column)
            
            # Extend the list of columns to drop
            drop_columns.extend(list(selected_columns))

    # Drop the original columns that were aggregated
    obj.df = obj.df.drop(columns=drop_columns)
    
    # Concatenate the original dataframe with the new summed columns
    obj.df = pd.concat([obj.df] + sum_columns, axis=1)
    
    if drop_unclassified:
        obj.df = obj.df.drop(columns=['unclassified'])
    
    # Shorten the taxa names to last significant level
    # d__Archaea;p__Thermoplasmatota;c__Thermoplasmata;o__Methanomassiliicoccales;f__Methanomethylophilaceae;g__uncultured -->
    # Methanomethylophilaceae uncultured
    if trim_taxa:
        obj = trim_taxa_names(obj)
        
    obj.out_dir_name = f'{obj.out_dir_name}_{level_names[index]}'
    obj.log_obj()
    
    return obj

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
