from sklearn.pipeline import Pipeline
import src.utils.logger_setup as log
import src.utils as utils
import src.preprocessing as pp
import numpy as np
import pandas as pd




class OTUDataFrameTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.transpose().assign(sample_id=lambda x: x.index.astype(str).map(int)).sort_index(axis=0)
        return X
    
    
def create_otu_table_transpose_pipeline():
    
    pipeline = Pipeline(steps=[
        ('dataframe_transformer', OTUDataFrameTransformer())
    ])
    
    return pipeline


def merge_mb_to_targets(df_mb: pd.DataFrame, df_meta: pd.DataFrame, add_features: list = None, _on = 'sample_id'):
    
    df_meta_filtered = df_meta
    if add_features is not None:
        features = add_features.copy()
        features.insert(0, _on) # Reference type, gets edited in main.py as well
        df_meta_filtered = df_meta[features]

    df_merged_mb = pd.merge(df_meta_filtered, df_mb, on=_on)
    return df_merged_mb


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
    logger = log.get_logger('main_log')
    exclude_features = exclude_features or []
    exclude_features.extend(col for col in pp.get_binary_features(df) if col not in exclude_features)
    logger.debug(f'Features excluded from minimum abundance ({min_pct}%) filtering: {utils.serialize_list(exclude_features)}')
    
    df_copy = df.copy()
    total_counts = df_copy.drop(columns=exclude_features).sum()
    threshold = (min_pct/100) * total_counts.sum()
    filtered_features = total_counts[total_counts >= threshold].index
    removed_features = total_counts[total_counts < threshold].index
    logger.debug(f'Dropped {len(removed_features)}/{len(df.columns)} features: {utils.serialize_list(removed_features)}')
    
    filtered_df = df[filtered_features].copy()
    filtered_df[exclude_features] = df[exclude_features]
    logger.debug(f'{len(filtered_df.columns)} features left')
    
    return filtered_df


#  the Shannon-Wiener index or Shannon entropy
def shannon_diversity_index(abundances):
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
        shannon_index = shannon_diversity_index(abundances_excluded)
        
        # Append Shannon diversity index to the list
        shannon_diversity_values.append(shannon_index)

    # Add Shannon diversity as a new feature column
    df['Shannon_diversity'] = shannon_diversity_values
    
    # Add Shannon_diversity to the exclude_features list
    if exclude_features is None:
        exclude_features = []
    exclude_features.append('Shannon_diversity')
    logger = log.get_logger('main_log')
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
    logger = log.get_logger('main_log')
    logger.debug(f'Alpha_diversity feature added, total features {len(df.columns)}')

    return df
