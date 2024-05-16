from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import src.utils.logger_setup as log
import src.utils as utils
import numpy as np
import pandas as pd



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
    logger = log.get_logger('main_log')
    # Initialize exclude_features if it's None
    exclude_features = exclude_features or []
    # Add binary columns to exclude_features if they are not already present
    exclude_features.extend(col for col in get_binary_features(df) if col not in exclude_features)
    logger.debug(f'Features excluded from zeros (>{threshold*100}) filtering: {utils.serialize_list(exclude_features)}')

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
    logger.debug(f'Dropped {len(filtered_features)}/{len(df.columns)} features: {utils.serialize_list(filtered_features)}')
    logger.debug(f'{len(df.columns)-len(filtered_features)} features left')
    df = df.drop(columns=filtered_features)
    
    # Filter rows based on the threshold
    if filter_records:
        filtered_records = num_zeros_row[num_zeros_row >= threshold_for_rows].index
        df = df.drop(index=filtered_records)
        logger.debug(f'Dropped {len(filtered_records)}/{total_rows} features: {utils.serialize_list(filtered_records)}')
        logger.debug(f'{total_rows-len(filtered_records)} records left')

    return df

class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_impute=None):
        self.features_to_impute = features_to_impute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
      if self.features_to_impute is None:
            return X
      
      imputer = SimpleImputer(strategy="median")
      X_copy = X.copy()

      for feature in self.features_to_impute:
          X_copy[feature] = imputer.fit_transform(X_copy[[feature]])[:, 0]

      return X_copy

class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_impute=None):
        self.features_to_impute = features_to_impute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
      if self.features_to_impute is None:
            return X
      
      cat_imputer = SimpleImputer(strategy='most_frequent')
      X_copy = X.copy()

      for feature in self.features_to_impute:
          # Custom imputer function to replace 'None' with NaN
          X_copy[feature] = X_copy[feature].apply(lambda x: np.nan if type(x) == type(None) else x)
          # Apply imputation
          X_copy[feature] = cat_imputer.fit_transform(X_copy[[feature]]).squeeze()

      return X_copy


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom Scaler transformer class to scale numerical features with options for different scaling methods,
    excluding specific columns.

    Parameters:
    - scaler (str, optional): The scaling method to use. Options: 'standard', 'robust'. Defaults to 'standard'.
    - exclude_columns (list, optional): List of column names to exclude from scaling. Defaults to None.
    """

    def __init__(self, scaler_type='standard', numeric_features=None):
        self.scaler_type = scaler_type
        self.numeric_features = numeric_features
        self.scaler_instance = None

    def fit(self, X, y=None):
        # Initialize the appropriate scaler based on the specified method
        if self.scaler_type == 'standard':
            self.scaler_instance = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler_instance = RobustScaler()
        else:
            raise ValueError("Invalid scaler method. Please choose 'standard' or 'robust'.")

        # Fit the scaler to the numerical features
        self.scaler_instance.fit(X[self.numeric_features])

        return self

    def transform(self, X):
        # Copy the input DataFrame to avoid modifying the original data
        X_scaled = X.copy()

        # Apply scaling to the numerical features
        X_scaled[self.numeric_features] = self.scaler_instance.transform(X_scaled[self.numeric_features])

        return X_scaled


class CustomNormalizer(BaseEstimator, TransformerMixin):
    """
    Custom Normalizer transformer class to normalize numerical features with options for different normalization methods,
    excluding specific columns.

    Parameters:
    - method (str, optional): The normalization method to use. Options: 'log2', 'log1p', 'yeo-johnson'. Defaults to 'log2'.
    - exclude_columns (list, optional): List of column names to exclude from normalization. Defaults to None.
    """

    def __init__(self, method='log2', numeric_features=None):
        self.method = method
        self.numeric_features = numeric_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Copy the input DataFrame to avoid modifying the original data
        X_transformed = X.copy()

        # Apply normalization to the numerical features
        for col in self.numeric_features:
            if self.method == 'log2':
                X_transformed[col] = np.log2(1 + X_transformed[col])
            elif self.method == 'log10':
                X_transformed[col] = np.log10(1 + X_transformed[col])
            elif self.method == 'log1p':
                X_transformed[col] = np.log1p(X_transformed[col])
            elif self.method == 'yeo-johnson':
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                X_transformed[col] = pt.fit_transform(X_transformed[col].values.reshape(-1, 1)).flatten()
            else:
                raise ValueError("Invalid normalization method. Please choose 'log2', 'log10', 'log1p', or 'yeo-johnson'.")
        
        return X_transformed
    

def get_numerical_feature_pipeline(df: pd.DataFrame, norm_method='yeo-johnson', scaler_type='robust', exclude_features: list = None):
    """
    Create a preprocessing pipeline for numerical features in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the dataset.
    - norm_method (str, optional): Normalization method to be applied. 
                                   Options: 'log2', 'log1p', 'yeo-johnson'. Defaults to 'yeo-johnson'.
    - scaler_type (str, optional): Type of scaler to be used. 
                                   Options: 'standard', 'robust'. Defaults to 'robust'.
    - exclude_features (list, optional): List of feature names to exclude from preprocessing. Default is None.

    Returns:
    - pipeline (Pipeline): Preprocessing pipeline for numerical features.

    Notes:
    - This method creates a preprocessing pipeline for numerical features in the input DataFrame.
    - Numeric features are automatically determined from the DataFrame by selecting columns with numeric data types.
    - Binary features (containing only 0s and 1s) are automatically excluded from preprocessing.
    - Any features specified in the `exclude_features` list are also excluded from preprocessing.
    - The pipeline includes normalization and scaling steps for numeric features.
    - Normalization is performed using the specified method ('log2', 'log1p', 'yeo-johnson').
    - Scaling is performed using the specified scaler type ('standard', 'robust').
    - The pipeline preserves the remaining non-numeric features ('remainder'='passthrough').

    Example:
    ```python
    # Create preprocessing pipeline
    pipeline = get_numerical_feature_pipeline(df, norm_method='log2', scaler_type='standard', exclude_features=['feature1', 'feature2'])
    
    # Fit and transform the pipeline on training data
    X_train_processed = pipeline.fit_transform(X_train)
    
    # Transform the pipeline on test data
    X_test_processed = pipeline.transform(X_test)
    ```
    """
    # Automatically determine numeric features
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    # Exclude binary features from numeric features
    numeric_features = [col for col in numeric_features if col not in get_binary_features(df)]
    # Exclude specified features if provided
    if exclude_features is not None:
        numeric_features = [col for col in numeric_features if col not in exclude_features]

    # Create pipeline for numeric features
    numeric_pipeline = Pipeline([
        ('normalizer', CustomNormalizer(method=norm_method, numeric_features=numeric_features)),
        ('scaler', CustomScaler(scaler_type=scaler_type, numeric_features=numeric_features))
    ])
    return numeric_pipeline

