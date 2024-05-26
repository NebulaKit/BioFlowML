from src.feature_analysis import get_categorical_features_info, get_categorical_features, get_binary_features
from src.utils.monitoring import timeit, log_errors_and_warnings
from src.utils.IOHandler import IOHandler
from src.utils.logger_setup import get_main_logger
from src.utils import serialize_dict
from src.BioFlowMLClass import BioFlowMLClass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump, load
import pandas as pd
import numpy as np
import os


class RowDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer to drop rows from a pandas DataFrame based on specified feature values.

    Parameters:
        feature_values_to_drop : dict, optional
            A dictionary where keys are feature names and values are lists of values to drop for each feature.
            Default is None.

    Example:
        ```python
        from sklearn.pipeline import Pipeline
        from preprocessing.metadata_processing import RowDropper
        
        # Initialize the RowDropper transformer
        row_dropper = RowDropper(feature_values_to_drop={'B': ['apple', 'banana']})

        # Transform the DataFrame
        transformed_df = row_dropper.transform(df)
        
        # Or use in a pipeline
        # Create a preprocessing pipeline
        preprocess_pipeline = Pipeline([
            ('row_dropper', RowDropper(feature_values_to_drop={'B': ['apple', 'banana']})),
            # Additional preprocessing steps
        ])

        # Apply preprocessing pipeline
        df_preprocessed = preprocess_pipeline.fit_transform(df)
        ```
    """
    def __init__(self, feature_values_to_drop=None):
        self.feature_values_to_drop = feature_values_to_drop

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters:
            X (pandas.DataFrame): Input DataFrame.

        Returns:
            FeatureEncoder: The fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transforms the input DataFrame by dropping rows based on specified feature values.

        Parameters:
            X (pandas.DataFrame): Input DataFrame containing the data to be transformed.

        Returns:
            pandas.DataFrame: Transformed DataFrame with rows dropped based on specified feature values.
        """
        if self.feature_values_to_drop is None:
            return X

        if not isinstance(self.feature_values_to_drop, dict):
            raise ValueError("Feature values to drop must be provided as a dictionary!")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Unsupported input data type. Must be pandas DataFrame.")

        X_copy = X.copy()

        for feature, values in self.feature_values_to_drop.items():
            if feature in X_copy.columns:
                X_copy = X_copy[~X_copy[feature].isin(values)]

        return X_copy

class FeatureLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using label encoding.

    Parameters:
        features_to_encode (list): List of features to encode.

    Example:
        ```python
        from sklearn.pipeline import Pipeline
        from preprocessing.metadata_processing import FeatureLabelEncoder

        # Define features to encode
        features_to_encode = ['sex']

        # Create a preprocessing pipeline
        preprocess_pipeline = Pipeline([
            ('feature_encoder', FeatureLabelEncoder(features_to_encode)),
            # Additional preprocessing steps
        ])

        # Apply preprocessing pipeline
        X_preprocessed = preprocess_pipeline.fit_transform(X)
        ```
    """

    def __init__(self, features_to_encode=None):
        self.features_to_encode = features_to_encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_to_encode is None:
            return X
        
        encoder = LabelEncoder()
        X_copy = X.copy()
        
        for feature in self.features_to_encode:
            X_copy[feature] = encoder.fit_transform(X_copy[feature])

        return X_copy

class FeatureOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using one-hot encoding and drops the original categorical features.

    Parameters:
        features_to_encode (list): List of features to encode.
        prefix_with_feature_name (bool): If True, prefix the encoded columns with the original feature name.
                                         Default is False.

    Example:
        ```python
        from sklearn.pipeline import Pipeline
        from preprocessing.metadata_processing import FeatureOneHotEncoder

        # Define features to encode
        features_to_encode = ['categorical_feature1', 'categorical_feature2']

        # Create a preprocessing pipeline
        preprocess_pipeline = Pipeline([
            ('feature_encoder', FeatureOneHotEncoder(features_to_encode, prefix_with_feature_name=True)),
            # Additional preprocessing steps
        ])

        # Apply preprocessing pipeline
        X_preprocessed = preprocess_pipeline.fit_transform(X)
        ```
    """

    def __init__(self, features_to_encode=None, prefix_with_feature_name=False):
        self.features_to_encode = features_to_encode
        self.prefix_with_feature_name = prefix_with_feature_name

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters:
            X (pandas.DataFrame): Input DataFrame.

        Returns:
            FeatureOneHotEncoder: The fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by encoding categorical features using one-hot encoding
        and dropping the original categorical features.

        Parameters:
            X (pandas.DataFrame): Input DataFrame containing categorical features.

        Returns:
            pandas.DataFrame: Transformed DataFrame with one-hot encoded columns.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Unsupported input data type. Must be pandas DataFrame!")
        
        if self.features_to_encode is None:
            return X

        encoder = OneHotEncoder()
        for feature in self.features_to_encode:
            matrix = encoder.fit_transform(X[[feature]]).toarray()
            if self.prefix_with_feature_name:
                column_names = [f'{feature}_{cat}' for cat in encoder.categories_[0]]
            else:
                column_names = [f"{cat}" for cat in encoder.categories_[0]]
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            X_copy = X.copy()

            # Transpose, flip matrix dimensions
            for i, col_name in enumerate(column_names):
                X_copy[col_name] = matrix[:, i]

            # Drop the original categorical feature
            X_copy.drop(columns=[feature], inplace=True)

            X = X_copy  # Update X for the next feature

        return X

class NumberExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numerical data from categorical features by removing non-numeric characters.

    Parameters:
        features_to_extract (list): List of features from which numerical data will be extracted.

    Example:
        ```python
        from sklearn.pipeline import Pipeline
        from preprocessing.metadata_processing import FeatureOneHotEncoder

        # Define features to extract numerical data from
        features_to_extract = ['categorical_feature1', 'categorical_feature2']

        # Create a preprocessing pipeline
        preprocess_pipeline = Pipeline([
            ('number_extractor', NumberExtractor(features_to_extract)),
            # Additional preprocessing steps
        ])

        # Apply preprocessing pipeline
        X_preprocessed = preprocess_pipeline.fit_transform(X)
        ```
    """

    def __init__(self, features_to_extract=None):
        self.features_to_extract = features_to_extract

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by extracting numerical data from specified features.

        Parameters:
            X (pandas.DataFrame): Input DataFrame containing categorical features.

        Returns:
            pandas.DataFrame: DataFrame with numerical data extracted from specified features.
        """
        if self.features_to_extract is None:
            return X
        
        X_copy = X.copy()

        for feature in self.features_to_extract:
            X_copy[feature] = X[feature].str.replace(r'\D', '', regex=True).astype(float)

        return X_copy

class FeatureDropper(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_to_drop is None:
            return X
        else:
            return X.drop(self.features_to_drop, axis=1, errors="ignore")

class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_impute=None, label_feature=None):
        self.features_to_impute = features_to_impute
        self.label_feature = label_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        imputer = SimpleImputer(strategy="median")
        X_copy = X.copy()

        # If label_feature is provided, perform separate imputation for each label
        if self.label_feature is not None:
            for label in X_copy[self.label_feature].unique():
                label_rows = X_copy[X_copy[self.label_feature] == label].copy()  # Ensure we're working on a copy
                for feature in self.features_to_impute:
                  if feature != self.label_feature:
                    label_rows.loc[:, feature] = imputer.fit_transform(label_rows[[feature]]).squeeze()
                X_copy.loc[X_copy[self.label_feature] == label, self.features_to_impute] = label_rows[self.features_to_impute]
        
        # If label_feature is not provided, impute all rows
        else:
            for feature in self.features_to_impute:
                X_copy.loc[:, feature] = imputer.fit_transform(X_copy[[feature]]).squeeze()

        return X_copy

class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_impute=None, label_feature=None):
        self.features_to_impute = features_to_impute
        self.label_feature = label_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_to_impute is None:
                return X
        
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_copy = X.copy()

        # If label_feature is provided, perform separate imputation for each label
        if self.label_feature is not None:
            for label in X_copy[self.label_feature].unique():
                label_rows = X_copy[X_copy[self.label_feature] == label].copy()  # Ensure working on a copy
                for feature in self.features_to_impute:
                    if feature != self.label_feature:
                        # Custom imputer function to replace 'None' with NaN
                        label_rows.loc[:, feature] = label_rows[feature].apply(lambda x: np.nan if pd.isnull(x) else x)
                        # Apply imputation
                        label_rows.loc[:, feature] = cat_imputer.fit_transform(label_rows[[feature]]).squeeze()
                X_copy.loc[X_copy[self.label_feature] == label, self.features_to_impute] = label_rows[self.features_to_impute]

        # If label_feature is not provided, impute all rows
        else:
            for feature in self.features_to_impute:
                # Custom imputer function to replace 'None' with NaN
                X_copy.loc[:, feature] = X_copy[feature].apply(lambda x: np.nan if pd.isnull(x) else x)
                # Apply imputation
                X_copy.loc[:, feature] = cat_imputer.fit_transform(X_copy[[feature]]).squeeze()

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
                pt = PowerTransformer(method='yeo-johnson')
                X_transformed[col] = pt.fit_transform(X_transformed[[col]])
            else:
                raise ValueError("Invalid normalization method. Please choose 'log2', 'log10', 'log1p', or 'yeo-johnson'.")
        
        return X_transformed

def has_non_unique_columns(df: pd.DataFrame):
    """
    Checks if a DataFrame has any non-unique (duplicate) column names.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for non-unique columns.

    Returns:
        bool: Returns True if there are non-unique columns in the DataFrame, otherwise returns False.
    """
    return len(set(df.columns)) != len(df.columns)

def get_non_unique_columns(df: pd.DataFrame):
    """
    Retrieves a list of non-unique (duplicate) column names from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for non-unique columns.

    Returns:
        list: A list of column names that are duplicated in the DataFrame.
    """
    non_unique_columns = [col for col in df.columns if df.columns.tolist().count(col) > 1]
    return non_unique_columns

def sum_duplicated_columns(df: pd.DataFrame):
    """
    Sums the values of duplicated columns in a DataFrame and retains only one column with the original name,
    discarding the other duplicates.

    Parameters:
        df (pd.DataFrame): The DataFrame in which to sum and consolidate duplicated columns.

    Returns:
        pd.DataFrame: The modified DataFrame with summed duplicated columns and only one column retained for each original name.
    """
    for col in df.columns:
        if df.columns.tolist().count(col) > 1:  # Check if column is duplicated
            duplicated_columns = [col for col in df.columns if df.columns.tolist().count(col) > 1]
            new_col_name = col
            df[new_col_name] = df[duplicated_columns].sum(axis=1)  # Sum the rows of duplicated columns
            df.drop(columns=duplicated_columns[1:], inplace=True)  # Drop duplicate columns except the first occurrence
    return df

def prompt_to_select_feature_by_index(df):
    print('Select feature by index:')
    for i, col_name in enumerate(df.columns, start=1):
        print(f"{i} - {col_name}")
    
    while True:
        selected_index_str = input('Selected index (or exit with e): ')
        if selected_index_str == 'e':
            return None
        if not selected_index_str.isdigit():
            print("Please enter a valid number.\n")
            continue
        selected_index = int(selected_index_str)
        if 1 <= selected_index <= len(df.columns):
            return df.columns[selected_index - 1]
        else:
            print("Please enter a number within the valid range.\n")

def prompt_to_select_value_by_index(unique_values):
    print('Select value by index:')
    for i, value in enumerate(unique_values, start=1):
        print(f"{i} - {value}")
    
    while True:
        selected_index_str = input('Selected index (or exit with e): ')
        if selected_index_str == 'e':
            return None
        if not selected_index_str.isdigit():
            print("Please enter a valid number.\n")
            continue
        selected_index = int(selected_index_str)
        if 1 <= selected_index <= len(unique_values):
            return unique_values[selected_index - 1]
        else:
            print("Please enter a number within the valid range.\n")

def prompt_to_select_features_values_to_drop(df):
    features_values_to_drop = {}
    drop_feature_values = input('Drop rows by feature values (y/n)? ')
    if drop_feature_values.lower() != 'y':
        return features_values_to_drop
    
    while True:
        selected_feature = prompt_to_select_feature_by_index(df)
        if selected_feature is None:
            break
        unique_values = df[selected_feature].unique()
        selected_value = prompt_to_select_value_by_index(unique_values)
        if selected_value is None:
            break
        features_values_to_drop.setdefault(selected_feature, []).append(selected_value)
        
        continue_selecting = input('\nSelect more feature values (y/n)? ')
        if continue_selecting.lower() != 'y':
            break
    
    return features_values_to_drop
   
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
        # ('scaler', CustomScaler(scaler_type=scaler_type, numeric_features=numeric_features)),
        ('normalizer', CustomNormalizer(method=norm_method, numeric_features=numeric_features)),
        
    ])
    return numeric_pipeline

def create_preprocessing_pipeline_OLD():
    
    # pipeline = Pipeline([("rowdropper", RowDropper(feature_name='subject_group', value_to_drop='OTHER')),
    #                 ("featureencoder", FeatureOneHotEncoder(features_to_encode=['subject_group','sex'])),
    #                 ("binaryencoder", FeatureBinaryEncoder(features_to_encode=['have_animals_or_pets','high_blood_pressure_ever', 'darbs_nakti_grupa'])),
    #                 ("customBinaryEncoder", CustomBinaryEncoder(features_to_encode=['ethnicity','energijas_atbilst_gr_0','izglitiba_grupas','vit_d_sufficiency'], positive_value=['Latvian','below_or_ok','higher_ed','Sufficient'], negative_value=['OTHER','too_much_energy','below_higher_ed','Insufficient'], new_column_names=['ethnicity_latvian','energijas_atbilst_gr_0','izglitiba_grupas','vit_d_sufficiency'])),
    #                 #("endTrimmer", EndTrimmer(features_to_trim=['fiber_sufficiency'])),
    #                 ("numberExtractor", NumberExtractor(['fiber_sufficiency'])),
    #                 ("intConverter", IntConverter(features_to_convert=['fiber_sufficiency','sample_id'])),
    #                 ("featuredropper", FeatureDropper(features_to_drop=['seq_batch','denoised_reads','ethnicity','subject_group','sex']))])
    
    pipeline = Pipeline([
        ("rowdropper", RowDropper({'subject_group':['OTHER']})),
        ("featureencoder1", FeatureOneHotEncoder(['subject_group','sex'])),
        ("featureencoder2", FeatureOneHotEncoder(['have_animals_or_pets','high_blood_pressure_ever', 'darbs_nakti_grupa','ethnicity','energijas_atbilst_gr_0','izglitiba_grupas','vit_d_sufficiency'], prefix_with_feature_name=True)),
        ("numberExtractor", NumberExtractor(['fiber_sufficiency','seq_batch']))
    ])
    
    return pipeline

@log_errors_and_warnings
def start_pipeline_wizard(obj: BioFlowMLClass):
    
    logger = get_main_logger()
    
    # 1. Drop any rows by value if necessary
    features_values_to_drop = prompt_to_select_features_values_to_drop(obj.df)
    logger.info(f'Dropping {obj.out_dir_name} feature values during preprocessing: {features_values_to_drop}\n')
    
    # 2. Encode categorical values
    categorical_features_info = get_categorical_features_info(obj.df, features_values_to_drop)
    
    # Select encoding methods for categorical features
    features_to_label_encode = []
    features_to_one_hot_encode_1 = []
    features_to_one_hot_encode_2 = []
    features_to_extract_numbers_from = []
    features_to_drop = []
    
    if not categorical_features_info.empty:
    
        # Set pandas display options to ensure full display of values
        pd.set_option('display.max_colwidth', None)
        # Log categorical feature info
        logger.info(f'Categorical features detected:\n{categorical_features_info}\n\n')
        
        # Prompt user to select encoding method
        for _, row in categorical_features_info.iterrows():
            f = row['Feature']
            option = -1
            options = ['1','2','3','4']
            while option not in options:
                prompt_text = (
                    f"How to transform the categorical values of '{f}' feature? "
                    f"Values: {row['Unique Values']}\n"
                    "1 - Label encode (one feature)\n"
                    "2 - One-hot encode (prefix with feature name)\n"
                    "3 - One-hot encode (use value only; Not recomminded if values are not unique across different features!)\n"
                    "4 - Drop feature\n"
                )
                if obj.df[f].str.replace(r'\D', '', regex=True).str.isnumeric().any():
                    prompt_text += f'5 - Extract numerical data\n'
                    options.append('5')
    
                prompt_text += 'Selected option: '
                option = input(prompt_text)
        
            match option:
                case '1': 
                    features_to_label_encode.append(f)
                    # Save categorical values for later reference in the same order they will be encoded
                    obj.set_encoded_features(key=f, value=sorted(row['Unique Values']))
                case '2': features_to_one_hot_encode_1.append(f)
                case '3': features_to_one_hot_encode_2.append(f)
                case '4': features_to_drop.append(f)
                case '5': features_to_extract_numbers_from.append(f)
            
    categorical_features = get_categorical_features(obj.df)
    numerical_features = [x for x in obj.df.columns if x not in categorical_features]
    
    # TODO: should impute by category (e.g. Control, Liver_disease)
    pipeline = Pipeline([
        # Impute
        ("categoricalImputer", CategoricalImputer(categorical_features, obj.label_feature)),
        ("numericalImputer", NumericalImputer(numerical_features, obj.label_feature)),
        # Drop rows if needed
        ("rowDropper", RowDropper(features_values_to_drop)),
        # Encode categorical features
        ('featureLabelEncoder', FeatureLabelEncoder(features_to_label_encode)),
        ("featureOneHotEncoder1", FeatureOneHotEncoder(features_to_one_hot_encode_1, prefix_with_feature_name=True)),
        ("featureOneHotEncoder2", FeatureOneHotEncoder(features_to_one_hot_encode_2)),
        ("numberExtractor", NumberExtractor(features_to_extract_numbers_from)),
        # Drop features if needed
        ('featureDropper', FeatureDropper(features_to_drop))
    ])
    
    # Save the pipeline
    path = IOHandler.get_absolute_path('src/preprocessing/pipelines', create_dir=True)
    file_path = os.path.join(path, f'{obj.out_dir_name}_preprocessing_pipeline.joblib')
    dump(pipeline, file_path)
    
    # Save label encoded feature names and encoding
    encoded_features = serialize_dict(obj.get_encoded_features())
    file_path = os.path.join(path, f'{obj.out_dir_name}_encoded_features.txt')
    with open(file_path, 'w') as file:
        file.write(encoded_features)
    
    obj.log_obj()
    return pipeline

def get_preprocessing_pipeline(obj: BioFlowMLClass, sort_by=None):
    
    if sort_by:
        if sort_by in obj.df.columns:
            obj.df = obj.df.sort_values(by=sort_by)
    
    pipeline=None
    
    # Check if pipeline saved
    directory_path = "src/preprocessing/pipelines"
    pipeline_file_name = f'{obj.out_dir_name}_preprocessing_pipeline.joblib'
    
    if os.path.isdir(directory_path):
        if pipeline_file_name in os.listdir(directory_path):
            # TODO: prompt user before load
            # TODO: retreive encoded features for loaded pipelines somehow
            pipeline = load(f'{directory_path}/{pipeline_file_name}')

    if not pipeline:
        pipeline = start_pipeline_wizard(obj)
    
    return pipeline
    