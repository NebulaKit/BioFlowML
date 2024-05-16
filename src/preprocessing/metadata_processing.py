from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import numpy as np
import os
from src.BioFlowMLClass import BioFlowMLClass
import src.exploratory_data_analysis as eda
from src.utils.logger_setup import get_main_logger
from src.utils.monitoring import timeit, log_errors_and_warnings
from src.preprocessing import NumericalImputer, CategoricalImputer
from src.utils.IO import get_project_root_dir, get_absolute_path


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


def create_preprocessing_pipeline():
    
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

def select_feature_by_index(df):
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

def select_value_by_index(unique_values):
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

def get_features_values_to_drop(df):
    features_values_to_drop = {}
    drop_feature_values = input('Drop rows by feature values (y/n)? ')
    if drop_feature_values.lower() != 'y':
        return features_values_to_drop
    
    while True:
        selected_feature = select_feature_by_index(df)
        if selected_feature is None:
            break
        unique_values = df[selected_feature].unique()
        selected_value = select_value_by_index(unique_values)
        if selected_value is None:
            break
        features_values_to_drop.setdefault(selected_feature, []).append(selected_value)
        
        continue_selecting = input('\nSelect more feature values (y/n)? ')
        if continue_selecting.lower() != 'y':
            break
    
    return features_values_to_drop

@log_errors_and_warnings
def start_pipeline_wizard(obj: BioFlowMLClass):
    
    # 1. Drop any rows by value if necessary
    features_values_to_drop = get_features_values_to_drop(obj.df)
    print(f'features_values_to_drop: {features_values_to_drop}\n')
    
    
    # 2. Encode categorical values
    categorical_features_info = eda.categorical_features_info(obj.df, features_values_to_drop)
    
     # Set pandas display options to ensure full display of values
    pd.set_option('display.max_colwidth', None)
    logger = get_main_logger()
    logger.info(f'Categorical features detected:\n{categorical_features_info}\n\n')
    
    features_to_label_encode = []
    features_to_one_hot_encode_1 = []
    features_to_one_hot_encode_2 = []
    features_to_extract_numbers_from = []
    features_to_drop = []
    
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
            
    
    categorical_features = categorical_features_info['Feature'].tolist()
    numerical_features = [x for x in obj.df.columns if x not in categorical_features]
    
    pipeline = Pipeline([
        # Impute
        ("categoricalImputer", CategoricalImputer(categorical_features)),
        ("numericalImputer", NumericalImputer(numerical_features)),
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
    path = get_absolute_path('preprocessing/pipelines',create_dir=True)
    file_path = os.path.join(path, f'{obj.out_dir_name}_preprocessing_pipeline.joblib')
    dump(pipeline, file_path)
    
    obj.log_obj()
    return pipeline