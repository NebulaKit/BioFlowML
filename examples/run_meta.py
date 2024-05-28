import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_transformations
from src.feature_analysis.correlations import check_correlations
from src.preprocessing import get_preprocessing_pipeline, get_numerical_feature_pipeline
from src.model_training.binary_classification import train_binary_classifiers
from src.model_training.multiclass_classification import train_multiclass_classifiers
from src.utils.IOHandler import IOHandler
import pandas as pd


def main():
    
    # Promt user to reset logfile if exit
    IOHandler.reset_logfile()
    
    # Read metadata feature matrix
    df = pd.read_csv('data/synthetic/metadata.csv')
    
    # Create and initialize BioFlowML class instance
    obj = BioFlowMLClass(df,                 
                    out_dir_name = 'metadata',
                    label_feature = 'subject_group',
                    exclude_features = ['sample_id'],
                    control_label = 'Control',
                    lang = 'lv')
    
    # Preprocess non-numerical features and missing values
    pipeline = get_preprocessing_pipeline(obj, sort_by='sample_id')
    obj.df = pipeline.fit_transform(obj.df)
    
    # Check data transformation distributions for all metadata features
    check_transformations(obj)
    
    # Normalize and scale numeric features
    normalization_pipeline = get_numerical_feature_pipeline(obj.df, exclude_features=obj.exclude_features + [obj.label_feature])
    obj.df = normalization_pipeline.fit_transform(obj.df)
    
    # Save normalized feature matrix as csv
    obj.df.to_csv(f'data/processed/{obj.out_dir_name}_normalized.csv', index=False)
    
    # Correlation analysis
    check_correlations(obj)
    
    # Binary classifier training and evaluation
    train_binary_classifiers(obj)
    
    # Multiclass classifier
    train_multiclass_classifiers(obj)
    
    
if __name__ == "__main__":
    main()