import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_transformations
from src.feature_analysis.correlations import check_correlations
from src.feature_analysis.comparisons import compare_distributions
from src.preprocessing import encode_and_impute_features, preprocess_numerical_features
from src.model_training.binary_classification import classify_binary
from src.model_training.multiclass_classification import train_multiclass_classifiers
from src.utils.IOHandler import IOHandler
import pandas as pd


def main():
    
    # Promt user to reset logfile if exits
    IOHandler.reset_logfile()
    
    # Create processed data directory if doesn't exist
    IOHandler.get_absolute_path('../data/processed', create_dir=True)
    
    # Read metadata feature matrix
    df = pd.read_csv('data/synthetic/metadata.csv')
    
    # Create and initialize BioFlowML class instance
    obj = BioFlowMLClass(df,                 
                    out_dir_name = 'metadata_original',
                    label_feature = 'subject_group',
                    exclude_features = ['sample_id'],
                    control_label = 'Control',
                    lang = 'en')
    
    # Preprocess non-numerical features and missing values
    obj = encode_and_impute_features(obj, sort_by='sample_id')
    
    # Check data transformation distributions for all metadata features
    check_transformations(obj)
    
    # Normalize and scale numeric features
    obj = preprocess_numerical_features(obj, exclude_features=obj.exclude_features + [obj.label_feature])
    # Save normalized feature matrix as csv if needed
    obj.df.to_csv(f'data/processed/{obj.out_dir_name}_normalized.csv', index=False)
    
    # Correlation analysis
    check_correlations(obj)
    
    # Feature distribution comparison
    compare_distributions(obj)
    
    # Binary classifier training and evaluation
    classify_binary(obj)
    
    # Multiclass classifier
    train_multiclass_classifiers(obj)
    
    
if __name__ == "__main__":
    main()