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
from src.preprocessing.microbiome_preprocessing import  merge_with_metadata
from src.preprocessing import encode_and_impute_features, preprocess_numerical_features
from src.feature_selection import aggregate_taxa_by_level, remove_low_variance_features
from src.model_training.binary_classification import classify_binary
from src.model_training.multiclass_classification import train_multiclass_classifiers
from src.utils.IOHandler import IOHandler
import pandas as pd


def main():
    
    # Promt user to reset logfile if exits
    IOHandler.reset_logfile()
    
    # Create processed data directory if doesn't exist
    IOHandler.get_absolute_path('../data/processed', create_dir=True)
    
    # Read microbiome feature matrix
    df_mb = pd.read_csv('data/synthetic/microbiome.csv')
    
    # Create and initialize BioFlowML class instance for the microbiome data
    id_column = 'sample_id'
    obj_mb = BioFlowMLClass(df_mb,
                            out_dir_name = 'microbiome',
                            exclude_features = [id_column],
                            lang = 'en')
    
    # Read metadata feature matrix
    df_meta = pd.read_csv('data/synthetic/metadata.csv')
    
    # Create and initialize BioFlowML class instance for the metadata (to map label feature to microbiome data)
    label_feature = 'subject_group'
    control_label = 'Control'
    obj_meta = BioFlowMLClass(df_meta,
                              out_dir_name = 'metadata',
                              label_feature = label_feature,
                              exclude_features = [id_column],
                              control_label = control_label,
                              lang = 'en')
    
    # Add label feature to the microbiome feature matrix
    obj_mb = merge_with_metadata(obj_mb, obj_meta, [label_feature])
    obj_mb.set_label_feature(label_feature, control_label)
    
    # Preprocess non-numerical features and missing values
    obj_mb = encode_and_impute_features(obj_mb, sort_by='sample_id')
    
    # Aggregate species data to specific taxonomic level
    # and trim taxa names
    aggregate_taxa_by_level(obj_mb, 'g', trim_taxa=True)
    obj_mb.df.to_csv(f'data/processed/{obj_mb.out_dir_name}l.csv', index=False)
    
    # Remove low varience features
    obj_mb = remove_low_variance_features(obj_mb, threshold=0.1)

    # Check data transformation distributions for all microbial features
    check_transformations(obj_mb)
    
    # Normalize and scale numeric features
    obj_mb = preprocess_numerical_features(obj_mb, exclude_features=obj_mb.exclude_features + [obj_mb.label_feature])
    
    # Save normalized feature matrix as csv if needed
    obj_mb.df.to_csv(f'data/processed/{obj_mb.out_dir_name}_normalized.csv', index=False)
    
    # Correlation analysis
    check_correlations(obj_mb)
    
    # Feature distribution comparison
    compare_distributions(obj_mb)

    # Binary classifier training and evaluation
    classify_binary(obj_mb)
    
    # Multiclass classifier
    train_multiclass_classifiers(obj_mb)


if __name__ == "__main__":
    main()