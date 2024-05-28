import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_transformations
from src.feature_analysis.correlations import check_correlations
from src.preprocessing.microbiome_preprocessing import  merge_with_metadata, aggregate_taxa_by_level
from src.preprocessing import get_preprocessing_pipeline, get_numerical_feature_pipeline
from src.model_training.binary_classification import train_binary_classifiers
from src.model_training.multiclass_classification import train_multiclass_classifiers
from src.utils.IOHandler import IOHandler
from src.utils import serialize_dict
import pandas as pd


def main():
    
    IOHandler.reset_logfile()
    
    # Read microbiome feature matrix
    df_mb = pd.read_csv('data/synthetic/microbiome.csv')
    
    # Create and initialize BioFlowML class instance for the microbiome data
    id_column = 'sample_id'
    obj_mb = BioFlowMLClass(df_mb,
                            out_dir_name = 'microbiome', # same as dataset name
                            exclude_features = [id_column],
                            lang = 'lv')
    
    # Read metadata feature matrix
    df_meta = pd.read_csv('data/synthetic/metadata.csv')
    
    # Create and initialize BioFlowML class instance for the metadata (to map label feature to microbiome data)
    label_feature = 'subject_group'
    control_label = 'Control'
    obj_meta = BioFlowMLClass(df_meta,
                              out_dir_name = 'metadata', # same as dataset name
                              label_feature = label_feature,
                              exclude_features = [id_column],
                              control_label = control_label,
                              lang = 'lv')
    
    # Add label feature to the microbiome feature matrix
    obj_mb = merge_with_metadata(obj_mb, obj_meta, [label_feature])
    obj_mb.set_label_feature(label_feature, control_label)
    
    # Preprocess non-numerical features and missing values
    pipeline = get_preprocessing_pipeline(obj_mb, sort_by='sample_id')
    obj_mb.df = pipeline.fit_transform(obj_mb.df)
    
    # Aggregate species data to specific taxonomic level
    # and trim taxa names
    aggregate_taxa_by_level(obj_mb, 'g', trim_taxa=True)

    # Check data transformation distributions for all microbial features
    check_transformations(obj_mb)
    
    # Normalize and scale numeric features
    normalization_pipeline = get_numerical_feature_pipeline(obj_mb.df, exclude_features=obj_mb.exclude_features + [obj_mb.label_feature])
    obj_mb.df = normalization_pipeline.fit_transform(obj_mb.df)
    
    # Save normalized feature matrix as csv
    obj_mb.df.to_csv(f'data/processed/{obj_mb.out_dir_name}_normalized.csv', index=False)
    
    # Correlation analysis
    check_correlations(obj_mb)
    
    # Binary classifier training and evaluation
    train_binary_classifiers(obj_mb)
    
    # Multiclass classifier
    train_multiclass_classifiers(obj_mb)


if __name__ == "__main__":
    main()