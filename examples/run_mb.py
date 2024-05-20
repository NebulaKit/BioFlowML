import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_distributions
from src.preprocessing.microbiome_preprocessing import trim_taxa_names, merge_with_metadata, aggregate_taxa_by_level
import src.utils.IO as io
import pandas as pd


def main():
    
    io.reset_logfile()
    
    # Create and initialize BioFlowML class instance for the microbiome data
    df_mb = pd.read_csv('data/synthetic/microbiome.csv')
    id_column = 'sample_id'
    obj_mb = BioFlowMLClass(df_mb,
                            out_dir_name = 'microbiome', # same as dataset name
                            exclude_features = [id_column],
                            lang = 'lv')
    
    # Create and initialize BioFlowML class instance for the metadata (to map label feature to microbiome data)
    df_meta = pd.read_csv('data/synthetic/metadata.csv')
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
    
    # TODO: need tp think of how to stop workflow from further execution if previous task fails
    # Aggregate species data to specific taxonomic level
    level_indicators = ['d', 'p', 'c', 'o', 'f', 'g', 's']
    level = 'd'
    aggregate_taxa_by_level(obj_mb, level)
    
    # Trim the lengthy taxa names for better vizualization
    trim_taxa_names(obj_mb)
    obj_mb.df.to_csv(f'data/processed/{obj_mb.out_dir_name}_level.csv', index=False)
 
    # Chech data distributions for all microbial features
    check_all_distributions(obj_mb)
    
    # Preprocess non-numerical features
    
    # Normalize and scale features
    
    # Correlation analysis
    
    # Differential distribution analysis
    
    # Feature selection and engineering
    
    # Classifier selection (autoML)
    
    # Binary classifier training and evaluation
    
    # Multiclass classifier

if __name__ == "__main__":
    main()