import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_distributions
from src.preprocessing.microbiome_preprocessing import  merge_with_metadata, aggregate_taxa_by_level
from src.preprocessing import get_preprocessing_pipeline, get_numerical_feature_pipeline
import src.utils.IO as io
from src.utils import serialize_dict
import pandas as pd


# TODO: need tp think of how to stop workflow from further execution if previous task fails
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
    
    # Preprocess non-numerical features and missing values
    pipeline = get_preprocessing_pipeline(obj_mb, sort_by='sample_id')
    obj_mb.df = pipeline.fit_transform(obj_mb.df)
    
    # Aggregate species data to specific taxonomic level
    # and trim taxa names
    aggregate_taxa_by_level(obj_mb, 'g', trim_taxa=True)

    # Chech data distributions for all microbial features
    check_all_distributions(obj_mb)
    
    # Normalize and scale numeric features
    normalization_pipeline = get_numerical_feature_pipeline(obj_mb.df, exclude_features=obj_mb.exclude_features + [obj_mb.label_feature])
    obj_mb.df = normalization_pipeline.fit_transform(obj_mb.df)
    
    obj_mb.df.to_csv(f'data/processed/{obj_mb.out_dir_name}.csv', index=False)
    
    # Select numerical columns
    # df = pd.read_csv('data/processed/microbiome_with_labels.csv')
    # numerical_columns = df.select_dtypes(include=['number']).columns
    # numerical_features = numerical_columns.tolist()
    # num_imputer = NumericalImputer(features_to_impute=numerical_features, label_feature='subject_group')
    # df_imputed = num_imputer.transform(df)
    # df_imputed.to_csv(f'data/processed/microbiome_with_labels_imputed.csv', index=False)
    
    # Normalize and scale features
    
    # Correlation analysis
    
    # Differential distribution analysis
    
    # Feature selection and engineering
    
    # Classifier selection (autoML)
    
    # Binary classifier training and evaluation
    
    # Multiclass classifier

if __name__ == "__main__":
    main()