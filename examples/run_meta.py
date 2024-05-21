import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_distributions
from src.preprocessing import start_pipeline_wizard, get_preprocessing_pipeline
import src.utils.IO as io
import pandas as pd
import joblib
import os
from src.utils import serialize_dict


def main():
    
    io.reset_logfile()
    
    # Create and initialize BioFlowML class instance
    df = pd.read_csv('data/synthetic/metadata_missing_values.csv')
    obj = BioFlowMLClass(df,                 
                    out_dir_name = 'metadata',
                    label_feature = 'subject_group',
                    exclude_features = ['sample_id'],
                    control_label = 'Control',
                    lang = 'lv')
    
    # Preprocess non-numerical features and missing values
    # TODO: integrate fit_transform in the preprocessing module
    pipeline = get_preprocessing_pipeline(obj)
    obj.df = pipeline.fit_transform(obj.df)
    # print(f'encoded_features: {serialize_dict(obj.get_encoded_features())}')
    
    # Check data distributions for all metadata features
    check_all_distributions(obj)
    
    
    obj.df.to_csv(f'data/processed/{obj.out_dir_name}_processed.csv', index=False)
    
    # Normalize and scale features
    
    # Correlation analysis
    
    # Differential distribution analysis
    
    # Feature selection and engineering
    
    # Classifier selection (autoML)
    
    # Binary classifier training and evaluation
    
    # Multiclass classifier
    
    # Anova
    
    
if __name__ == "__main__":
    main()