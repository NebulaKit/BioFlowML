import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_distributions
from src.preprocessing.metadata_processing import create_preprocessing_pipeline, start_pipeline_wizard
import src.utils.IO as io
import pandas as pd
import joblib
import os
from src.utils import serialize_dict


def main():
    
    io.reset_logfile()
    
    # Create and initialize BioFlowML class instance
    relative_data_path = 'data/synthetic/metadata_synthetic.csv'
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    df = pd.read_csv(relative_data_path)
    obj = BioFlowMLClass(df,                 
                    out_dir_name = out_dir_name,
                    label_feature = 'subject_group',
                    exclude_features = ['sample_id'],
                    control_label = 'Control',
                    lang = 'lv')
    
    # Chech data distributions for all microbial features
    # Check if exists and prompt
    #check_all_distributions(obj)
    
    # Preprocessing pipeline
    pipeline=None
    
    # Check if pipeline saved
    directory_path = "preprocessing/pipelines"
    pipeline_file_name = f'{obj.out_dir_name}_preprocessing_pipeline.joblib'

    if os.path.isdir(directory_path):
        if pipeline_file_name in os.listdir(directory_path):
            # prompt user before load
            pipeline = joblib.load(f'{directory_path}/{pipeline_file_name}')

    if not pipeline:
        pipeline = start_pipeline_wizard(obj)
    
    df_preprocessed = pipeline.fit_transform(obj.df)

    # TODO: retreive encoded features for loaded pipelines somehow
    print(f'encoded_features: {serialize_dict(obj.get_encoded_features())}')
    df_preprocessed.to_csv(f'data/processed/{out_dir_name}_processed_v2.csv', index=False)
    
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