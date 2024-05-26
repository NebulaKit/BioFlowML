import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_transformations
from src.preprocessing import get_preprocessing_pipeline
import pandas as pd
from src.utils.IOHandler import IOHandler


def main():
    
    IOHandler.reset_logfile()
    
    # Create and initialize BioFlowML class instance for the microbiome data
    df = pd.read_csv('data/test/blood_samples.csv')
    obj = BioFlowMLClass(df,
                            out_dir_name = 'blood_samples', # same as dataset name
                            label_feature = 'Disease',
                            lang = 'lv')
    
        
    # Preprocess non-numerical features and missing values
    pipeline = get_preprocessing_pipeline(obj)
    obj.df = pipeline.fit_transform(obj.df)
    
    # Chech data distributions for all microbial features
    check_all_transformations(obj)


if __name__ == "__main__":
    main()