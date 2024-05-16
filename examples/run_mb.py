import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.feature_analysis.distributions import check_all_distributions
from src.preprocessing.microbiome_preprocessing import filter_unclassified_taxa, trim_taxa_names
import src.utils.IO as io
import pandas as pd


def main():
    
    io.reset_logfile()
    
    # Create and initialize BioFlowML class instance
    relative_data_path = 'data/synthetic/microbiome_synthetic.csv'
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    df = pd.read_csv(relative_data_path)
    obj = BioFlowMLClass(df,
                         out_dir_name = out_dir_name,
                         label_feature = 'subject_group',
                         exclude_features = ['sample_id'],
                         control_label = 'Control',
                         lang = 'lv')
    
    # Filter taxa unclassified to genus level
    filter_unclassified_taxa(obj, 'g')
    
    # Trim the lengthy taxa names for better vizualization
    trim_taxa_names(obj)

    # Chech data distributions for all microbial features
    #check_all_distributions(obj)
    
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