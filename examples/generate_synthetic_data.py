import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.sdv_utils import generate_synthetic_data, generate_merged_synthetic_data
from src.BioFlowMLClass import BioFlowMLClass
from src.utils.IOHandler import IOHandler
import pandas as pd

def main():
    
    IOHandler.reset_logfile()
    
    # Load transposed OTU table of microbiome data
    relative_data_path = 'data/processed/microbiome.csv'
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    df = pd.read_csv(relative_data_path)
    obj = BioFlowMLClass(df, out_dir_name = out_dir_name, lang = 'lv')
    
    # Generate synthetic dataset
    relative_synthetic_data_path = generate_synthetic_data(obj, evaluate=True)
    print(f'relative_synthetic_data_path: {relative_synthetic_data_path}')
    
    # Load metadata
    relative_data_path = 'data/raw/metadata.tsv'
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    df = pd.read_csv(relative_data_path, sep='\t')
    obj2 = BioFlowMLClass(df, out_dir_name = out_dir_name, lang = 'lv')
    
    # Generate merged synthetic dataset
    relative_merged_synthetic_data_path = generate_merged_synthetic_data(obj, obj2, 'sample_id', evaluate=True)
    print(f'relative_merged_synthetic_data_path: {relative_merged_synthetic_data_path}')

if __name__ == "__main__":
    main()