import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.preprocessing.microbiome_preprocessing import transpose_otu_table, merge_with_metadata, filter_unclassified_taxa
import src.utils.IO as io
import pandas as pd


def main():
    
    io.reset_logfile()
    
    # Create and initialize BioFlowML class instance
    relative_data_path = 'data/raw/features-taxlevel-6.tsv'
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    df = pd.read_csv(relative_data_path, sep='\t')
    obj = BioFlowMLClass(df, out_dir_name = out_dir_name, lang = 'lv')
    
    # Transpose OTU Table for microbiome data
    obj = transpose_otu_table(obj, sort_samples_by_id=True)
    
    # Join with metadata labels (and other metadata features if needed)
    relative_data_path_meta = 'data/processed/metadata_original.csv'
    out_dir_name_meta = relative_data_path.split('/')[-1].split('.')[0]
    df_meta = pd.read_csv(relative_data_path_meta)
    label_feature = 'subject_group'
    obj_meta = BioFlowMLClass(df_meta, out_dir_name_meta, label_feature=label_feature)
    obj = merge_with_metadata(obj, obj_meta, [label_feature])
    # obj.df.to_csv('data/processed/microbiome.csv', index=False)
    
    # Filter unclassified taxa (option - level: s/g/f)
    obj = filter_unclassified_taxa(obj, 'g')
    obj.df.to_csv('data/processed/microbiome_filtered.csv', index=False)
    

if __name__ == "__main__":
    main()