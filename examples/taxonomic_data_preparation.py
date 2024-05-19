import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.preprocessing.microbiome_preprocessing import transpose_otu_table, merge_with_metadata, filter_unclassified_taxa, trim_taxa_names, aggregate_taxa_by_level
import src.utils.IO as io
import pandas as pd


def prepare_species_level_feature_matrix(relative_data_path:str):
    
    df_asd = pd.read_csv(relative_data_path)
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    obj = BioFlowMLClass(df_asd, out_dir_name = out_dir_name, drop_features=['OTU'])
    
    # Transpose OTU Table for microbiome data
    obj = transpose_otu_table(obj)
    obj.df.to_csv(f'data/processed/{out_dir_name}_transposed.csv', index=False)
    
    # Filter taxa unclassified to species level
    # (use the most specific taxonomic rank for machine learning models)
    obj = filter_unclassified_taxa(obj, 's')
    
    # Trim taxa names and save as csv
    obj = trim_taxa_names(obj)
    obj.df.to_csv(f'data/processed/{out_dir_name}_species_level.csv', index=False)


def prepare_phylum_level_feature_matrix(relative_data_path:str):
    
    df_asd = pd.read_csv(relative_data_path)
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    obj = BioFlowMLClass(df_asd, out_dir_name = out_dir_name, drop_features=['OTU'])
    
    # Transpose OTU Table for microbiome data
    obj = transpose_otu_table(obj)
    
    # Aggregate species data to phylum level data
    obj = aggregate_taxa_by_level(obj, 'p')
    
    # Trim taxa names and save as csv
    obj = trim_taxa_names(obj)
    obj.df.to_csv(f'data/processed/{out_dir_name}_phylum_level.csv', index=False)

def main():
    
    io.reset_logfile()
    
    # For this example 16S rRNA OTU table containing gut microbiome species level 
    # taxonomic profiles of children with ASD is applied
    # Data source: https://www.kaggle.com/datasets/antaresnyc/human-gut-microbiome-with-asd
    relative_data_path = 'data/test/ASD/GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv'
    
    # Run example of species level taxonomix data preparation
    prepare_species_level_feature_matrix(relative_data_path)
    
    # Run example of phylum level taxonomix data preparation
    prepare_phylum_level_feature_matrix(relative_data_path)
    

if __name__ == "__main__":
    main()