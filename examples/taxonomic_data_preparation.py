import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.BioFlowMLClass import BioFlowMLClass
from src.preprocessing.microbiome_preprocessing import transpose_otu_table, trim_taxa_names, aggregate_taxa_by_level
from src.utils.IOHandler import IOHandler
import pandas as pd


def prepare_species_level_feature_matrix(relative_data_path:str):
    
    df_asd = pd.read_csv(relative_data_path)
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    obj = BioFlowMLClass(df_asd, out_dir_name = out_dir_name, drop_features=['OTU'])
    
    # Transpose OTU Table for microbiome data
    obj = transpose_otu_table(obj)
    
    # Prepare species level feature matrix with unclassified taxa dropped
    obj = aggregate_taxa_by_level(obj, 's', drop_unclassified=True)
    
    # Trim taxa names and save as csv
    obj = trim_taxa_names(obj)
    obj.df.to_csv(f'data/processed/{out_dir_name}_species_level.csv', index=False)


def prepare_phylum_level_feature_matrix(relative_data_path:str):
    
    df_asd = pd.read_csv(relative_data_path)
    out_dir_name = relative_data_path.split('/')[-1].split('.')[0] # same as dataset name
    obj = BioFlowMLClass(df_asd, out_dir_name = out_dir_name, drop_features=['OTU'])
    
    # Transpose OTU Table for microbiome data
    obj = transpose_otu_table(obj)
    
    # Aggregate species data to phylum level data and drop unclassified
    obj = aggregate_taxa_by_level(obj, 'p', drop_unclassified=True)
    
    # Trim taxa names and save as csv
    obj = trim_taxa_names(obj)
    obj.df.to_csv(f'data/processed/{out_dir_name}_phylum_level.csv', index=False)

def main():
    
    IOHandler.reset_logfile()
    
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