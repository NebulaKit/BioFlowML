"""
Module for generating synthetic datasets using SDV (Synthetic Data Vault).
"""

from src.BioFlowMLClass import BioFlowMLClass
from src.utils.logger_setup import get_logger
from src.utils.IOHandler import IOHandler
from src.utils.monitoring import log_errors_and_warnings, timeit
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import pandas as pd
import os

@timeit
@log_errors_and_warnings
def generate_synthetic_data(obj: BioFlowMLClass, num_rows=1000, evaluate=False):
    """
    Generates a synthetic dataset using SDV.

    Args:
        obj (BioFlowML): An instance of BioFlowML containing the original dataset.
        num_rows (int, optional): Number of rows to generate in the synthetic dataset. Defaults to 1000.
        evaluate (bool, optional): Whether to evaluate the quality of the generated synthetic dataset. 
            If set to True, evaluation might take several minutes. Defaults to False.

    Returns:
        str: Relative file path to the generated synthetic dataset.

    Usage:
        synthetic_file_path = generate_synthetic_data(obj, num_rows=1000, evaluate=False)
    """
    
    relative_out_dir_path = '../data/synthetic'
    abs_out_dir_path = IOHandler.get_absolute_path(relative_out_dir_path, create_dir=True)
    
    # Metadata avout the provided dataset, e.g. column names, data types etc.
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(obj.df)
    
    metadata_json_file_path = os.path.join(abs_out_dir_path, f'{obj.out_dir_name}_sdv_single_table_metadata.json')

    # Check if the file exists
    if os.path.exists(metadata_json_file_path):
        overwrite = input(f"A metadata json file ('{metadata_json_file_path}') already exists in. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() == 'y':
            os.remove(metadata_json_file_path)
            # Proceed with saving metadata to JSON file
            metadata.save_to_json(filepath=metadata_json_file_path)
            print("File overwritten successfully.")
        else:
            use_existing = input("Use the existing metadata json file to generate a synthetic dataset? (y/n): ")
            if use_existing.lower() == 'y':
                metadata = SingleTableMetadata.load_from_json(filepath=metadata_json_file_path)
    else:
        # File does not exist, proceed with saving metadata to JSON file
        metadata.save_to_json(filepath=metadata_json_file_path)
    
    
    # Gaussian Copula is a statistical method used for modeling the dependence structure
    # between random variables. In the context of synthetic data generation, Gaussian Copula
    # can be used to generate synthetic data that captures the correlation structure present
    # in the original data.
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(obj.df)
    
    
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    out_file_path = os.path.join(abs_out_dir_path, f'{obj.out_dir_name}_synthetic.csv')
    relative_out_file_path = os.path.join(relative_out_dir_path,f'{obj.out_dir_name}_synthetic.csv')
    synthetic_data.to_csv(out_file_path, index=False)
    
    if evaluate:
        # Call evaluate_quality function
        quality_report = evaluate_quality(obj.df, synthetic_data, metadata)
        quality_report_file_path = os.path.join(abs_out_dir_path, f'{obj.out_dir_name}_quality_report.pkl')
        quality_report.save(filepath=quality_report_file_path)

        # Convert the quality_report to a string
        report_summary = quality_report.get_properties()
        overall_score = quality_report.get_score()
        
        # Log the quality report
        log_file_path = os.path.join(abs_out_dir_path, f'{__name__}.log')
        logger = get_logger(__name__, log_file_path)
        logger.info(f"Quality Report for synthetic dataset ({out_file_path}):\n{report_summary}\nOverall Score (Average): {overall_score}")
    
    return relative_out_file_path

def generate_merged_synthetic_data(obj1: BioFlowMLClass, obj2: BioFlowMLClass, merge_on, num_rows=1000, evaluate=False):
    """
    Generates a synthetic dataset by merging two datasets and then applying SDV.

    Args:
        obj1 (BioFlowML): An instance of BioFlowML containing the first dataset.
        obj2 (BioFlowML): An instance of BioFlowML containing the second dataset.
        merge_on (str): The column name to merge the datasets on.
        num_rows (int, optional): Number of rows to generate in the synthetic dataset. Defaults to 1000.
        evaluate (bool, optional): Whether to evaluate the quality of the generated synthetic dataset. 
            If set to True, evaluation might take several minutes. Defaults to False.

    Returns:
        str: Relative file path to the generated synthetic dataset.

    Usage:
        synthetic_file_path = generate_merged_synthetic_data(obj1, obj2, merge_on, num_rows=1000, evaluate=False)
    """
    
    if merge_on not in obj1.df.columns or merge_on not in obj2.df.columns:
        raise ValueError(f'The provided id field ({merge_on}) not present in the DataFrame/-s!')
    
    df_merged = pd.merge(obj1.df, obj2.df, on=merge_on)
    out_dir_name = f'{obj1.out_dir_name}_{obj2.out_dir_name}'
    obj = BioFlowMLClass(df_merged, out_dir_name = out_dir_name, lang = 'lv')
    relative_synthetic_file_path = generate_synthetic_data(obj, evaluate=evaluate)
    
    return relative_synthetic_file_path
    