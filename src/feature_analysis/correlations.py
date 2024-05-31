"""
Module for calculating and visualizing correlations between features and labels.
"""

from src.BioFlowMLClass import BioFlowMLClass
from src.utils.monitoring import timeit, log_errors_and_warnings, log_execution
from src.utils.logger_setup import get_main_logger
from src.utils.IOHandler import IOHandler
import src.translate as tr

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
import pandas as pd
import os


def get_correlations(obj: BioFlowMLClass, case_label: str = None, control_label: str = None, method: str = 'spearman', n_features=15):
    """
    Compute correlation matrix between features and a label of a BioFlowMLClass instance containing pandas DataFrame property.

    Args:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing the dataset.
        case_label (str, optional): Label indicating the cases in the dataset. Defaults to None.
        control_label (str, optional): Label indicating the controls in the dataset. Defaults to None.
        method (str, optional): Method for computing correlations. Supported methods are 'spearman', 'pearson', and 'kendall'. Defaults to 'spearman'.
        n_features (int, optional): Number of top correlated features to include in the output. Defaults to 15.

    Returns:
        pd.DataFrame: Correlation matrix between selected features and label.
        
    Example:
        ```python
        corr_matrix = get_correlations(obj, 'case', 'control')
        ```
    """
    methods = ['spearman', 'pearson', 'kendall']
    
    if method not in methods:
        raise ValueError(f'Supported correlation analysis methods are [{methods}]!')
    
    if not obj.label_feature:
        raise ValueError(f'BioFlowMLClass instance must have label_feature property set!')
    
    
    df = obj.df.copy()
    df = df.drop(columns=obj.exclude_features)
    
    if control_label and case_label != None:
        # Filter only case and control samples when control label provided
        controls_df = df[df[obj.label_feature] == control_label]
        cases_df = df[df[obj.label_feature] == case_label]
        df = pd.concat([controls_df, cases_df], ignore_index=True)
    
    # Compute correlations between features and label
    correlation_with_target = df.corr(method=method)[obj.label_feature]

    # Sort correlation values by absolute value, strongest correlations at the top
    sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

    # Filter features with strongest correlations
    top_correlations = sorted_correlation.head(n_features + 1)
    top_feature_names = top_correlations.index.tolist()

    # Compute the full correlation matrix based on selected features
    filtered_correlation_matrix = df[top_feature_names].corr(method='spearman')
    
    return filtered_correlation_matrix

def plot_correlations(obj: BioFlowMLClass, case_label: str = None, control_label: str = None, method: str = 'spearman', n_features=15):
    """
    Plot a heatmap of correlation matrix between features a label of a BioFlowMLClass instance containing pandas DataFrame property.

    Args:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing the dataset.
        case_label (str, optional): Label indicating the cases in the dataset. Defaults to None.
        control_label (str, optional): Label indicating the controls in the dataset. Defaults to None.
        method (str, optional): Method for computing correlations. Supported methods are 'spearman', 'pearson', and 'kendall'. Defaults to 'spearman'.
        n_features (int, optional): Number of top correlated features to include in the output. Defaults to 15.

    Example:
        ```python
        plot_correlations(obj, 'case', 'control')
        ```
    """
    methods = ['spearman', 'pearson', 'kendall']
    
    if method not in methods:
        raise ValueError(f'Supported correlation analysis methods are [{methods}]!')
    
    if not obj.label_feature:
        raise ValueError(f'BioFlowMLClass instance must have label_feature property set!')
    
    correlation_strategy = 'binary' if (control_label and case_label != None) else 'ovr'
    
    # Get the fitered correlation matrix
    filtered_correlation_matrix = get_correlations(obj, case_label, control_label, method, n_features)
    top_feature_names = filtered_correlation_matrix.columns
    
    # Trim all feature names to a maximum of 30 characters
    top_feature_names = [s[:30] for s in top_feature_names]
    
    # Load plot label translation for plots
    translations = tr.load_translations(obj.lang)
    
    encoded_features = obj.get_encoded_features()
    encoded_label_features = encoded_features[obj.label_feature]
    
    target_name = encoded_label_features[case_label] if correlation_strategy == 'binary' else obj.label_feature
    top_feature_names[0] = target_name
    
    logger = get_main_logger()
    logger.info(f'Plotting top {n_features} correlations with target: {target_name}')
    
    # Translate the value associated with the first key (label)
    if obj.lang == 'lv':
        translated_label = tr.translate(f"cohort.{target_name}", translations)
        if translated_label:
            top_feature_names[0] = translated_label
    
    # Plot the correlation heatmap
    plt.figure(figsize=(32, 25))  # Adjust size as needed
    sns.set_context("notebook", font_scale=2)
    cmap = sns.diverging_palette(220, 20, l=70, as_cmap=True)
    sns.heatmap(filtered_correlation_matrix, cmap=cmap, annot=True, xticklabels=top_feature_names, yticklabels=top_feature_names)
    title = f'{tr.translate("correlation_heatmap.title", translations)} "{top_feature_names[0]}"' if correlation_strategy == 'binary' else tr.translate("correlation_heatmap.title", translations)
    plt.title(title, fontsize=30, pad=20)
    plt.xticks(rotation=90, fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.tight_layout()
    
    # Save the plot
    l = top_feature_names[0].replace(' ','_')
    out_dir_name = IOHandler.get_absolute_path(f'../results/feature_analysis/correlations/{obj.out_dir_name}/{correlation_strategy}', create_dir=True)
    out_file_path = os.path.join(out_dir_name, f'{l}_top_{n_features}_{method}_{correlation_strategy}_correlations.png')
    plt.savefig(out_file_path, dpi=250)
    plt.close()

@timeit
@log_execution
@log_errors_and_warnings
def check_correlations(obj: BioFlowMLClass, method='binary'):
    """
    Check and plot correlation heatmaps between features and label of a BioFlowMLClass instance containing pandas DataFrame.

    Args:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing the dataset.
        method (str, optional): Method for correlation analysis. Supported methods are 'binary' and 'multi'. Defaults to 'binary'.

    Example:
        ```python
        check_correlations(obj, 'multi')
        ```
    """
    methods = ['binary', 'multi']
    if method not in methods:
        raise ValueError(f'Supported correlation analysis methods are [{methods}]!')
    
    if not obj.label_feature:
        raise ValueError(f'BioFlowMLClass instance must have label_feature property set!')
    
    # Check if results already present
    directory_path = f'results/feature_analysis/correlations/{obj.out_dir_name}/{method}'
    if os.path.isdir(directory_path):
        print(f'Directory [{directory_path}] already exists!')
        user_input = input(f'Skip [{obj.out_dir_name}] correlation plotting (y/n)? ')
        if user_input.lower() != 'n':
            return
    
    labels_encoded  = obj.df[obj.label_feature].unique()
    labels = obj.get_encoded_features()[obj.label_feature]
    control_label_encoded = labels.index(obj.control_label) if (method == 'binary' and obj.control_label) else None
    
    if control_label_encoded:
        Parallel(n_jobs=-1)(
            delayed(plot_correlations)(
                obj,
                l,
                control_label_encoded # if None then 'multi-class', otherwise 'cases-vs-controls'
            )
            for l in labels_encoded if l != control_label_encoded
        )
    else:
        plot_correlations(obj)
    