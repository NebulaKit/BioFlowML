"""
Module for analyzing and visualizing data distributions for the original and transformed data.
"""

from src.BioFlowMLClass import BioFlowMLClass
from src.utils.monitoring import timeit, log_errors_and_warnings, log_execution
from src.utils.logger_setup import get_main_logger, get_logger
from src.utils.IOHandler import IOHandler
import src.utils.microbiome as mb
import src.translate as tr

from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro, anderson
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from joblib import Parallel, delayed
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


@log_errors_and_warnings
def check_normality(data: pd.core.series.Series, translations: dict = {}, alpha: float = 0.05):
    """
    Check for normality using Shapiro-Wilk, Lilliefors, and Anderson-Darling tests.

    Parameters:
        - data (pandas.core.series.Series): the input dataset to be checked
        - translations (dict, optional): a dictionary containing translations for test result labels
        - alpha (float, optional): significance level (default is 0.05)

    Returns:
        (bool, dict, str): A tuple containing:
            - True if at least two tests do not reject the null hypothesis (data is approximately normal),
              False otherwise.
            - A dictionary containing the test names as keys and their corresponding p-values or statistics.
            - A string containing the formatted results of the tests.

    Example:
        ```python
        is_normal, test_values, test_results = check_normality(data)
        ``` 
    """
    test_result_str = ''
    
    # Shapiro-Wilk test
    _, shapiro_pvalue = shapiro(data)
    if translations:
        temp_str = f'{tr.translate("histogram_labels.shapiro_p_value", translations)}: {shapiro_pvalue:.3f}'
        test_result_str += f'{temp_str}\n'

    # Lilliefors test
    _, lilliefors_pvalue = lilliefors(data)
    if translations:
        temp_str = f'{tr.translate("histogram_labels.lilliefors_p_value", translations)}: {lilliefors_pvalue:.3f}'
        test_result_str += f'{temp_str}\n'

    # Anderson-Darling test
    anderson_result = anderson(data)
    
    if translations:
        temp_str = (
            f'{tr.translate("histogram_labels.anderson_statistic", translations)}: '
            f'{anderson_result.statistic:.3f}\n'
            f'(α=0.05, {tr.translate("histogram_labels.anderson_critical_value", translations)}='
            f'{anderson_result.critical_values[2]:.3f})'
        )
        test_result_str += f'{temp_str}\n'
    
    
    test_values_dict = {"Shapiro-Wilk": shapiro_pvalue,
                        "Lilliefors": lilliefors_pvalue,
                        "Anderson-Darling": anderson_result}

    # Majotiry voting (cnt tests not rejecting null hypothesis)
    norm_not_rejected_cnt = sum(1 for pvalue in [shapiro_pvalue, lilliefors_pvalue] if float(pvalue) > alpha)
    if anderson_result.statistic < anderson_result.critical_values[2]:
      norm_not_rejected_cnt+=1

    # Check if data normality can be rejected based on the number of tests not rejecting the null hypothesis
    norm_not_rejected = norm_not_rejected_cnt >= 2
    
    # Return True if at least two tests do not reject the null hypothesis
    return norm_not_rejected, test_values_dict, test_result_str

@log_errors_and_warnings
def plot_transformations(obj: BioFlowMLClass, feature_name:str):
    """
    Applies various transformations to a specified feature in a DataFrame, plots the distributions of the transformed data, 
    and saves the resulting plots.

    Parameters:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing the DataFrame and additional configuration.
        feature_name (str): The name of the feature in the DataFrame to be transformed and plotted.

    Returns:
        dict: A dictionary indicating whether each transformation achieved normality (1) or not (0).

    Notes:
        The function performs the following transformations:
        1. Original data (no transformation)
        2. Logarithmic transformation (log10(1 + x))
        3. Min-Max scaling
        4. Standard scaling
        5. Robust scaling
        6. Max-Abs scaling
        7. Power transformation (Yeo-Johnson method)
        8. Quantile transformation to a normal distribution

        For each transformation, the function:
        - Applies the transformation to the feature.
        - Checks the normality of the transformed data using statistical tests.
        - Plots the histogram of the transformed data.
        - Saves the plot with appropriate labels and titles.

        If any transformation achieves normality, the output directory is 'normal_features'. Otherwise, it is 'other_features'.
    
    Example:
        ```python
        result = plot_transformations(obj, 'feature')
        ```
    """
    norm_result = {}
    normality_achieved = False
    
    x = obj.df[[feature_name]]
    
    transformations = [
        (None, "X"),
        (lambda x: np.log10(1 + x), "log10(1 + x)"),
        (lambda x: MinMaxScaler().fit_transform(x), "MinMaxScaler"),
        (lambda x: StandardScaler().fit_transform(x), "StandardScaler"),
        (lambda x: RobustScaler().fit_transform(x), "RobustScaler"),
        (lambda x: MaxAbsScaler().fit_transform(x), "MaxAbsScaler"),
        (lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x), "PowerTransformer (Yeo-Johnson)"),
        (lambda x: QuantileTransformer(output_distribution='normal', n_quantiles=x.shape[0]).fit_transform(x),
        "QuantileTransformer")
    ]
    
    # Load plot label translation for plots
    translations = tr.load_translations(obj.lang)
    
    # Create subplots for each data transformation method
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, (transform, title) in enumerate(transformations):
        ax = axes[i]

        x_transformed = x if transform is None else transform(x)

        if isinstance(x_transformed, np.ndarray):
            x_transformed = pd.DataFrame(x_transformed, index=x.index, columns=x.columns)

        x_transformed = x_transformed.iloc[:, 0]

        # Do statistical testing for data normality
        is_normal, _, _ = check_normality(x_transformed)

        if is_normal:
            sns.histplot(x_transformed, ax=ax, kde=True, color='#186948') # green
            norm_result[title] = 1
            normality_achieved = True
        else:
            sns.histplot(x_transformed, ax=ax, kde=True, color='#0d2980') # blue
            norm_result[title] = 0

        if i == 0 or i == 4:
            ax.set_ylabel(tr.translate("histogram_labels.y_axis", translations), fontsize=18)
        else:
            ax.set_ylabel('')

        ax.set_xlabel(tr.translate("histogram_labels.x_axis", translations), fontsize=18)
        ax.set_title('\n' + title, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)

    if 'microbiome' in obj.out_dir_name:
        fig.suptitle(f"{feature_name.replace('_',' ')}", ha='center', fontsize=26, fontstyle='italic')
    else:
        fig.suptitle(f"{feature_name}", ha='center', fontsize=26)
    
    plt.tight_layout()
    
    # Log results to main
    logger = get_main_logger()
    logger.info(f'{feature_name} normalized: {normality_achieved}')

    # Save result plot
    output_dir = 'normal_features' if normality_achieved else 'other_features'
    out_dir_name = IOHandler.get_absolute_path(f'../results/feature_analysis/distributions/{obj.out_dir_name}/{output_dir}', create_dir=True)
    index = obj.df.columns.get_loc(feature_name)
    out_file_path = os.path.join(out_dir_name, f'{index}_{mb.trim_id(feature_name)}.png')
    plt.savefig(out_file_path, dpi=250)
    plt.close()
    
    return norm_result

@timeit
@log_execution
@log_errors_and_warnings
def check_transformations(obj: BioFlowMLClass, n_features=None):
    """
    Applies various transformations to multiple features in a DataFrame, plots the distributions of the transformed data,
    and logs the results.

    Parameters:
        obj (BioFlowMLClass): An instance of BioFlowMLClass containing the DataFrame and additional configuration.
        n_features (int, optional): The number of features to transform. If None, all features in the DataFrame are used.

    Notes:
        The function performs the following steps:
        1. Selects the features to be transformed. If n_features is specified, only the first n_features are selected.
        Otherwise, all features are selected.
        2. Applies the plot_transformations function to each feature in parallel, excluding features specified in
        obj.exclude_features and the label feature (obj.label_feature).
        3. Logs the results of the transformations.
        
    Example:
        ```python
        check_all_transformations(obj, n_features=50)
        ```
    """
    # Ensure the specified n_features is within the bounds of the DataFrame
    if n_features is not None and (n_features < 1 or n_features > obj.df.shape[1]):
        raise ValueError(f"n_features must be between 1 and {obj.df.shape[1]}")
    
    # Check if results already present
    directory_path = f'results/feature_analysis/distributions/{obj.out_dir_name}'
    if os.path.isdir(directory_path):
        print(f'Directory [{directory_path}] already exists!')
        user_input = input(f'Skip [{obj.out_dir_name}] transformation distribution plotting (y/n)? ')
        if user_input.lower() != 'n':
            return
    
    # Features to transform and plot
    features = obj.df.iloc[:, :n_features].columns.tolist() if n_features else obj.df.columns
    
    # Process in parallel
    results = Parallel(n_jobs=-1)(
        delayed(plot_transformations)(obj, f) 
        for f in features
        if (f not in obj.exclude_features) and f != obj.label_feature
    )
    
    # Log the summary results
    log_results(results, obj.out_dir_name)

@log_errors_and_warnings
def log_results(results:list[dict], out_dir_name:str):
    """
    Logs the summary of transformation results for multiple features, including a count of features 
    that achieved normality.

    Parameters:
        results (dict): A list of dictionaries where each dictionary contains the results of normality checks 
                    for a feature's transformations from check_all_transformations method.
        out_dir_name (str): The name of the output directory where the log file will be saved.

    Notes:
        The function performs the following steps:
        1. Initializes a Counter to summarize the results.
        2. Iterates over each dictionary in the results list, updating the summary Counter with each dictionary.
        3. Converts the Counter to a DataFrame for easier logging.
        4. Counts how many features have at least one transformation that achieved normality.
        5. Logs the summary of the results, including the total count of features normalized, to a log file 
        in the specified output directory.
    
    Example:
        ```python
        results = [{'X': 0, 'log10(1 + x)': 1, 'MinMaxScaler': 0, 'QuantileTransformer': 1}, {'X': 0, 'log10(1 + x)': 0, 'MinMaxScaler': 0, 'QuantileTransformer': 1}]
        log_results(results, 'out_dir_name')
        ```
    """
    # Initialize summary Counter
    summary_counter = Counter()

    # Iterate over each dictionary in the list
    for result in results:
        # Update the summary Counter with the dictionary
        summary_counter.update(result)

    # Convert Counter to a regular dictionary
    result_summary = dict(summary_counter)
    df_result_summary = pd.DataFrame(result_summary.items(), columns=['Key', 'Value'])
    
    # Count of results where at least one value is 1 (normalized)
    normal_feature_cnt = sum(any(val == 1 for val in r.values()) for r in results)

    # PLog result summary
    out_dir_path = IOHandler.get_absolute_path(f'../results/feature_analysis/distributions/{out_dir_name}', create_dir=True)
    logger_path = os.path.join(out_dir_path,'result_summary.log')
    logger = get_logger(f'check_all_transformations [{out_dir_name}]', logger_path)
    logger.info(f'\n{df_result_summary}')
    logger.info(f'{normal_feature_cnt}/{len(results)} features normalized')
