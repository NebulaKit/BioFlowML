"""
Module for analyzing and visualizing data distributions for the original and transformed data.
"""

from src.utils.monitoring import timeit, log_errors_and_warnings
from src.utils.logger_setup import get_main_logger, get_logger
from src.BioFlowMLClass import BioFlowMLClass
import src.feature_analysis as fa
import src.utils.microbiome as mb
import src.utils.IO as io
from src.utils.IOHandler import IOHandler
import src.translate as tr
from scipy.stats import shapiro, anderson, skew, kurtosis
from statsmodels.stats.diagnostic import lilliefors
from sklearn.preprocessing import PowerTransformer
from joblib import Parallel, delayed
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
def plot_transformations(obj: BioFlowMLClass, feature_name: str, log_transform_type: int = 1, color: str = '#0d2980'):
    """
    Plot various transformations of a numerical feature's distribution and save the plots.

    Parameters:
        - obj (BioFlowML): The BioFlowML object containing the DataFrame.
        - feature_name (str): Name of the numerical feature to be transformed and plotted.
        - log_transform_type (int): Type of logarithmic transformation to apply. Default is 1.
                                    - 1: Natural logarithm (ln).
                                    - 2: Base-2 logarithm.
                                    - 3: Base-10 logarithm.
        - color (str): Color to be used in plots. Default is '#0d2980'.

    Returns:
    Tuple containing:
        - norm_result (Dict[str, int]): Dictionary indicating which normalization methods were applied.
                                        Keys:
                                            - 'norm_by_default': 1 if feature was normally distributed initially or with no transformation, else 0.
                                            - 'norm_by_log': 1 if feature was normalized by logarithmic transformation, else 0.
                                            - 'norm_by_yeo_johnson': 1 if feature was normalized by Yeo-Johnson transformation, else 0.
        - is_numerical (bool): Indicates whether the specified feature is numerical or not.
        - feature_name (str): The name of the feature.

    Notes:
        - This function plots the original distribution of the specified numerical feature along with distributions after applying various transformations.
        - It saves the generated plots in appropriate directories based on whether the data achieves normality or not.
        - Additionally, it logs the normalization summary results in the 'Normalized_Features' directory if any features have been succesfully normalized.

    Example usage:
        ```python
        norm_result, is_numerical, feature_name = plot_transformations(obj, 'metadata', log_transform_type=1, color='#7318f2')
        ```
    """
    normality_achieved = False # set to True if data distribution is normal initially or with either transformation method
    norm_result = {'norm_by_default': 0,
                   'norm_by_log': 0,
                   'norm_by_yeo_johnson': 0}
    is_numerical = fa.is_numerical(obj.df[feature_name])
    if not is_numerical:
        return norm_result, is_numerical, feature_name

    # Define transformations to apply
    transformations = [(None, "X")]
    # Add user-specified log transformation
    match log_transform_type:
        case 1:
            transformations.append((np.log1p, "ln(1 + x)"))
        case 2:
            transformations.append((lambda x: np.log2(1 + x), "log2(1 + x)"))
        case 3:
            transformations.append((lambda x: np.log10(1 + x), "log10(1 + x)"))
    # Add Yeo-Johnson power transformation
    # transformations.append((lambda x: power_transform(np.array(x).reshape(-1, 1), method='yeo-johnson', standardize=False).flatten().tolist(), "Yeo-Johnson"))
    transformations.append((lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x), "PowerTransformer (Yeo-Johnson)"))
    
    # Plotting configurations
    colors = sns.light_palette(color, n_colors=6)
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.6)
    fig, axes = plt.subplots(2, len(transformations), figsize=(6.7*len(transformations), 10))
    plt.subplots_adjust(hspace=0.5)

    # Adjust position of the second row of subplots to add space between rows
    bottom_row = axes[1, :]
    for ax in bottom_row:
        pos = ax.get_position()
        pos.y0 += 0.35  # Adjust this value to add more or less space between rows
        ax.set_position(pos)
    
    # Load label translation for plots
    translations = tr.load_translations(obj.lang)
    
    # Plot data distributions as histograms and barplots for each transformation
    for i, (transform, title) in enumerate(transformations):
        
        row = i // len(transformations)
        col = i % len(transformations)
        
        
        X = obj.df[[feature_name]]
        # data = X if transform is None else pd.DataFrame(transform(X).flatten(), index=X.index, columns=X.columns)
        data = X if transform is None else transform(X)
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, index=X.index, columns=X.columns)
        
        data = data.iloc[:, 0]
        
        is_normal, _, test_result_str = check_normality(data, translations)
                
        if is_normal:
            normality_achieved = True

            # Logarithm
            if 'ln' in title or 'log' in title:
                norm_result['norm_by_log'] = 1
            # Yeo-Johnson
            elif title == transformations[2][1]:
                norm_result['norm_by_yeo_johnson'] = 1
            # Initially normal
            else:
                norm_result['norm_by_default'] = 1
        
        # Plot histogram
        sns.histplot(data, kde=True, bins='auto', color=colors[i+2], ax=axes[row, col])
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel(tr.translate("histogram_labels.y_axis", translations))
        axes[row, col].tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels for better fit

        # Plot boxplot
        sns.boxplot(data=data, orient='v', width=0.2, color=colors[i+2], showfliers=True, ax=axes[row+1, col])
        axes[row+1, col].set_title(title)
        axes[row+1, col].set_xlabel('')
        axes[row+1, col].set_ylabel(tr.translate("boxplot_labels.y_axis", translations))
        axes[row+1, col].set_xticklabels([])
        
        # Calculate skewness and kurtosis
        skewness = skew(data)
        kurt = kurtosis(data)
        success_text = tr.translate("histogram_labels.yes", translations) if is_normal else tr.translate("histogram_labels.no", translations)
        axes[row, col].set_title(f'{tr.translate("histogram_labels.skewness", translations)}: {skewness:.3f}\n'
                                 f'{tr.translate("histogram_labels.kurtosis", translations)}: {kurt:.3f}\n\n'
                                 f'{test_result_str}\n'
                                 f'{tr.translate("histogram_labels.normal_distribution", translations)}: {success_text}\n\n'
                                 f'{title}')

        
    fig.suptitle(f"{feature_name}\n", ha='center', fontsize=26)
    plt.tight_layout()
    
    # Save output figure
    output_dir = 'Normal_features' if normality_achieved else 'Non_Normal_Features'
    out_dir_name = io.get_absolute_path(f'../data/processed/distributions/{obj.out_dir_name}/{output_dir}', create_dir=True)
    index = obj.df.columns.get_loc(feature_name)
    out_file_path = os.path.join(out_dir_name, f'{index}_{mb.trim_id(feature_name)}.png')
    plt.savefig(out_file_path, dpi=250)
    plt.close()
    
    # Log normalization result   
    logger = get_main_logger()
    logger.debug(f'{feature_name} - NORMALIZED: {is_normal}')
    
    return norm_result, is_numerical, feature_name

@timeit
@log_errors_and_warnings
def check_all_distributions(obj: BioFlowMLClass):
    """
    Perform distribution analysis and transformation on all features of the dataset.

    Parameters:
        - obj (BioFlowMLClass): The BioFlowML object containing the dataset.

    Example usage:
        ```python
        check_all_distributions(obj)
        ```
    """
    # Check if the directory exists
    print(f'project_root_dir: {IOHandler.get_project_root_dir()}')
    if os.path.isdir(f'{IOHandler.get_project_root_dir()}/data/processed/distributions/{obj.out_dir_name}'):
        user_input = input(f'Directory for {obj.out_dir_name} distributions detected. Skip distribution analysis (y/n)? ')
        if user_input.lower() == 'y':
            return
            
    # Main logger
    logger = get_main_logger()
    logger.debug('-' * 20 + f' STARTING DISTRIBUTION ANALYSIS FOR {obj.out_dir_name} FEATURES ' + '-' * 20)
    
    # Prompt the user to choose a logarithmic transformation from given options
    log_transform_name = ['ln(1 + x)', 'log2(1 + x)', 'log10(1 + x)']
    log_transform_type = -1
    while log_transform_type not in ['1','2','3']:
        log_transform_type = input(f"Which logarithmic transformation method to apply to '{obj.out_dir_name}' dataset features?  \n(1) {log_transform_name[0]}\n(2) {log_transform_name[1]}\n(3) {log_transform_name[2]}\nOption: ")
        if log_transform_type not in ['1', '2', '3']:
            print("Unsupported option! Please choose option 1, 2, or 3.\n")
    
    # Plot color options
    plot_color = '#0d2980'
    if 'microbiome' in obj.out_dir_name:
        plot_color = "#186948"
    elif 'metadata' in obj.out_dir_name:
        plot_color = "#007a8c"
    
    # Run distribution and transformation analysis in parallel
    results = Parallel(n_jobs=-1)(
        delayed(plot_transformations)(
            obj,
            column,
            int(log_transform_type),
            plot_color) 
        for column in obj.df.columns if (column not in obj.exclude_features) and column != obj.label_feature)
    
    # Collect the summary statistics of normalized features
    norm_by_default_list = []
    norm_by_log_list = []
    norm_by_yeo_johnson_list = []
    not_numerical_features = []
    
    for result in results:
        if result is not None:
            # Unpack the tuple returned by each worker
            norm_result, is_numerical, feature_name = result
            # Aggregate the results
            if norm_result['norm_by_default']:
                norm_by_default_list.append(feature_name)
            
            if norm_result['norm_by_log']:
                norm_by_log_list.append(feature_name)
                
            if norm_result['norm_by_yeo_johnson']:
                norm_by_yeo_johnson_list.append(feature_name)
                
            if not is_numerical:
                not_numerical_features.append(feature_name)
        
    # Count all normalized features after parallel processing
    norm_distr_all = np.concatenate([norm_by_default_list, norm_by_log_list, norm_by_yeo_johnson_list])
    total_norm_cnt = len(np.unique(norm_distr_all))
    if total_norm_cnt > 0:
        
        # Initialize normalization statistics logger
        out_dir_path = io.get_absolute_path(f'../data/processed/distributions/{obj.out_dir_name}/Normal_features')
        normality_logger_path = os.path.join(out_dir_path,'logfile.log')
        norm_logger = get_logger(f'distribution_analysis_{obj.out_dir_name}', normality_logger_path)
        
        # Log normalization results
        norm_logger.debug('-' * 20 + f' RESULTS for {obj.out_dir_name}' + '-' * 20)
        norm_logger.debug(f'Initially normal ({len(norm_by_default_list)}/{total_norm_cnt}): {norm_by_default_list}')
        norm_logger.debug(f'Normalized by {log_transform_name[int(log_transform_type)-1]} ({len(norm_by_log_list)}/{total_norm_cnt}): {norm_by_log_list}')
        norm_logger.debug(f'Normalized by Yeo_Johnson ({len(norm_by_yeo_johnson_list)}/{total_norm_cnt}): {norm_by_yeo_johnson_list}')
        norm_logger.debug(f'{total_norm_cnt} features normalized out of total {len(obj.df.columns)} features')
        norm_logger.debug(f'{total_norm_cnt} features normalized out of total {len(obj.df.columns) - len(not_numerical_features)} quantitative features')
        norm_logger.debug(f'Qualitative features: {not_numerical_features}') 
    
    logger.debug('-' * 20 + f' FINISHED ANALYSIS FOR {obj.out_dir_name} FEATURES ' + '-' * 20)
