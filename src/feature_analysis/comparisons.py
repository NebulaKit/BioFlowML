"""
Module for comparing feature distributions between different lasses and visualizing them.
"""

from src.BioFlowMLClass import BioFlowMLClass
from src.utils.monitoring import timeit, log_errors_and_warnings, log_execution
from src.model_training import translate_label_names, get_translation
from src.utils.logger_setup import get_main_logger
from src.utils.IOHandler import IOHandler
import src.translate as tr

from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
import pandas as pd
import numpy as np
import os


def mann_whitney_association_test(X: pd.DataFrame, label_feature, alpha=0.05, method='fdr_bh'):
    """
    Perform Mann-Whitney U test for each feature in the data and apply multiple testing correction.

    Parameters:
    - data: Feature matrix where each row represents a sample and each column represents a feature.
    - labels: Label vector indicating the class for each sample (e.g., 0 for control, 1 for case).
    - alpha: Significance level.
    - method: Method for multiple testing correction (e.g., 'fdr_bh' for Benjamini-Hochberg).

    Returns:
    - p_values_corrected: Array of corrected p-values for each feature.
    - fold_changes: Array of fold changes for each feature.
    """
    p_values_corrected = []
    significant_features = []

    for feature in X.columns:
        if feature != label_feature:
          group1 = X[(X[label_feature] == 0)][feature]  # Control group
          group2 = X[(X[label_feature] == 1)][feature]  # Case group

          # Perform Mann-Whitney U test
          _, p_value = mannwhitneyu(group1, group2)

          # Apply multiple testing correction
          reject, p_value_corrected, _, _ = multipletests([p_value], alpha=alpha, method=method)

          if reject[0]:  # Significant association and higher mean in case group
              significant_features.append(feature)
              p_values_corrected.append(p_value_corrected[0])

    return significant_features, p_values_corrected

def plot_boxplots(obj: BioFlowMLClass, features, case_data, control_data, case_encoding, p_values=None):
    
    labels = obj.get_encoded_features()[obj.label_feature]
    control_label = obj.control_label
    case_label = labels[case_encoding]
    control_label_translated = get_translation(obj, f'cohort.{control_label}').lower()
    case_label_translated = get_translation(obj, f'cohort.{case_label}').lower()
    
    # Create a DataFrame for plotting
    data = []
    for f, case, control in zip(features, case_data, control_data):
        data.extend([(f, value, case_label_translated) for value in case])
        data.extend([(f, value, control_label_translated) for value in control])
    df = pd.DataFrame(data, columns=['Feature', 'Data', 'Group'])


    # Get the coolwarm palette
    cmap = sns.diverging_palette(220, 20, l=70, n=10)
    cases_color = cmap[-1]
    controls_color = cmap[0]

    # Plot boxplots with Seaborn
    sns.set_context("notebook", font_scale=1)
    plt.figure(figsize=(int(0.7 * len(features)), 8))
    ax = sns.boxplot(data=df, x='Feature', y='Data', hue='Group', palette={case_label_translated: cases_color, control_label_translated: controls_color}, linewidth=0.5, fliersize=5, width=0.6)

    # Add y-axis tick lines as horizontal lines
    for tick in ax.yaxis.get_majorticklocs():
      if tick >= 0:
        ax.axhline(y=tick, color='gray', linestyle='-', linewidth=0.3, zorder=0)

    # Add significance annotations
    if p_values:
        for i, p_value in enumerate(p_values):
            if p_value < 0.001:
                significance_level = '***'
            elif p_value < 0.01:
                significance_level = '**'
            elif p_value < 0.05:
                significance_level = '*'
            else:
                significance_level = 'ns'
            plt.text(i, -0.06, significance_level, fontsize=12, ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Vērtība')
    plt.xlabel('')
    plt.title(f'Būtiskākās pazīmju sadalījumu atšķirības:\n ({case_label_translated} pret {control_label_translated})\n', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add legend
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=cases_color, markersize=10, label=case_label_translated),
                      plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=controls_color, markersize=10, label=control_label_translated)]
    plt.legend(handles=legend_handles, loc='upper left')

    # Frame thickness
    plt.gca().spines['top'].set_linewidth(0.5)
    plt.gca().spines['right'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)

    # Remove tick lines
    plt.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    out_dir_name = IOHandler.get_absolute_path(f'../results/feature_analysis/comparisons/{obj.out_dir_name}', create_dir=True)
    out_file_path = os.path.join(out_dir_name, f'{case_label}_vs_{control_label}.png')
    plt.savefig(out_file_path, dpi=300)
    plt.close()


@timeit
@log_execution
@log_errors_and_warnings
def compare_distributions(obj: BioFlowMLClass, n_features = 15):

    
    target_column = obj.label_feature
    labels = obj.get_encoded_features()[target_column]
    labels_translated = translate_label_names(obj)
    control_label_encoded = labels.index(obj.control_label)
    
    for l in range(len(labels)):
        
        if l == control_label_encoded:
            continue
    
        df_copy = obj.df.copy()

        # Filter cases and controls
        df_copy = df_copy[df_copy[target_column].isin([l, control_label_encoded])]
        df_copy = df_copy.reset_index(drop=True)

        # Re-encode the target column: 0 for control, 1 for case
        df_copy[target_column] = df_copy[target_column].apply(lambda x: 0 if x == control_label_encoded else 1)
        
        # Perform Wilcoxon test
        significant_features, p_values = mann_whitney_association_test(df_copy, target_column)
        
        # Sort the p-values in ascending order and get the indices of the sorted p-values
        sorted_indices = np.argsort(p_values)

        # Reorder the feature names list based on the sorted indices of p-values
        significant_features_sorted = [significant_features[i] for i in sorted_indices]
        p_values_sorted = [p_values[i] for i in sorted_indices]

        controls = df_copy[(df_copy[target_column] == 0)]
        cases = df_copy[(df_copy[target_column] ==  1)]
        
        # Example data (replace with your actual data)
        control_data = []
        case_data = []

        for f in significant_features_sorted[:n_features]:
            # control_data.append(np.log(1 + controls[f]))
            # case_data.append(np.log(1 + cases[f]))
            control_data.append(controls[f])
            case_data.append(cases[f])

        plot_boxplots(obj, significant_features_sorted[:n_features], case_data, control_data, l, p_values_sorted[:n_features])
