"""
Module for calculating and visualizing correlations between variables.
"""

import src.utils.logger_setup as log
from src.utils.monitoring import timeit, log_errors_and_warnings
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
from src.utils.IOHandler import IOHandler
import seaborn as sns
import src.translate as tr
import src.utils.microbiome as mb
import src.utils as utils
import os


class CorrelationAnalyzer:
    @log_errors_and_warnings
    def __init__(self, df: pd.DataFrame, out_dir_name, target_features, exclude_features=[], control_feature_name=None, lang='en'):
        self.df = df
        
        if not IOHandler.is_valid_directory_name(out_dir_name):
            raise ValueError("Invalid directory name (out_dir_name)!")
        self.out_dir_name = out_dir_name
        
        missing_target_features = [col for col in target_features if col not in self.df.columns]
        if missing_target_features:
            raise ValueError(f"The following target features are not present in the DataFrame: {', '.join(missing_target_features)}.")
        elif not target_features:
            raise ValueError("No target feature/-s provided!")
        self.target_features = target_features
        
        missing_exclude_features = [col for col in exclude_features if col not in self.df.columns]
        if missing_exclude_features:
            raise ValueError(f"The following features to be excluded are not present in the DataFrame: {', '.join(missing_exclude_features)}.")
        self.exclude_features = exclude_features
        
        if control_feature_name:
            if control_feature_name not in self.df.columns:
                raise ValueError(f"The control feature '{control_feature_name}' not the DataFrame!")
        self.control_feature_name = control_feature_name
        
        lang_json_files = IOHandler.get_json_files(f'{IOHandler.get_project_root_dir()}/translate')
        supported_langs = [os.path.splitext(file_name)[0] for file_name in lang_json_files]
        if lang not in supported_langs:
            raise ValueError(f"Unsupported language selected! Possible options: {supported_langs}")
        self.lang = lang
        
    @log_errors_and_warnings
    def plot_correlations_with_target(self, target_feature_name, top_n=15, method='spearman'):
        # Implementation of plot_correlations_with_target function
        print()
        
    # @timeit
    # def check_all_correlations(self):
    #     Parallel(n_jobs=-1)(
    #         delayed(plot_correlations_with_target)(
    #             df,
    #             target,
    #             target_features,
    #             exclude_features,
    #             control_feature_name, # if None then 'all-samples', othervise 'cases-vs-controls'
    #             lang = lang
    #         )
    #         for target in target_features
    #     )


# @log_errors_and_warnings
# def plot_correlations_with_target(df: pd.DataFrame, feature_set_name: str, target_feature_name: str, target_features: list = [],
#                                   exclude_features: list = [], control_feature_name: str = None, trim_mb_feature_names: bool = False, 
#                                   top_n: int = 15, method: str = 'spearman', lang: str = 'en'):
#     """
#     TODO: p-values un būtiskumi?
#     if control feature provided (control_feature_name) then correlation method 'cases-vs-controls' applied
#     when not ptovided, correlation method 'all-samples' applied
#     """
#     # In the case control name is provided and 
#     # 'cases-vs-controls'correlation type applied as a result
#     if control_feature_name:
#         # Skip if control-vs-control
#         if target_feature_name.lower() == control_feature_name.lower():
#             return
#         else:
#             # Filter only case and control samples
#             controls_df = df[df[control_feature_name] == 1]
#             cases_df = df[df[target_feature_name] == 1]
#             df = pd.concat([controls_df, cases_df], ignore_index=True)
    
#     logger = log.get_logger('main')
#     logger.debug(f'Plotting top {top_n} correlations for target: {target_feature_name}')
    
#     translations = tr.load_translations(lang)
    
#     # Prepare feature list for dropping
#     # For multi-class cases should drop other labels and leave only one target label 
#     # to avoid label correlationss
#     # For binary class cases targets list is empty
#     features_to_drop = target_features.copy()
#     if features_to_drop and target_feature_name in features_to_drop:
#         features_to_drop.remove(target_feature_name)
#     features_to_drop.extend(exclude_features)
    
#     # Filter the list of labels to drop only existing column names
#     confirmed_features_to_drop = [feature for feature in features_to_drop if feature in df.columns]
#     df = df.drop(columns=confirmed_features_to_drop)
    
#     # Output directory
#     out_dir_name = os.path.join(io.get_absolute_path('data/processed/correlations', create_dir=True), f'{feature_set_name}')
    
#     # Result logger
#     correlation_type = 'cases-vs-controls' if control_feature_name else 'all-samples'
#     out_log_path = os.path.join(out_dir_name, f'{target_feature_name}_correlations_top_{top_n}_{correlation_type}.log')
#     corr_logger = log.get_logger(f'correlation_analysis_{feature_set_name}_{target_feature_name}', out_log_path)
#     corr_logger.debug(f'Features excluded from correlation analysis: {utils.serialize_list(confirmed_features_to_drop)}')
    
#     # Compute correlations between features and target column
#     # Spearman does not assume linearity, making it more suitable for detecting non-linear associations between features
#     # Spearman correlation is less sensitive to outliers compared to Pearson correlation
#     # Spearman correlation does not assume that the data follows a normal distribution
#     correlation_with_target = df.corr(method='spearman')[target_feature_name]

#     # Sort correlation values by absolute value, strongest correlations at the top
#     sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

#     # Filter features with strongest correlations
#     top_correlations = sorted_correlation.head(top_n + 1)
#     top_feature_names = top_correlations.index.tolist()

#     # Compute the full correlation matrix based on selected features
#     filtered_correlation_matrix = df[top_feature_names].corr(method='spearman')
    
#     if trim_mb_feature_names:
#         short_column_names = {col: mb.trim_id(col) for col in top_feature_names}
#         top_feature_names = list(short_column_names.values())
#         # Log full and shortened feature names
#         corr_logger.debug(utils.serialize_dict(short_column_names))
    
#     # Translate the value associated with the first key (label)
#     if lang == 'lv':
#         top_feature_names[0] = tr.translate(f"cohort.{target_feature_name}", translations)
    
#     # Plot the correlation heatmap
#     plt.figure(figsize=(32, 25))  # Adjust size as needed
#     sns.set_context("notebook", font_scale=2)
#     cmap = sns.diverging_palette(220, 20, l=70, as_cmap=True)
#     sns.heatmap(filtered_correlation_matrix, cmap=cmap, annot=True, xticklabels=top_feature_names, yticklabels=top_feature_names)
#     translated_target_feature_name = tr.translate(f"cohort.{target_feature_name}", translations) if lang == "lv" else target_feature_name
#     plt.title(f'{tr.translate("correlation_heatmap.title", translations)} "{translated_target_feature_name}"', fontsize=30, pad=20)
#     plt.xticks(rotation=90, fontsize = 30)
#     plt.yticks(fontsize = 30)
#     plt.tight_layout()

#     # Save the plot
#     out_file_path = os.path.join(out_dir_name, f'{target_feature_name}_correlations_top_{top_n}_{feature_set_name}_spearman_{correlation_type}.png')
#     plt.savefig(out_file_path, dpi=250)
#     plt.close()


# @timeit
# def check_all_correlations(relative_data_path: str, target_features: list, exclude_features: list = None, control_feature_name: str = None, lang = 'en'):
    
#     feature_set_name = relative_data_path.split('/')[-1].split('.')[0]

#     logger = log.get_logger('main')
#     logger.debug('-' * 20 + f' STARTING CORRELATION ANALYSIS FOR {feature_set_name} FEATURES ' + '-' * 20)
    
#     # Read data
#     df = pd.read_csv(io.get_absolute_path(relative_data_path))
    
#     Parallel(n_jobs=-1)(
#         delayed(plot_correlations_with_target)(
#             df,
#             feature_set_name,
#             target,
#             target_features,
#             exclude_features,
#             control_feature_name, # if None then 'all-samples', othervise 'cases-vs-controls'
#             lang = lang
#         )
#         for target in target_features
#     )
        
#     logger.debug('-' * 20 + f' FINISHED CORRELATION ANALYSIS FOR {feature_set_name} FEATURES ' + '-' * 20)
