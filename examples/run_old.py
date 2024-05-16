import sys
import os

# Add the parent directory of the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

import pandas as pd
import src.utils.IO as io
import src.preprocessing.microbiome_data_processing as mbdp
import src.preprocessing.metadata_processing as metaproc
import src.utils.logger_setup as log
import src.exploratory_data_analysis as eda
from src.exploratory_data_analysis.distributions import check_all_distributions, plot_transformations
from src.exploratory_data_analysis.correlations import CorrelationAnalyzer
from src.utils.monitoring import timeit
import src.preprocessing as pp
import os
from src.BioFlowMLClass import BioFlowMLClass

# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
# for train_indices, test_indices in split.split(df_meta, df_meta[['Liver_disease', 'Control', 'Vegan', 'Celiac', 'Vegetarian']]):
#     strat_train_set = df_meta.loc[train_indices]
#     strat_test_set = df_meta.loc[test_indices]
    
# X_train = strat_train_set.drop(["Liver_disease"], axis=1)
# y_train = strat_train_set["Liver_disease"]
# X_train_processed = pipeline.fit_transform(X_train)
# print(f'X_train_processed: {X_train_processed}')   

@timeit
def test_meta_preprocessing(path: str, delimiter: str = ','):
    exclude_features = ['sample_id']
    # 1. Get data
    df_meta = pd.read_csv(io.get_data_file_path(path), sep=delimiter)
    df_meta.to_csv(io.get_data_file_path("data/processed/metadata_original.csv"), index=False)
    
    # 2. Preprocess metadata (transform categorical features)
    pipeline = metaproc.create_preprocessing_pipeline()
    df_meta = pipeline.fit_transform(df_meta)
    df_meta.reset_index(drop=True, inplace=True)
    df_meta.to_csv(io.get_data_file_path("data/processed/metadata.csv"), index=False)
    
    # 3. Filter features
    df_meta_filtered = pp.apply_zeros_filter(df_meta, exclude_features = exclude_features)
    df_meta_filtered.to_csv(io.get_data_file_path("data/processed/metadata_filtered.csv"), index=False)

    # 3. Normalize and scale (both filtered and full feature matrix)
    pipeline = pp.get_numerical_feature_pipeline(df_meta, norm_method = 'yeo-johnson', scaler_type = 'robust', exclude_features = exclude_features)
    meta_feature_matrix = pipeline.fit_transform(df_meta)
    df_meta = pd.DataFrame(meta_feature_matrix, columns=df_meta.columns)
    
    pipeline = pp.get_numerical_feature_pipeline(df_meta_filtered, norm_method = 'yeo-johnson', scaler_type = 'robust', exclude_features = exclude_features)
    meta_filtered_feature_matrix = pipeline.fit_transform(df_meta_filtered)
    df_meta_filtered = pd.DataFrame(meta_filtered_feature_matrix, columns=df_meta_filtered.columns)
    
    # 3. Save as csv (both filtered and full feature matrix)
    out_path = "data/processed/metadata_normalized_scaled.csv"
    df_meta.to_csv(io.get_data_file_path(out_path), index=False)
    df_meta_filtered.to_csv(io.get_data_file_path("data/processed/metadata_filtered_normalized_scaled.csv"), index=False)
    return out_path

@timeit
def test_mb_preprocessing(path_mb: str, path_meta: str, label_features: list, merge_on: str = 'sample_id', delimiter_mb: str = ','):
    # 1. Get microbiome data
    otu_table = pd.read_csv(io.get_data_file_path(path_mb), sep=delimiter_mb, index_col=0)
    
    # 2. Preprocess OTU table (transpose and sort)
    pipeline = mbdp.create_otu_table_transpose_pipeline()
    df_mb = pipeline.fit_transform(otu_table)
    df_mb.to_csv(io.get_data_file_path("data/processed/microbiome_data_transposed.csv"), index=False)
    
    # 3. Merge with metadata and add target labels based on sample_id column
    df_meta = pd.read_csv(io.get_data_file_path(path_meta))
    df_merged_mb = mbdp.merge_mb_to_targets(df_mb, df_meta, label_features, merge_on)
    df_merged_mb.to_csv(io.get_data_file_path("data/processed/microbiome_data.csv"), index=False)
    
    # 4. Apply minimum abundance and missingness filters
    exclude_features = label_features.copy()
    exclude_features.insert(0, merge_on)
    df_mb_filtered = mbdp.apply_minimum_abundance_filter(df_merged_mb, exclude_features)
    df_mb_filtered = pp.apply_zeros_filter(df_mb_filtered, exclude_features)
    
    # 5. Add alpha diversity features (both filtered and full feature matrix)
    df_mb_filtered = mbdp.add_shannon_alpha_diversity_feature(df_mb_filtered, exclude_features)
    df_mb_filtered = mbdp.add_simple_alpha_diversity_feature(df_mb_filtered, exclude_features)
    df_mb_filtered.to_csv(io.get_data_file_path("data/processed/microbiome_data_filtered.csv"), index=False)
    exclude_features = label_features.copy()
    exclude_features.insert(0, merge_on)
    df_merged_mb = mbdp.add_shannon_alpha_diversity_feature(df_merged_mb, exclude_features)
    df_merged_mb = mbdp.add_simple_alpha_diversity_feature(df_merged_mb, exclude_features)
    ''' TODO: can also potentially add:
    Evenness: Evenness measures how evenly abundant the different taxa are within a sample. It can be calculated using various metrics such as Pielou's evenness index or Simpson's evenness index.
    Dominance: Dominance measures the relative abundance of the most abundant taxa in a sample compared to the rest. It can provide insights into the degree of dominance of certain taxa within a community.
    Taxonomic Diversity: In addition to alpha diversity measures like Shannon entropy, you can also consider incorporating other diversity indices such as Simpson's diversity index, Inverse Simpson's diversity index, or the Berger-Parker index.
    '''
    
    # 6. Normalize and scale (both filtered and full feature matrix)
    pipeline = pp.get_numerical_feature_pipeline(df_mb_filtered, norm_method = 'yeo-johnson', scaler_type = 'robust', exclude_features = exclude_features)
    mb_filtered_feature_matrix = pipeline.fit_transform(df_mb_filtered)
    df_mb_filtered = pd.DataFrame(mb_filtered_feature_matrix, columns=df_mb_filtered.columns)
    
    pipeline = pp.get_numerical_feature_pipeline(df_merged_mb, norm_method = 'yeo-johnson', scaler_type = 'robust', exclude_features = exclude_features)
    mb_feature_matrix = pipeline.fit_transform(df_merged_mb)
    df_mb = pd.DataFrame(mb_feature_matrix, columns=df_merged_mb.columns)
    
    # 7. Save as csv (both filtered and full feature matrix)
    out_path = "data/processed/microbiome_data_normalized_scaled.csv"
    df_mb.to_csv(io.get_data_file_path(out_path), index=False)
    df_mb_filtered.to_csv(io.get_data_file_path("data/processed/microbiome_data_filtered_normalized_scaled.csv"), index=False)
    return out_path


def test_EDA_distributions():
    
    feature_files = ['data/processed/metadata_original.csv']

    for relative_data_path in feature_files:
        check_all_distributions(relative_data_path,'lv')
        
        

# def test_EDA_correlations():
    
#     feature_files = ['data/processed/microbiome_data_v2_norm.csv']
#                     # 'data/processed/microbiome_data_v2.csv',
#                     #  'data/processed/metadata.csv']

#     target_features = ['Liver_disease', 'Control', 'Vegan', 'Celiac', 'Vegetarian']
#     control_feature_name = target_features[1]
#     exclude_features = ['sample_id']
    
#     # 'one-to-many' correlations
#     for relative_data_path in feature_files:
#         check_all_correlations(relative_data_path, target_features, exclude_features, lang = 'lv')
    
#     # 'case-vs-control' correlations
#     for relative_data_path in feature_files:
#         check_all_correlations(relative_data_path, target_features, exclude_features, control_feature_name, lang = 'lv')


def main():
    
    print('This is an old script')
    # io.reset_logfile()
    
    
    # logger = log.get_logger('main_log')
    # eda.log_descriptive_stats(path="data/raw/nomissRepresentative-2021-06-28.tsv", delimiter='\t', id_field = 'sample_id')
    # path_meta = test_meta_preprocessing(path="data/raw/nomissRepresentative-2021-06-28.tsv", delimiter='\t')
    # logger.debug('Metadata preprocessing done!')
    # logger.debug(f'Output at: {path_meta}')
    
    
    # TODO: handle as label feature and then one-hot encode as label_A, label_B etc., but display as A, B in figures
    # labels = ['Liver_disease', 'Control', 'Vegan', 'Celiac', 'Vegetarian']
    # eda.log_descriptive_stats(path="data/processed/microbiome_data_transposed.csv", id_field = 'sample_id')
    # path_meta = 'data/processed/metadata.csv'
    # path_mb = test_mb_preprocessing(path_mb="data/raw/features-taxlevel-6.tsv", path_meta=path_meta, label_features=labels, delimiter_mb='\t')
    # logger.debug('Microbiome data preprocessing done!')
    # logger.debug(f'Output at: {path_mb}')
    
    
    # Test eda distributions for all
    # test_EDA_distributions()
    
    # Test eda distributions for one feature
    # df = pd.read_csv(io.get_absolute_path('data/processed/metadata.csv'))
    # norm_result, is_numerical, feature_name = plot_transformations(df, 'metadata', 'daily_energy_kcal', log_transform_type = 1, color = '#7318f2', lang = 'lv')
    # print(f'feature_name: {feature_name}')
    # print(f'norm_result: {norm_result}')
    # print(f'is_numerical: {is_numerical}')
    
    # test_EDA_correlations()
    
    # Tests on test_datasets
    # eda.log_descriptive_stats(path="data/test_datasets/blood_samples.csv")
    # eda.log_descriptive_stats(path="data/test_datasets/breast_cancer.csv")
    # eda.log_descriptive_stats(path="data/test_datasets/diabetes.csv")
    
    # corrs.main(f'data/test_datasets/blood_samples.csv', 'blood_samples', labels, exclude_features, 'lv')
    


if __name__ == "__main__":
    main()