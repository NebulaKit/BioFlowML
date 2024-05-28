from src.utils.monitoring import timeit, log_errors_and_warnings, log_execution
from src.model_training import convert_dict_arrays_to_lists
from src.model_training import plot_all_classifier_results
from src.model_training import optimize_hyperparameters
from src.model_training import plot_feature_importances
from src.model_training import cross_validate_model
from src.model_training import get_classifiers
from src.model_training import save_json
from src.BioFlowMLClass import BioFlowMLClass
from src.utils.logger_setup import get_main_logger
from src.utils.IOHandler import IOHandler

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import statistics


@timeit
@log_execution
@log_errors_and_warnings
def train_binary_classifiers(obj: BioFlowMLClass):
    
    df = obj.df.copy()
    groups = obj.get_encoded_features()[obj.label_feature]
    control_encoding = groups.index(obj.control_label)
    target_column = obj.label_feature

    # Get classifier dictionary with params to optimize
    classifiers = get_classifiers()

    # Define the cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=11)

    # Preprocess feature names and drop features to be excluded (id columns etc.)
    df.columns = [col.replace('[', '_').replace(']', '_') for col in df.columns]
    df = df.drop(columns=obj.exclude_features)

    # Iterate through each case label
    for i, value in enumerate(groups):
        
        # Skip controls for case-control binary classifier training
        if groups[i] == "Control":
            continue
        
        logger = get_main_logger()
        logger.info(f'Generating classification models {value} vs {obj.control_label}')

        # Create the results directory if it doesn't exist
        output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary/{value.lower()}_vs_controls', create_dir=True)

        # Instantiate a DataFrame for collecting the cross-validation summary results
        df_results = pd.DataFrame(columns=['classifier', 'cv_aucs', 'avg_auc', 'models_trained', 'grid_search_time_s'])

        # Initialize dictionaries to store other metrics and feature importances
        metrics_all = {classifier_name: {'AUROC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'Specificity': []} for classifier_name in classifiers}
        feature_importances_all = {}

        filtered_df = df.copy()

        # Filter cases and controls
        filtered_df = filtered_df[filtered_df[target_column].isin([i, control_encoding])]
        filtered_df = filtered_df.reset_index(drop=True)

        # Re-encode the target column: 0 for control, 1 for case
        filtered_df[target_column] = filtered_df[target_column].apply(lambda x: 0 if x == control_encoding else 1)

        # Extract features and labels from the filtered binary-labeled samples
        X = filtered_df.drop(columns=[target_column])
        y = filtered_df[target_column]

        # Iterate over each classifier
        for classifier_name, (classifier, param_grid) in classifiers.items():

            # Perform grid search for best hyperparameters
            grid_search, models_trained_cnt, grid_search_time = optimize_hyperparameters(X, y, classifier, param_grid, cv, metric='f1')

            # Extract and save the best hyperparameters
            best_params = grid_search.best_params_
            clf_file_name = classifier_name.lower().replace(' ','_')
            out_class_name = value.lower().replace(' ','_')
            params_out_path = f'{output_dir}/params/{clf_file_name}_{out_class_name}_vs_controls_best_params.json'
            save_json(best_params, params_out_path)

            # Cross-validate
            metrics, feature_importances = cross_validate_model(X, y, cv, classifier, best_params, output_dir, classifier_name, obj, case_name=value)
            metrics_all[classifier_name] = metrics
            if feature_importances:
                feature_importances_all[classifier_name] = feature_importances

            # Log and save classifier results
            aurocs = metrics['AUROC']
            logger.info(f'{classifier_name} ({obj.out_dir_name}) ({value}) (avg AUC={round(statistics.mean(aurocs), 2)}) (Models trained={models_trained_cnt}) (Grid search time={round(grid_search_time, 2)}s)')
            df_results.loc[len(df_results)] = [classifier_name, aurocs, round(statistics.mean(aurocs), 2), models_trained_cnt, round(grid_search_time, 2)]

        # Plot summary results (cross-validation boxplots for all metrics and all classifiers)
        plot_all_classifier_results(metrics_all, output_dir, obj, value)

        # Save all cross-validation results
        metrics_all_out_path = f'{output_dir}/{value.lower()}_vs_controls_cv_metrics.json'
        convert_dict_arrays_to_lists(metrics_all)
        save_json(metrics_all, metrics_all_out_path)

        # Plot top 20 most significant features and their average weights per classifier
        plot_feature_importances(feature_importances_all, X.columns, output_dir, obj, value)

        # Save summary auroc results and training time as csv table
        df_results.to_csv(f'{output_dir}/{value.lower()}_vs_controls_cv_aurocs.csv', index=False)
