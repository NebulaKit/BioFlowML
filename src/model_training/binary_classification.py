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

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
import pandas as pd
import statistics
import joblib
import json


@timeit
@log_execution
@log_errors_and_warnings
def classify_binary(obj: BioFlowMLClass, mode = 'pairwise'):
    """
    Parameters:
    - mode (str, optional): The binary classification mode. Options 'pairwise' for each class vs all others;
                            'ovr' for controls vs rest. Defaults to 'pairwise'.
    """
    
    logger = get_main_logger()
    groups = obj.get_labels()
    
    if mode == 'pairwise':
        logger.info('Pairwise binary classification starting...')
        for class_a, class_b in combinations(groups, 2):
            train_binary_classifiers(obj, class_a, class_b, mode)
            # TODO: think of what effect class encoding of 0 or 1 has on the end result since A vs B is treated the same as B vs A
            # Thresholds and metrics are computed with respect to class_1
    elif mode == 'ovr':
        logger.info('OvR binary classification starting...')
        if obj.control_label is None:
            logger.error('Control class not spedified in the BioFlowMLClass object!')
            return
        else:
            for value in map(str, groups):
                if value != obj.control_label:
                    train_binary_classifiers(obj, obj.control_label, value, mode)

    else:
        logger.error(f"Unsupported binary classification mode: '{mode}'. ")
        return

@log_errors_and_warnings
def train_binary_classifiers(obj: BioFlowMLClass, class_0: str, class_1: str, mode):
    
    logger = get_main_logger()
    logger.info(f'Generating binary classification models {class_0} vs {class_1}')
    
    target_name = obj.label_feature
    groups = obj.get_labels()
    class_0_code = groups.index(class_0)
    class_1_code = groups.index(class_1)
    
    # Define the cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=11)
    
    # Get classifier dictionary with params to optimize
    classifiers = get_classifiers()

    # Create the results directory if it doesn't exist
    output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary_{mode}/{class_0}_vs_{class_1}', create_dir=True)

    # Instantiate a DataFrame for collecting the cross-validation summary results
    df_results = pd.DataFrame(columns=['classifier', 'cv_aucs', 'avg_auc', 'models_trained', 'grid_search_time_s'])

    # Initialize dictionaries to store other metrics and feature importances
    metrics_all = {classifier_name: {'AUROC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'Specificity': []} for classifier_name in classifiers}
    feature_importances_all = {}

    df = obj.df.copy()
    
    # TODO: move this to preprocessing
    # Preprocess feature names and drop features to be excluded (id columns etc.)
    df.columns = [col.replace('[', '_').replace(']', '_') for col in df.columns]
    df = df.drop(columns=obj.exclude_features)

    # Filter classes from DataFrame
    df = df[df[target_name].isin([class_0_code, class_1_code])]
    df = df.reset_index(drop=True)
    
    # Re-encode the target column: 0 for control, 1 for case
    df[target_name] = df[target_name].apply(lambda x: 0 if x == class_0_code else 1)
    data_output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary_{mode}/{class_0}_vs_{class_1}/data', create_dir=True)
    
    # Extract features and labels from the filtered binary-labeled samples
    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    # Split data to train and test subsets and reset indices
    X_train, X_test, y_train, y_test = [
        df.reset_index(drop=True) for df in train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    ]

    # Save test and train datasets to CSV files
    data_output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary_{mode}/{class_0}_vs_{class_1}/data', create_dir=True)
    pd.concat([X_train, y_train], axis=1).to_csv(f'{data_output_dir}/train_set_{class_0}_vs_{class_1}.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f'{data_output_dir}/test_set_{class_0}_vs_{class_1}.csv', index=False)
    
    
    # Iterate over each classifier
    for classifier_name, (classifier, param_grid) in classifiers.items():
        
        clf_file_name = classifier_name.lower().replace(' ','_')
        
        # Perform Recursive feature elimination (RFECV) feature selection
        rfecv = RFECV(estimator=classifier, step=10, cv=cv, scoring='f1', verbose=1, n_jobs=-1)
        rfecv.fit(X_train, y_train)
        logger.info(f'Optimal number of features: {rfecv.n_features_}')
        
        # Select returned features
        selected_columns = X_train.columns[rfecv.support_]
        X_train_selected = X_train[selected_columns]
        X_test_selected = X_test[selected_columns]
        
        # Concatenate X and y for training and test sets
        train_df = pd.concat([X_train_selected, y_train], axis=1)
        test_df = pd.concat([X_test_selected, y_test], axis=1)

        # Save to CSV files
        data_output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary_{mode}/{class_0}_vs_{class_1}/data/{clf_file_name}', create_dir=True)
        train_df.to_csv(f'{data_output_dir}/{clf_file_name}_{class_0}_vs_{class_1}_train_set.csv', index=False)
        test_df.to_csv(f'{data_output_dir}/{clf_file_name}_{class_0}_vs_{class_1}_test_set.csv', index=False)

        # Perform grid search for best hyperparameters
        grid_search, models_trained_cnt, grid_search_time = optimize_hyperparameters(X_train_selected, y_train, classifier, param_grid, cv, metric='f1')

        # Extract and save the best hyperparameters
        best_params = grid_search.best_params_
        params_out_path = f'{output_dir}/hyperparameters/{clf_file_name}_{class_0}_vs_{class_1}.json'
        save_json(best_params, params_out_path)
    
        classifier.set_params(**best_params)

        # Cross-validate
        metrics, feature_importances = cross_validate_model(X_train_selected, y_train, cv, (classifier, classifier_name), output_dir, obj, class_0=class_0, class_1=class_1)
        metrics_all[classifier_name] = metrics
        if feature_importances:
            feature_importances_all[classifier_name] = feature_importances
  
        # Log and save classifier results
        aurocs = metrics['AUROC']
        logger.info(f'{classifier_name} ({obj.out_dir_name}) ({class_0} vs {class_1}) (avg AUC={round(statistics.mean(aurocs), 2)}) (Models trained={models_trained_cnt}) (Grid search time={round(grid_search_time, 2)}s)')
        df_results.loc[len(df_results)] = [classifier_name, aurocs, round(statistics.mean(aurocs), 2), models_trained_cnt, round(grid_search_time, 2)]
        
        # Train the final model
        final_model = classifier.set_params(**best_params)
        final_model.fit(X_train_selected, y_train)
        
        # Predict the target for the test set
        y_pred = final_model.predict(X_test_selected)
        y_proba = final_model.predict_proba(X_test_selected)[:, 1] 
        
        # Evaluate the performance using an appropriate metric (e.g., accuracy)
        # Calculate other metrics: accuracy, precision, recall, f1-score
        auc = roc_auc_score(y_test, y_proba) 
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.info(f'Final {classifier_name} Validation AUC: {auc}')
        
        # Save the final model
        models_output_dir = IOHandler.get_absolute_path(f'../results/model_training/{obj.out_dir_name}/binary_{mode}/{class_0}_vs_{class_1}/models', create_dir=True)
        model_out_path = f'{models_output_dir}/{clf_file_name}_{class_0}_vs_{class_1}.pkl'
        joblib.dump(final_model, model_out_path)
        
        n_features = final_model.n_features_in_
        feature_names = final_model.feature_names_in_.tolist() if hasattr(final_model, 'feature_names_in_') else None
        feature_imp = final_model.feature_importances_.tolist() if hasattr(final_model, 'feature_importances_') else None
        
        # Save the evaluation metrics
        metrics = {
            "Number of Features": n_features,
            "Selected Features": feature_names,
            "Feature Importances": feature_imp,
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion matrix": conf_matrix.tolist()
        }

        metrics_output_path = f'{models_output_dir}/{clf_file_name}_metrics.json'
        with open(metrics_output_path, 'w') as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
            
        if feature_names is not None:
            # Convert feature names to a DataFrame
            feature_names_df = pd.DataFrame(feature_names, columns=["ID"])
            
            # Save to CSV
            feature_names_csv_path = f"{models_output_dir}/{clf_file_name}_features.csv"
            feature_names_df.to_csv(feature_names_csv_path, index=False)


    # Plot summary results (cross-validation boxplots for all metrics and all classifiers)
    # TODO: adjust for the new binary classification options (pairwise and ovr)
    plot_all_classifier_results(metrics_all, output_dir, obj, class_1)

    # Save all cross-validation results
    metrics_all_out_path = f'{output_dir}/{class_0}_vs_{class_1}_cv_metrics.json'
    convert_dict_arrays_to_lists(metrics_all)
    save_json(metrics_all, metrics_all_out_path)

    # Plot top 20 most significant features and their average weights per classifier
    # plot_feature_importances(feature_importances_all, X_test_selected.columns, output_dir, obj, class_1)
    # TODO: cannot perform this way due to the fact that each classifier has different subest of selected features after RFECV

    # Save summary auroc results and training time as csv table
    df_results.to_csv(f'{output_dir}/{class_0}_vs_{class_1}_cv_aurocs.csv', index=False)
