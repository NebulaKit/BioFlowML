from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from src.BioFlowMLClass import BioFlowMLClass
import src.translate as tr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import time
import os


def visualize_splits(train_counts, test_counts, output_path, obj:BioFlowMLClass, case_name=None):
    
    groups = translate_label_names(obj)
    class_prefix = [get_translation(obj, f'cohort_prefix.train'),get_translation(obj, f'cohort_prefix.test')]

    if case_name:
        control_label = get_translation(obj, f'cohort.{obj.control_label}')
        groups = [control_label, case_name]

    # Convert counts to DataFrames for easier plotting
    train_df = pd.DataFrame(train_counts).fillna(0).astype(int)
    test_df = pd.DataFrame(test_counts).fillna(0).astype(int)

    # Combine train and test counts into a single DataFrame
    combined_df = pd.DataFrame()
    for fold in range(len(train_counts)):
        train_fold = train_df.iloc[fold].rename(lambda x: f"{class_prefix[0]} ({groups[x].lower()})")
        test_fold = test_df.iloc[fold].rename(lambda x: f"{class_prefix[1]} ({groups[x].lower()})")
        combined_fold = pd.concat([train_fold, test_fold], axis=0)
        combined_df = pd.concat([combined_df, combined_fold.rename(fold + 1)], axis=1)

    # Use a Seaborn color palette
    palette1 = sns.color_palette("Paired", n_colors=len(groups)*2)[1::2]
    palette2 = sns.color_palette("Paired", n_colors=len(groups)*2)[::2]
    palette = palette1 + palette2

    # Plotting
    ax = combined_df.T.plot(kind='bar', figsize=(10, 5), color=palette, alpha=0.7, width=0.85)
    plt.title(get_translation(obj, 'split_graph.title'))
    plt.xlabel(get_translation(obj, 'split_graph.xlabel'))
    plt.ylabel(get_translation(obj, 'split_graph.ylabel'))
    plt.xticks(rotation=0)
    t = get_translation(obj, 'split_graph.legend_title')
    plt.legend(title=t, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Make borders and ticks thinner
    plt.tick_params(axis='both', which='both', width=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def count_class_samples_by_indices(y, indices):
    unique, counts = np.unique(y[indices], return_counts=True)
    return dict(zip(unique, counts))

def optimize_hyperparameters(X, y, classifier, param_grid, cv, metric='f1_weighted'):
    start_time = time.time()  # Start time measurement

    # Calculate the total number of hyperparameter combinations
    total_param_combinations = 1
    for param_values in param_grid.values():
        total_param_combinations *= len(param_values)

    # Calculate the total number of models trained
    total_models_trained = total_param_combinations * cv.n_splits

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=cv,
        scoring=metric,
        n_jobs=-1,
        return_train_score=True
    )

    # Fit GridSearchCV
    grid_search.fit(X, y)

    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time  # Calculate elapsed time

    return grid_search, total_models_trained, elapsed_time

def translate_label_names(obj: BioFlowMLClass):
    labels = obj.get_encoded_features()[obj.label_feature].copy()
    translations = tr.load_translations(obj.lang)
    for i, l in enumerate(labels):
        labels[i] = tr.translate(f'cohort.{l}', translations)
    return labels

def get_translation(obj: BioFlowMLClass, label):
    translations = tr.load_translations(obj.lang)
    label_translated = tr.translate(label, translations)
    return label_translated

def get_translations(obj: BioFlowMLClass, label_list):
    translations = tr.load_translations(obj.lang)
    labels_translated = []
    for l in label_list:
        label_translated = tr.translate(l, translations)
        labels_translated.append(label_translated)
    return labels_translated
   
def cross_validate_model(X, y, cv, classifier, params, output_dir, classifier_name, obj:BioFlowMLClass, case_name=None):
    groups = translate_label_names(obj)
    case_name_translated = get_translation(obj, f'cohort.{case_name}') if case_name else None
    feature_importances_per_fold = []
    conf_matrices = []
    metrics = {'AUROC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    # For vizualizing the sample counts
    train_counts = []
    test_counts = []

    # Initialize lists to store true labels and predicted probabilities for each fold
    y_true_list = []
    y_proba_list = []

    for train_indices, test_indices in cv.split(X, y):
        train_counts.append(count_class_samples_by_indices(y, train_indices))
        test_counts.append(count_class_samples_by_indices(y, test_indices))

        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]

        # Instantiate the estimator with the best parameters from grid search
        clf = classifier
        clf = clf.set_params(**params)

        # Fit the model
        clf.fit(X_train, y_train)

        # Extract feature importances from each fold if available
        if hasattr(clf, 'feature_importances_'):
            feature_importances_per_fold.append(clf.feature_importances_)


        # Predict probabilities and calculate AUC
        y_proba = clf.predict_proba(X_test) if len(y.unique()) > 2 else clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_proba, multi_class='ovr') if len(y.unique()) > 2 else roc_auc_score(y_test, y_proba)

        # Sellect all y_test labels and predicted probabilities for
        # AUROC plotting
        y_true_list.append(y_test)
        y_proba_list.append(y_proba)

        # Get predictions and calculate other metrics
        y_pred = clf.predict(X_test)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate metrics for the current fold and classifier
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0) if len(y.unique()) > 2 else precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0) if len(y.unique()) > 2 else recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) if len(y.unique()) > 2 else f1_score(y_test, y_pred, zero_division=0)

        # Store metrics for the current classifier and fold
        metrics['AUROC'].append(round(auroc, 4))
        metrics['Accuracy'].append(round(accuracy, 4))
        metrics['Precision'].append(round(precision, 4))
        metrics['Recall'].append(round(recall, 4))
        metrics['F1 Score'].append(round(f1, 4))
        conf_matrices.append(conf_matrix)

    # Calculate average confussion matrix
    metrics['Confusion matrix'] = (sum(conf_matrices) / len(conf_matrices))
    # Plot confussion matrix
    if len(y.unique()) > 2:
        plot_confusion_matrix(metrics['Confusion matrix'], groups, output_dir, obj, classifier_name)
    else:
        
        classes = [groups[1], case_name_translated]
        plot_confusion_matrix(metrics['Confusion matrix'], classes, output_dir, obj, classifier_name)

    # Visualise sample counts in cv splits
    cv_split_file_name = f'{output_dir}/cv_splits.png'
    if case_name:
        case_name_out = case_name.lower().replace(' ','_')
        cv_split_file_name = f'{output_dir}/{case_name_out}_vs_controls_cv_splits.png'
        if not os.path.exists(cv_split_file_name):
            visualize_splits(train_counts, test_counts, cv_split_file_name, obj, case_name_translated)
    else:
        if not os.path.exists(cv_split_file_name):
            visualize_splits(train_counts, test_counts, cv_split_file_name, obj)

    # Concatenate true labels and predicted probabilities across all folds
    # And plot roc curves
    y_true_cv = np.concatenate(y_true_list)
    y_proba_cv = np.concatenate(y_proba_list)
    plot_roc_curves(y_true_cv, y_proba_cv, output_dir, classifier_name, case_name, obj)

    return metrics, feature_importances_per_fold

def plot_roc_curves(y_true_cv, y_proba_cv, output_dir, classifier_name, case_name, obj:BioFlowMLClass):

    groups = translate_label_names(obj)
    case_name_out = get_translation(obj, f'cohort.{case_name}') if case_name else ''
    classes_cnt = len(np.unique(y_true_cv))
    is_multiclass = classes_cnt > 2
    # Initialize lists to store AUROC values for each class
    auroc_list = []

    plt.figure(figsize=(8, 6))

    if is_multiclass:
        # Plot ROC curves for each class
        for i, count in enumerate(range(classes_cnt)):
            y_true = (y_true_cv == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true, y_proba_cv[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{groups[i]} (AUC = {roc_auc:.2f}) (n = {sum(y_true)})')
            auroc_list.append(roc_auc)
    else:
        fpr, tpr, _ = roc_curve(y_true_cv, y_proba_cv)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{case_name_out} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    plt.xlabel(get_translation(obj, 'roc_curve.xlabel'))
    plt.ylabel(get_translation(obj, 'roc_curve.ylabel'))
    clf_name = get_translation(obj, f'classifiers.{classifier_name}')
    control_name = get_translation(obj, f'cohort.{obj.control_label}').lower()
    case_name_out = case_name_out.lower()
    title = (
        f'{clf_name}:\n {get_translation(obj, f"roc_curve.title_multi")}' if is_multiclass else 
        f'{clf_name}:\n {get_translation(obj, f"roc_curve.title_binary")} '
        f'({case_name_out} {get_translation(obj, f"roc_curve.vs")} {control_name})'
    )
    plt.title(title)

    # Calculate and plot the average AUROC
    if auroc_list:
        average_auroc = np.mean(auroc_list)
        avg_name = get_translation(obj, 'roc_curve.avg')
        plt.axhline(y=average_auroc, color='r', linestyle='-',
                    label=f'{avg_name} AUC = {average_auroc:.2f}')

    plt.legend(loc="lower right")
    clf_name = classifier_name.lower().replace(' ','_')
    plt.tight_layout()
    case_name = case_name.lower() if case_name else None
    out_dir = f'{output_dir}/roc_curves/'
    os.makedirs(out_dir, exist_ok=True)
    fig_path = f'{out_dir}/{clf_name}_roc_curves.png' if is_multiclass else f'{out_dir}/{clf_name}_{case_name}_roc_curves.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_all_classifier_results(metrics: dict, output_dir, obj: BioFlowMLClass, case_name=None):
    # Create a new dictionary excluding the 'Confusion matrix'
    filtered_metrics = {
        (metric, get_translation(obj, f'classifiers.{classifier_name}')): values
        for classifier_name, metrics_dict in metrics.items()
        for metric, values in metrics_dict.items()
        if metric != 'Confusion matrix'
    }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(filtered_metrics)

    labels_to_translate = [
        'evaluation_metrics.auc',
        'evaluation_metrics.accuracy',
        'evaluation_metrics.precision',
        'evaluation_metrics.recall',
        'evaluation_metrics.f1'
    ]
    metric_names = get_translations(obj, labels_to_translate)
    metrics = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Set up subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6), sharey=True)
    sns.set_context("notebook", font_scale=1.2)
    colors = sns.color_palette("viridis", n_colors=10)

    # Loop through classifiers and create boxplots
    for i, metric in enumerate(metrics):
        with sns.plotting_context(font_scale=1.4):
            sns.boxplot(data=metrics_df[metric], ax=axes[i], orient='h', palette=colors)
        axes[i].set_title(metric_names[i], fontsize=10)

        # Thinner axis ticks
        axes[i].tick_params(axis='both', which='both', length=3, width=0.5)

        # Add lighter and thinner borders
        for spine in axes[i].spines.values():
            spine.set_linewidth(0.5)

    # Add title
    if case_name:
        control_name_translated = get_translation(obj, f'cohort.{obj.control_label}').lower()
        case_name_translated = get_translation(obj, f'cohort.{case_name}').lower()
        title = get_translation(obj, 'evaluation_metrics.title_binary')
        fig.suptitle(f"{title} ({case_name_translated} {get_translation(obj, 'roc_curve.vs')} {control_name_translated})", fontsize=16)
    else:
        title = get_translation(obj, 'evaluation_metrics.title_multi')
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cv_metrics.png', dpi=300)
    plt.close()

    
def save_json(d: dict, out_file_path):

    # Extract the directory path
    dir_path = os.path.dirname(out_file_path)

    if dir_path:
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

    # Save the best parameters to the file
    with open(out_file_path, 'w') as file:
        json.dump(d, file)

def convert_dict_arrays_to_lists(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            convert_dict_arrays_to_lists(value)

def plot_confusion_matrix(confusion_matrix, classes, output_dir, obj:BioFlowMLClass, classifier_name, cmap='Blues'):
    """
    Plot confusion matrix.

    Parameters:
        confusion_matrix (numpy.ndarray): Confusion matrix.
        classes (list): List of class labels.
        title (str): Title of the plot.
        cmap: Seaborn colormap.
    """
    translations = tr.load_translations(obj.lang)
    clf_name = tr.translate(f"classifiers.{classifier_name}", translations)
    conf_matrix_title = tr.translate(f"conf_matrix.title", translations)
    
    # Plot confusion matrix
    plt.figure(figsize=(len(classes)+3, len(classes)+3))
    sns.heatmap(confusion_matrix, annot=True, cmap=cmap, xticklabels=classes, yticklabels=classes)
    
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    
    plt.title(f'{clf_name}: \n{conf_matrix_title}')
    plt.xlabel(tr.translate(f"conf_matrix.xlabel", translations))
    plt.ylabel(tr.translate(f"conf_matrix.ylabel", translations))
    plt.tight_layout()
    
    # Save figure
    out_dir = f'{output_dir}/conf_matrices'
    os.makedirs(out_dir, exist_ok=True)
    title = classifier_name.lower().replace(' ','_')
    file_name = f'{out_dir}/{title}_confusion_matrix.png'
    plt.savefig(file_name, dpi=300)
    plt.close()

def plot_feature_importances(feature_importances:dict, feature_names, output_dir, obj: BioFlowMLClass, case_name=None):
    
    average_feature_importance = {get_translation(obj, f'classifiers.{classifier_name}'): np.mean(importance, axis=0) 
                                  for classifier_name, importance in feature_importances.items()}

    # Aggregate average feature importance across classifiers
    aggregate_feature_importance = np.zeros(len(feature_names))
    for importance in average_feature_importance.values():
        aggregate_feature_importance += importance

    # Sort feature importance and select top 20
    top_indices = np.argsort(aggregate_feature_importance)[::-1][:20]
    top_feature_names = [feature_names[i] for i in top_indices]
    top_feature_importance = np.array([average_feature_importance[classifier_name][top_indices] for classifier_name in average_feature_importance])

    # Create a DataFrame for visualization
    top_features_df = pd.DataFrame(top_feature_importance.T, columns=average_feature_importance.keys())
    top_features_df['Feature'] = top_feature_names
    top_features_df = top_features_df.set_index('Feature')

    # Define bar width and space between bars
    bar_width = 0.5
    bar_spacing = 0.2

    plt.figure(figsize=(15, 10))

    # Use a Seaborn color palette
    palette = sns.color_palette("viridis", n_colors=len(average_feature_importance))

    # Visualize top 20 most important features by each classifier
    ax = top_features_df.plot(kind='bar', stacked=True, color=palette, width=bar_width, legend=False, alpha=0.7)
    plt.xlabel('')
    plt.ylabel(get_translation(obj, 'feature_graph.ylabel'), fontsize=10)

    classification_translated = get_translation(obj, 'feature_graph.classification')
    if case_name:
        case_name_translated = get_translation(obj, f'cohort.{case_name}').lower()
        title = get_translation(obj, 'feature_graph.title_binary')
        plt.title(f'{title} "{case_name_translated}" {classification_translated}', fontsize=10)
    else:
        title = get_translation(obj, 'feature_graph.title_multi')
        plt.title(f'{title} {classification_translated}', fontsize=10)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    # Set xticks position with spacing between bars
    plt.xticks(np.arange(len(top_feature_names)) + (bar_width + bar_spacing) / 2, top_feature_names)

    legend_title = get_translation(obj, 'feature_graph.legend')
    plt.legend(title=legend_title, loc='upper right', fontsize=8, title_fontsize=8)
    ax.xaxis.grid(False)

    # Add thinner borders
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        
    # Thinner axis ticks
    ax.tick_params(axis='both', which='both', length=3, width=0.5)
    
    plt.tight_layout()

    plt.savefig(f'{output_dir}/top_20_features_plot.png', dpi=300)
    plt.close()



def get_classifiers():
    # Initialize classifiers with a dictionary of parameter grids for grid search
    classifiers = {
        # Linear models
        "Lasso Logistic Regression": (LogisticRegression(penalty='l1', solver='liblinear'), {'C': [0.01, 0.1, 1, 10]}),
        "Elastic Net Logistic Regression": (LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000), {'C': [0.01, 0.1, 1, 10]}),
        # Probabilistic models
        "Naive Bayes": (GaussianNB(), {}),
        # Instance-based models
        "K-Nearest Neighbors": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
        # Margin-based models
        "Support Vector Machines": (SVC(probability=True), {'C': [0.1, 1, 3]}),
        # Tree-based models
        "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [None, 5, 10, 20, 50, 60], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}),
        # Ensemble bagging models
        "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100, 200, 500]}),
        "Extra Trees": (ExtraTreesClassifier(), {'n_estimators': [50, 100, 200, 500], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 20, 50, 60]}),
        # Ensemble boosting models
        "XGBoost": (XGBClassifier(), {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100, 300]}),
        # Atrificial Neural Network models
        "Multi-layer Perceptron": (MLPClassifier(), {'hidden_layer_sizes': [(100,), (50, 100, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01]})
    }
    return classifiers
