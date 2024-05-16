# group_comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from src.utils.monitoring import timeit, log_errors_and_warnings


class FeatureDistributionComparator:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def compare(self):
        p_values = []
        for feature in self.features:
            group1 = feature[self.labels == 0]  # Assuming 0 represents controls
            group2 = feature[self.labels == 1]  # Assuming 1 represents cases
            _, p_value = mannwhitneyu(group1, group2)
            p_values.append(p_value)
        
        p_values = np.array(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_features = [self.features[i] for i in sorted_indices]
        sorted_p_values = p_values[sorted_indices]
        
        return sorted_features, sorted_p_values
    
    def plot_top_features(self, n=5):
        sorted_features, sorted_p_values = self.compare()
        top_features = sorted_features[:n]
        top_p_values = sorted_p_values[:n]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(n), top_p_values, color='skyblue')
        plt.yticks(range(n), [f"Feature {i+1}" for i in range(n)])
        plt.xlabel('Mann-Whitney U test p-value')
        plt.title('Top Features with Lowest p-values')
        plt.gca().invert_yaxis()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Assuming you have features and labels data
    features = np.random.randn(100, 10)  # Example features
    labels = np.random.randint(0, 2, 100)  # Example labels (0s and 1s)
    
    comparator = FeatureDistributionComparator(features, labels)
    comparator.plot_top_features(n=5)
