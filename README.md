# BioFlowML

BioFlowML is a machine learning toolkit designed for developing classification workflows and identifying biomarkers from labeled tabular omics data. It is implemented in Python 3.11.5 and interfaces with libraries such as scikit-learn, XGBoost, and seaborn. This allows users to explore and preprocess data, create binary and multiclass classifiers, perform feature selection, optimize hyperparameters, and conduct cross-validation. BioFlowML also offers tools for visualizing ROC curves and confusion matrices, comparing evaluation metrics, and evaluating feature importances across various classifiers.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

Make sure you have Python and pip installed on your system.

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies. Follow these steps to set up a virtual environment:

```bash
# Create a virtual environment (replace 'env' with your preferred name)
python -m venv env

# Activate the virtual environment
# On Windows
.\env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### Installing Dependencies

```bash
# Install all the necessary dependencies
pip install -r requirements.txt
```
This will install all the necessary dependencies for the project.

### Running Examples

The examples can be run using command line from the project root directory (BioFlowML).

```bash
# Run microbiome workflow
python .\examples\run_mb.py

# Run metadata workflow
python .\examples\run_meta.py
```

