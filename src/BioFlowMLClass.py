from src.utils.monitoring import log_errors_and_warnings
from src.utils.IOHandler import IOHandler
from src.utils.logger_setup import get_main_logger
from src.utils import serialize_dict
import pandas as pd
import os


class BioFlowMLClass:
    """
    BioFlowML is a class for managing machine learning workflows in bioinformatics.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the data for analysis.
        label_feature (str): The name of the column in df representing the label feature.
        out_dir_name (str): The name of the output directory for saving results.
        exclude_features (list, optional): A list of features to be excluded from analysis 
            (e.g. id columns) but not dropped from the DataFrame. Defaults to None.
        control_label (str, optional): The name of the control value (from the label_feature values)
            for cases vs. controls analysis. Defaults to None.
        drop_features (list, optional): A lsit of features to be removed from the pandas DataFrame.
            Defaults to None.
        lang (str, optional): The language used in figures and plots. Defaults to 'en'.
    
    Methods:
        __init__(df: pd.DataFrame, label_feature: str, out_dir_name: str, 
                 exclude_features: list = None, control_feature: str = None, lang: str = 'en'):
            Initializes a BioFlowML object with the provided parameters.
            Raises ValueError if input parameters are invalid.
        to_dict(): Serialize the BioFlowML object into a dictionary for logging purposes.
        # TODO: extent the documentation
    """
    @log_errors_and_warnings
    def __init__(self, df: pd.DataFrame, out_dir_name: str, label_feature: str=None, exclude_features: list=None, control_label: str=None, drop_features: list=None, lang: str='en'):
        # Check if df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The 'df' parameter must be a pandas DataFrame!")
        self.df = df
        
        # Validate out directory name
        if not IOHandler.is_valid_directory_name(out_dir_name):
            raise ValueError("Invalid directory name (out_dir_name)!")
        self.out_dir_name = out_dir_name
        
        # Check if label feature is contained within the df
        if label_feature:
            if label_feature not in self.df.columns:
                raise ValueError(f"The provided label feature is not present in the DataFrame!")
        self.label_feature = label_feature
        
        # Check if provided features to be excluded from further analysis are contained within the df
        if exclude_features:
            missing_exclude_features = [col for col in exclude_features if col not in self.df.columns]
            if missing_exclude_features:
                raise ValueError(f"The following features to be excluded are not present in the DataFrame: {', '.join(missing_exclude_features)}.")
        self.exclude_features = exclude_features if exclude_features else []
        
        # Check if provided control label is present in label_feature
        if control_label:
            if control_label not in df[self.label_feature].values:
                raise ValueError(f"The label feature '{label_feature}' does not contain the provided control label '{control_label}'!")
        self.control_label = control_label
        
        # Check if provided features to be dropped are contained within the df
        if drop_features:
            missing_drop_features = [col for col in drop_features if col not in self.df.columns]
            if missing_drop_features:
                raise ValueError(f"The following features to be dropped are not present in the DataFrame: {', '.join(missing_exclude_features)}.")
        self.drop_features = drop_features if drop_features else []
        self.df = self.df.drop(columns=self.drop_features)
        
        # Check if provided lanuage is available in resources
        lang_json_files = IOHandler.get_json_files(f'{IOHandler.get_project_root_dir()}/src/translate')
        supported_langs = [os.path.splitext(file_name)[0] for file_name in lang_json_files]
        if lang not in supported_langs:
            raise ValueError(f"Unsupported language selected! Possible options: {supported_langs}")
        self.lang = lang
        
        self.log_obj()
        
        # Internal properties
        self._encoded_features = {}

        
    # Setter methods for internal properties
    def set_encoded_features(self, key, value):
        self._encoded_features[key] = value
        
    def set_label_feature(self, label_feature, control_label=None):
        
        if label_feature not in self.df.columns:
                raise ValueError(f"The provided label feature is not present in the DataFrame!")
        
        if control_label:
            if control_label not in self.df[label_feature].values:
                raise ValueError(f"The label feature '{label_feature}' does not contain the provided control label '{control_label}'!")
            
        self.label_feature = label_feature
        
        if control_label:
            self.control_label = control_label
    
    # Getter methods for internal properties
    def get_encoded_features(self):
        return self._encoded_features


    def log_obj(self):
        """
        Log the serialized BioFlowML object using the main logger
        """
        logger = get_main_logger()
        logger.debug(serialize_dict(self.to_dict()))
        
    def to_dict(self) -> dict:
        """
        Serialize the BioFlowML object into a dictionary for logging purposes.

        Returns:
            dict: A dictionary representation of the BioFlowML object.
        """
        bioflowml_dict = {
            'df': serialize_dict(self.df.head().to_dict()),
            'label_feature': self.label_feature,
            'out_dir_name': self.out_dir_name,
            'exclude_features': self.exclude_features,
            'control_label': self.control_label,
            'lang': self.lang
        }
        return bioflowml_dict
        
