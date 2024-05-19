import os

class IOHandler:
    @staticmethod
    def is_valid_directory_name(directory_name):
        """
        Validates a directory name.
        
        Args:
            directory_name (str): The name of the directory to validate.
        
        Returns:
            bool: True if the directory name is valid, False otherwise.
        """
        if not isinstance(directory_name, str):
            return False
        
        # Check if directory name contains any invalid characters
        invalid_chars = set('\\/:*?"<>|')
        if any(char in invalid_chars for char in directory_name):
            return False
        
        # Check if directory name is empty or consists of whitespace only
        if not directory_name.strip():
            return False
        
        # Check if directory name is valid on the current OS
        try:
            os.makedirs(os.path.join('.', directory_name), exist_ok=True)
        except OSError:
            return False
        else:
            os.removedirs(os.path.join('.', directory_name))
            return True
    
    @staticmethod    
    def get_json_files(directory):
        """
        Get the names of JSON files in a directory.
        
        Args:
            directory (str): The directory path.
        
        Returns:
            list: A list of JSON file names.
        """
        # List files in the directory
        files = os.listdir(directory)
        
        # Filter JSON files
        json_files = [file for file in files if file.endswith('.json')]
        
        return json_files
    
    @staticmethod
    def get_project_root_dir():
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../..'))

