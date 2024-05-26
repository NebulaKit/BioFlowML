from src.utils.logger_setup import get_main_logger
import os


class IOHandler:
    @staticmethod
    def reset_logfile():
        root_dir = IOHandler.get_project_root_dir()
        log_path = os.path.join(root_dir, 'src/main.log')
        
        # Check if the file exists before attempting to delete it
        if os.path.exists(log_path):
            # Prompt the user
            delete_previous_log = input("Do you wish to delete the previous logfile (y/n)? ")
            
            if delete_previous_log.lower() == 'y':
                # Delete the file
                os.remove(log_path)
                print(f"The logfile {os.path.normpath(log_path)} has been deleted.")
    
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
                
    @staticmethod
    def create_directory(path):
        # Log the creation of the output folder
        if not os.path.exists(path):
            logger = get_main_logger()
            logger.debug(f'Output folder created at: {path}')
        
        # Check if the output folder exists and create it if it doesn't
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_absolute_path(relative_path, create_dir = False):
        try:
            # Get the absolute path of the current script
            utils_module_path = os.path.dirname(os.path.abspath(__file__))
            
            # Construct the absolute path to the desired file or directory
            abs_path = os.path.normpath(os.path.join(utils_module_path, '..', relative_path))

            # Check if the resulting path exists
            if not os.path.exists(abs_path):
                # Determine if the provided path is likely a file path (ends with an extension)
                is_file_path = '.' in os.path.basename(abs_path)
                
                # If create_dir is True and the provided path is a file path, create the parent directory
                if create_dir and is_file_path:
                    dir_path = os.path.dirname(abs_path)
                    IOHandler.create_directory(dir_path)
                # If create_dir is True and the provided path is a directory path, create the full directory structure
                elif create_dir and not is_file_path:
                    IOHandler.create_directory(abs_path)
                else:
                    raise FileNotFoundError(f"File or directory '{abs_path}' does not exist.")
            return abs_path
        except Exception as e:
            logger = get_main_logger()
            logger.error(f"Error occurred while getting data file path: {e}")
            return None
