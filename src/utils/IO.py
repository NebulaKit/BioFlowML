import src.utils.logger_setup as log
import os



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
                create_directory(dir_path)
            # If create_dir is True and the provided path is a directory path, create the full directory structure
            elif create_dir and not is_file_path:
                create_directory(abs_path)
            else:
                raise FileNotFoundError(f"File or directory '{abs_path}' does not exist.")


        return abs_path
    except Exception as e:
        logger = log.get_logger('main_log')
        logger.error(f"Error occurred while getting data file path: {e}")
        return None


def get_project_root_dir():
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))


def create_directory(path):
    
    # Check if the output folder exists and create it if it doesn't
    os.makedirs(path, exist_ok=True)

    # Log the creation of the output folder
    if not os.path.exists(path):
        logger = log.get_logger('main_log')
        logger.debug(f'Output folder created at: {path}')


def reset_logfile():
    
    root_dir = get_project_root_dir()
    log_path = os.path.join(root_dir, 'main.log')
    
    # Check if the file exists before attempting to delete it
    if os.path.exists(log_path):
        # Prompt the user
        delete_previous_log = input("Do you wish to delete the previous logfile (y/n)? ")
        
        if delete_previous_log.lower() == 'y':
            # Delete the file
            os.remove(log_path)
            print(f"The logfile {os.path.normpath(log_path)} has been deleted.")
