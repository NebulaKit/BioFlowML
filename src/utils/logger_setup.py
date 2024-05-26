import logging
import sys
import os

loggers = {}  # Dictionary to store initialized loggers

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Get the name of the parent module
        parent_module = os.path.basename(os.path.dirname(record.pathname))
        # Format the message with the parent module included
        record.module = f'{parent_module}/{record.module}' if parent_module else record.module
        # Call the default formatter to format the message
        return super().format(record)

def get_logger(logger_name, log_file_path=None):
    
    global loggers
    
    if not log_file_path:
        log_file_path = os.path.join(os.path.dirname(__file__), f'../../results/{logger_name}.log')
    
    if logger_name not in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create file handler
        formatter = CustomFormatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplication
        logger.handlers.clear()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        loggers[logger_name] = logger

    return loggers[logger_name]

def get_main_logger():
    return get_logger('main')
