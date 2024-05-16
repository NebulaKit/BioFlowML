from src.utils.logger_setup import get_main_logger
from time import time
from warnings import catch_warnings, simplefilter
from traceback import format_exc
import functools


def log_errors_and_warnings(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_main_logger()
        with catch_warnings(record=True) as captured_warnings:
            simplefilter("always")  # Ensure that all warnings are always raised
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # Log the exception
                logger.error("An error occurred: %s", e)
                # Log the stack trace
                logger.error(format_exc())
                result = None
            
            # Check if there are any warnings
            if captured_warnings:
                # Log the warnings and their stack traces
                logger.warning(f'{len(captured_warnings)} WARNINGS ENCOUNTERED')
                for warning in captured_warnings:
                    warning_message = str(warning.message) if warning.message else ""
                    logger.warning(f'Warning category: {warning.category.__name__}')
                    logger.warning(f'Warning message: {warning_message}')
                    logger.warning(f'Warning filename: {warning.filename}')
                    logger.warning(f'Warning function: {func.__name__}')
                    logger.warning(f'Warning line number: {warning.lineno}')

            return result
    return wrapper


def timeit(func):
    def wrapper(*args, **kwargs):
        logger = get_main_logger()
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time_seconds = end_time - start_time
        
        # Convert seconds to days, hours, minutes, and seconds
        days, remainder = divmod(execution_time_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format the execution time as "dd:hh:mm:ss"
        formatted_time = "{:02.0f}:{:02.0f}:{:02.0f}:{:02.0f}".format(days, hours, minutes, seconds)
        
        logger.debug(f'Execution time for {func.__name__}: {formatted_time}')
        return result
    return wrapper


