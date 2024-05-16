import src.utils.logger_setup as log
import json
import os



def load_translations(language):
    
    logger = log.get_logger('main_log')
    tr_path = os.path.join(os.path.dirname(__file__), f'{language}.json')
    try:
        with open(tr_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        return translations
    except FileNotFoundError:
        logger.warning(f"Translation file for '{language}' not found.")
        return None


def translate(label, translations):
    
    if translations:
        # Split the label by '.' to handle nested translations
        keys = label.split('.')
        current_translation = translations
        for key in keys:
            if key in current_translation:
                current_translation = current_translation[key]
            else:
                # If any part of the label is not found, return the original label
                return label
        return current_translation
    else:
        return label  # Return the original label if translations are not availablee
