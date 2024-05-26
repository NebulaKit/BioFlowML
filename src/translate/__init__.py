from src.utils.logger_setup import get_main_logger
import json
import os


def load_translations(language):
    tr_path = os.path.join(os.path.dirname(__file__), f'{language}.json')
    try:
        with open(tr_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        return translations
    except FileNotFoundError:
        logger = get_main_logger()
        logger.warning(f"Translation file for '{language}' not found.")
        return None

def translate(label, translations):
    if translations:
        keys = label.split('.')
        current_translation = translations
        for key in keys:
            if key in current_translation:
                current_translation = current_translation[key]
            else:
                return None
        return current_translation
    else:
        return None
