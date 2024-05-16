
def serialize_list(lst):
    """
    Serialize a list into a comma-separated string.
    
    Parameters:
        lst (list): The list to serialize.
    
    Returns:
        str: The serialized string.
    """
    return '\n' + '\t' + ('\n' + '\t').join(map(str, lst))


def serialize_dict(dictionary):
    """
    Serialize a dictionary into a formatted string.
    
    Parameters:
        dictionary (dict): The dictionary to serialize.
    
    Returns:
        str: The serialized string.
    """
    serialized = "\n"
    for key, value in dictionary.items():
        serialized += f"\t{key}: \t{value}\n"
    return serialized