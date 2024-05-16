from src.BioFlowMLClass import BioFlowMLClass
from src.utils.monitoring import log_errors_and_warnings
from src.utils.logger_setup import get_main_logger
from src.utils import serialize_list, serialize_dict
import pandas as pd


@log_errors_and_warnings
def transpose_otu_table(obj: BioFlowMLClass, samples_id=None, sort_samples_by_id=False) -> BioFlowMLClass:
    """
    Transpose the DataFrame within a BioFlowML object and optionally sort samples by their ID.

    Args:
        obj (BioFlowMLClass): The BioFlowMLClass object containing the DataFrame to be transposed.
        samples_id (str, optional): The name of the column containing sample IDs. Defaults to None.
        sort_samples_by_id (bool, optional): Whether to sort samples by their ID. Defaults to False.

    Returns:
        BioFlowMLClass: The BioFlowMLClass object with the transposed DataFrame.

    Raises:
        ValueError: This error might be raised internally if an error occurs during conversion of sample IDs to integers,
                but it is handled within the method, and the original object is returned without sorting.

    Example:
        Consider a BioFlowMLClass object 'obj' with the following DataFrame:

        +------------+-----+-----+-----+
        | #OTU ID    | OTU1| OTU2| OTU3|
        +------------+-----+-----+-----+
        | sample_1   | 10  | 20  | 30  |
        | sample_2   | 15  | 25  | 35  |
        +------------+-----+-----+-----+

        ```python
        transpose_otu_table(obj, samples_id='sample_id', sort_samples_by_id=True)
        ```
        +------------+----------+--------+
        | sample_id  | sample_1 |sample_2|
        +------------+----------+--------+
        | OTU1       | 10       | 15     |
        | OTU2       | 20       | 25     |
        | OTU3       | 30       | 35     |
        +------------+----------+--------+

        If 'sort_samples_by_id' is True, the resulting DataFrame will be sorted by the 'sample_id' column.
    """
    id_column = samples_id if samples_id else 'sample_id'
    
    # Set the first column as the index
    obj.df.set_index(obj.df.columns[0], inplace=True)

    # Transpose the DataFrame
    df_transposed = obj.df.transpose()

    # Reset the index to bring the previous column names into a new 'sample_id' column
    df_transposed.reset_index(drop=True, inplace=True)
    df_transposed.insert(0, id_column, obj.df.columns)
    obj.df = df_transposed

    if sort_samples_by_id:
        # Attempt to convert 'sample_id' values to integers
        try:
            df_transposed[id_column] = df_transposed[id_column].astype(int)
            # Sort the DataFrame by the 'sample_id' column
            df_sorted= df_transposed.sort_values(by=id_column)
            # Reset the index after sorting
            df_sorted.reset_index(drop=True, inplace=True)
            obj.df = df_sorted
        except ValueError as e:
            # Log the error
            logger = get_main_logger()
            logger.error(f"An error occurred while converting '{id_column}' values to integers: {e}")
            # Return the original object without sorting
            return obj
    
    logger = get_main_logger()
    logger.debug(f'OTU table DataFrame transposed!')
    obj.log_obj()
    return obj

@log_errors_and_warnings
def merge_with_metadata(obj: BioFlowMLClass, obj_meta: pd.DataFrame, features_to_add, id_column = 'sample_id'):
    """
    Merge the main data DataFrame with a metadata DataFrame based on a common identifier column.

    Parameters:
        obj (BioFlowMLClass): An instance of the BioFlowMLClass containing the main data DataFrame.
        obj_meta (pd.DataFrame): A DataFrame containing metadata to be merged with the main data.
        features_to_add (list): A list of feature names to add from the metadata DataFrame to the main data.
        id_column (str, optional): The name of the identifier column used for merging. Default is 'sample_id'.

    Returns:
        BioFlowMLClass: An instance of BioFlowMLClass with the main data DataFrame merged with the metadata DataFrame.

    Raises:
        ValueError: If some features to add are not present in the metadata DataFrame or if the identifier column
                    is not present in the metadata DataFrame.

    Note:
        This function modifies the DataFrame stored in the BioFlowMLClass object by merging it with the metadata.

    Example:
        ```python
        merge_with_metadata(obj, obj_meta, ['subject_group', 'gender'], id_column='subject_id')
        ```
    """
    missing_meta_features = [feature for feature in features_to_add if feature not in obj_meta.df.columns]
    if missing_meta_features:
        raise ValueError(f'Some features not present in the metadata DataFrame: {missing_meta_features}')
    
    if features_to_add:
        if id_column not in obj_meta.df.columns or id_column not in obj.df.columns:
            raise ValueError(f'{id_column} not present in the DataFrame\-s!')
        
        meta_filtered = obj_meta.df[[id_column] + features_to_add]
        obj.df = pd.merge(obj.df, meta_filtered, on=id_column)
    
    logger = get_main_logger()
    logger.debug(f'DataFrame merged with metadata features: {features_to_add}')
    obj.log_obj()
    return obj

@log_errors_and_warnings
def filter_unclassified_taxa(obj: BioFlowMLClass, level, aggregate=False):
    """
    Filter unclassified taxa from a taxonomic profile DataFrame at the specified taxonomic level.

    Parameters:
        obj (BioFlowMLClass): An instance of the BioFlowMLClass containing a DataFrame with taxonomic profile data.
        level (str): The taxonomic level to filter unclassified taxa. Should be one of 'd' (domain), 'p' (phylum),
                     'c' (class), 'o' (order), 'f' (family), 'g' (genus), or 's' (species).
        aggregate (bool, optional): Whether to aggregate unclassified taxa counts into a single column. Default is False.

    Returns:
        BioFlowMLClass: An instance of BioFlowMLClass with unclassified taxa filtered out from the DataFrame.

    Raises:
        ValueError: If the provided taxonomic level is not supported or if the taxonomic profile has not been
                    classified to the selected level.

    Note:
        This function modifies the DataFrame stored in the BioFlowMLClass object.

    Example:
        ```python
        # Assuming obj is an instance of BioFlowMLClass with a DataFrame containing taxa names
        filter_unclassified_taxa(obj, 'g', aggregate=True)
        ```      
    """
    level_indicators = ['d','p','c','o','f','g','s']
    level_names = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    if level not in level_indicators:
        raise ValueError(f"Provided taxonomic level '{level}' not supported! Available options: {level_indicators}")
    
    index_level = level_indicators.index(level)
    indication_str = level_indicators[index_level] + '__'
    prev_indication_str = level_indicators[index_level-1] + '__'
    
    if not any(indication_str in col for col in obj.df.columns):
        raise ValueError(f"The provided taxonomic profile has not been classified to the selected level ({level_names[index_level]})!")
    
    features_to_drop = []

    for c in obj.df.columns:
        if indication_str not in c and '__' in c:
            features_to_drop.append(c)
        else:
            level_name = c.split(indication_str)[-1]
            level_name = level_name.replace('_',' ')

            if level_name == 'uncultured':
                prev_level_name = c.split(indication_str)[0].split(prev_indication_str)[-1]
                prev_level_name = prev_level_name.replace('_',' ').replace(';','')
                if prev_level_name == 'uncultured':
                    features_to_drop.append(c)

    if aggregate:
      selected_columns = obj.df[features_to_drop]
      sum_column = selected_columns.sum(axis=1)
      obj.df = obj.df.drop(columns=features_to_drop)
      obj.df['unclassified'] = sum_column
    else:
      obj.df = obj.df.drop(columns=features_to_drop)
    
    logger = get_main_logger()
    logger.debug(f'Unclassified taxa ({level_names[index_level]} level) dropped from microbiome data:\n{serialize_list(features_to_drop)}')
    obj.log_obj()
    return obj

@log_errors_and_warnings
def trim_taxa_names(obj: BioFlowMLClass):
    """
    Trim and rename taxa names in the DataFrame associated with a BioFlowMLClass object.

    Args:
    - obj (BioFlowMLClass): An instance of BioFlowMLClass containing a DataFrame with taxa names.

    Returns:
    - BioFlowMLClass: The modified BioFlowMLClass object with trimmed and renamed taxa names.

    This method iterates through each column name in the DataFrame associated with the BioFlowMLClass object,
    splits the name by ';' to extract taxonomic levels, and keeps the last part of the name. If 'uncultured' is
    found, it incorporates the previous taxonomic level. Then, it logs the changes made and renames the columns
    in the DataFrame according to the modified names.

    Example:
        ```python
        # Assuming obj is an instance of BioFlowMLClass with a DataFrame containing taxa names
        obj = trim_taxa_names(obj)
        ```
    """
    new_column_names = {}
    
    for c in obj.df.columns:
      name_parts = c.split(';')
      last_part = name_parts[-1].split('__')[-1]
      if last_part != 'uncultured':
        new_column_names[c] = last_part
      else:
        previous_part = name_parts[-2].split('__')[-1]
        new_column_names[c] = f'{previous_part} {last_part}'
    
    # Find repeating values
    repeating_values = {}
    for key, value in new_column_names.items():
        if list(new_column_names.values()).count(value) > 1:
            repeating_values[key] = value
    
    continue_renaming = 'y'
    if repeating_values:
        continue_renaming = input(f'Some trimmed values are not unique possibly due to unclassified taxta:\n{serialize_dict(repeating_values)}\nSome columns with non-unique names will be omitted from the DataFrame! Continue? (y/n)')
    
    if continue_renaming.lower() == 'y':
        obj.df.rename(columns=new_column_names, inplace=True)
        
        # Drop columns with empty column names
        obj.df = obj.df.loc[:, [col for col in obj.df.columns if col != '']]
        
        # Drop columns with non unique names
        non_unique_columns = obj.df.columns[obj.df.columns.duplicated()]
        obj.df = obj.df.drop(non_unique_columns, axis=1)
        
        log_str = ''
        for i, n in enumerate(new_column_names.values()):
            try:
                log_str += f"{n}: \t\t\t\t{obj.df.columns[i]}\n"
            except IndexError:
                continue
        
        logger = get_main_logger()
        logger.debug(f'Taxa names trimmed and changed:\n{log_str}')
        obj.log_obj()
    return obj