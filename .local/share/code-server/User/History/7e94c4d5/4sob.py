"""
This file loads data from different data sources for pipelines
"""
import pandas as pandas

def load_csv_data(file_name):
    """
    :param file_path: the csv file path to the data source
    :return: Dataframe
    """
    df = pd.read_csv(file_name)
    return df

def load_from_json(file_name):
    """
    :param file_path: the json file path to the data source
    :return: Dataframe
    """
    df = pd.read_json(file_name)
    return df
