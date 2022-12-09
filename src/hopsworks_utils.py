import pandas as pd
import numpy as np
import json

def convert_feature_names(df: pd.DataFrame)-> pd.DataFrame:

    '''
    Converts hopsworks.ai feature names back to original feature names that are saved in JSON file
    '''

    # hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    # converting these back to original format is needed for optimal code re-useability.

    hopsworks_f_names = df.columns.to_list()

    # read in list of original feature names
    with open('feature_names.json', 'rb') as fp:
        original_f_names = json.load(fp)
        
    # create a dictionary
    feature_mapper = {hopsworks_f_names[i]: original_f_names[i] for i in range(len(hopsworks_f_names))}

    df = df.rename(columns=feature_mapper)

    return df