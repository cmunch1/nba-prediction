import pandas as pd
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

def save_feature_names(df: pd.DataFrame)-> pd.DataFrame:

    '''
    Saves original feature name mixed-case formatting to JSON file
    '''
    # hopsworks "sanitizes" feature names by converting to all lowercase
    # this function saves the original so that they can be re-mapped later
    # for code re-usability
    
    feature_names = df.columns.tolist()
    with open("feature_names.json", "w") as fp:
        json.dump(feature_names, fp)
        
    return "File Saved."