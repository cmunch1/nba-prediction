import pandas as pd
import json
import hopsworks

from datetime import datetime, timedelta
from pathlib import Path  #for Windows/Linux compatibility

from constants import (
    FEATURE_GROUP_VERSION,
)


CONFIGS_PATH = Path.cwd() / "configs"
FEATURE_NAMES_FILE = CONFIGS_PATH / "feature_names.json" # dictionary of {lower case feature names : original mixed-case feature names}

def save_feature_names(df: pd.DataFrame) -> str:
    """
    Saves dictionary of {lower case feature names : original mixed-case feature names} to JSON file

    Args:
        df (pd.DataFrame): the dataframe with the features to be saved

    Returns:
        "File Saved."
    """

    # hopsworks "sanitizes" feature names by converting to all lowercase
    # this function saves the original so that they can be re-mapped later
    # for code re-usability
    
    original_f_names = df.columns.tolist()
    hopsworks_f_names = [x.lower() for x in original_f_names]

    # create a dictionary
    feature_mapper = {hopsworks_f_names[i]: original_f_names[i] for i in range(len(hopsworks_f_names))}

    with open(FEATURE_NAMES_FILE, "w") as fp:
        json.dump(feature_mapper, fp)
        
    return "File Saved."


def convert_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts hopsworks.ai lower-case feature names back to original mixed-case feature names that have been saved in JSON file

    Args:
        df (pd.DataFrame): the dataframe with features in all lower-case

    Returns:
        pd.DataFrame: the dataframe with features in original mixed-case
    """

    # hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    # converting these back to original format is needed for optimal code re-useability.


    # read in list of original feature names
    with open(FEATURE_NAMES_FILE, 'rb') as fp:
        feature_mapper = json.load(fp)
        
    df = df.rename(columns=feature_mapper)

    return df


def create_train_test_data(HOPSWORKS_API_KEY:str, STARTDATE:str, DAYS:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns train and test data from Hopsworks.ai feature store based upon how many DAYS back to use as test data

    Args:
        HOPSWORKS_API_KEY (str): subscription key for Hopsworks.ai
        STARTDATE (str): start date for train data, format YYYY-MM-DD
        DAYS (int): number of days back to use as test data, the train data will be all data except the last DAYS 
    
    Returns:
        Train and Test data as pandas dataframes
    """

    # log into hopsworks.ai and create a feature view object from the feature group
    # the api makes it easier to retrieve training/test data from a feature view than a feature group
    
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    rolling_stats_fg = fs.get_or_create_feature_group(
        name="rolling_stats",
        version=FEATURE_GROUP_VERSION,
    )

    query = rolling_stats_fg.select_all()

    feature_view = fs.create_feature_view(
        name = 'rolling_stats_fv',
        version = 2,
        query = query
    )

    # calculate the start and end dates for the train and test data and then retrieve the data from the feature view
    # the train data will be all data except the last DAYS

    
    TODAY = datetime.now()
    LASTYEAR = (TODAY - timedelta(days=DAYS)).strftime('%Y-%m-%d')
    TODAY = TODAY.strftime('%Y-%m-%d') 

    td_train, td_job = feature_view.create_training_data(
            start_time=STARTDATE,
            end_time=LASTYEAR,    
            description='All data except last ' + str(DAYS) + ' days',
            data_format="csv",
            coalesce=True,
            write_options={'wait_for_job': False},
        )

    td_test, td_job = feature_view.create_training_data(
            start_time=LASTYEAR,
            end_time=TODAY,    
            description='Last ' + str(DAYS) + ' days',
            data_format="csv",
            coalesce=True,
            write_options={'wait_for_job': False},
        )

    train = feature_view.get_training_data(td_train)[0]
    test = feature_view.get_training_data(td_test)[0]

    # hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    # converting these back to original format is needed for optimal code re-useability.
    
    train = convert_feature_names(train)
    test = convert_feature_names(test)

    # fix date format (truncate to YYYY-MM-DD)
    train["GAME_DATE_EST"] = train["GAME_DATE_EST"].str[:10]
    test["GAME_DATE_EST"] = test["GAME_DATE_EST"].str[:10]

    # feature view is no longer needed, so delete it
    feature_view.delete()

    return train, test

