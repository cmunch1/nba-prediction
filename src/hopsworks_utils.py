import pandas as pd
import json
import hopsworks

from datetime import datetime, timedelta

def convert_feature_names(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Converts hopsworks.ai lower-case feature names back to original mixed-case feature names that are saved in JSON file
    '''

    # hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    # converting these back to original format is needed for optimal code re-useability.

    hopsworks_f_names = df.columns.to_list()

    # read in list of original feature names
    with open('feature_names.json', 'rb') as fp:
        feature_mapper = json.load(fp)
        
    df = df.rename(columns=feature_mapper)

    return df



def save_feature_names(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Saves dictionary of {lower case feature names : original mixed-case feature names} to JSON file
    '''
    # hopsworks "sanitizes" feature names by converting to all lowercase
    # this function saves the original so that they can be re-mapped later
    # for code re-usability
    
    original_f_names = df.columns.tolist()
    hopsworks_f_names = [x.lower() for x in original_f_names]

    # create a dictionary
    feature_mapper = {hopsworks_f_names[i]: original_f_names[i] for i in range(len(hopsworks_f_names))}

    with open("feature_names.json", "w") as fp:
        json.dump(feature_mapper, fp)
        
    return "File Saved."



def create_train_test_data(HOPSWORKS_API_KEY:str, DAYS:int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates and returns train and test data from Hopsworks.ai feature store

    Args:
        HOPSWORKS_API_KEY (str): subscription key for Hopsworks.ai
        DAYS (int): number of days back to use as test data
    
    Returns:
        Train and Test data as pandas dataframes
    """

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    rolling_stats_fg = fs.get_or_create_feature_group(
    name="rolling_stats",
    version=1,
    )

    query = rolling_stats_fg.select_all()

    feature_view = fs.create_feature_view(
        name = 'rolling_stats_fv',
        version = 1,
        query = query
    )

    STARTDATE = "2003-01-01" #data goes back to 2003 season
    TODAY = datetime.now()
    LASTYEAR = (TODAY - timedelta(days=DAYS)).strftime('%Y-%m-%d')
    TODAY = TODAY.strftime('%Y-%m-%d') 

    td_train, td_job = feature_view.create_training_data(
            start_time=STARTDATE,
            end_time=LASTYEAR,    
            description='All data except last 365 days',
            data_format="csv",
            coalesce=True,
            write_options={'wait_for_job': False},
        )

    td_test, td_job = feature_view.create_training_data(
            start_time=LASTYEAR,
            end_time=TODAY,    
            description='Last365 days',
            data_format="csv",
            coalesce=True,
            write_options={'wait_for_job': False},
        )

    train = feature_view.get_training_data(td_train)[0]
    test = feature_view.get_training_data(td_test)[0]

    #hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    train = convert_feature_names(train)
    test = convert_feature_names(test)

    #fix date format
    train["GAME_DATE_EST"] = train["GAME_DATE_EST"].str[:10]
    test["GAME_DATE_EST"] = test["GAME_DATE_EST"].str[:10]

    return train, test


def get_train_test_data(HOPSWORKS_API_KEY:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns train and test data from Hopsworks.ai feature store
    
    This is the data that was used to train the current model saved in the Model Registry at Hopsworks.ai

    Args:
        HOPSWORKS_API_KEY (str): subscription key for Hopsworks.ai
    
    Returns:
        Train and Test data as pandas dataframes
    """

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    rolling_stats_fg = fs.get_feature_group(
        name="rolling_stats",
        version=1,
    )

    feature_view = fs.get_feature_view(
        name = 'rolling_stats_fv',
        version = 1, #version 1 will always be current model; old versions are deleted when new model is trained
    )

    # get_training_data retrieves features and labels, but these were not separated when the training data was created
    # adding [0] to the end of the function call returns only the features
    train = feature_view.get_training_data(training_dataset_version=1)[0] #training_dataset_version=1 is the training dataset created
    test = feature_view.get_training_data(training_dataset_version=2)[0] #training_dataset_version=2 is the test dataset created

    #hopsworks converts all feature names to lower-case, while the original feature names use mixed-case
    train = convert_feature_names(train)
    test = convert_feature_names(test)

    #fix date format
    train["GAME_DATE_EST"] = train["GAME_DATE_EST"].str[:10]
    test["GAME_DATE_EST"] = test["GAME_DATE_EST"].str[:10]


    return train, test




