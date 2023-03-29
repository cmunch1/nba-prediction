
import pandas as pd

def process_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning on the games dataset.

    Args:
        games (pd.DataFrame): the raw games dataframe

    Returns:
        the cleaned games dataframe

    """
        
   
    # remove preseason games (GAME_ID begins with a 1)
    games = games[games['GAME_ID'] > 20000000]

    # flag postseason games (GAME_ID begins with >2)
    games['PLAYOFF'] = (games['GAME_ID'] >= 30000000).astype('int8')

    # remove duplicates (each GAME_ID should be unique)
    games = games[~games.duplicated(subset=['GAME_ID'])]

    # drop unnecessary fields
    all_columns = games.columns.tolist()
    drop_columns = ['GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away']
    use_columns = [item for item in all_columns if item not in drop_columns]
    games = games[use_columns]
    
    return games


def process_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning on the ranking dataset.
    
    Args:  
        ranking (pd.DataFrame): the raw ranking dataframe

    Returns:
        the cleaned ranking dataframe
      
    """


    # remove preseason rankings (SEASON_ID begins with 1)
    ranking = ranking[ranking['SEASON_ID'] > 20000]

    # convert home record and road record to numeric
    ranking['HOME_W'] = ranking['HOME_RECORD'].apply(lambda x: x.split('-')[0]).astype('int')
    ranking['HOME_L'] = ranking['HOME_RECORD'].apply(lambda x: x.split('-')[1]).astype('int')
    ranking['HOME_W_PCT'] = ranking['HOME_W'] / ( ranking['HOME_W'] + ranking['HOME_L'] )

    ranking['ROAD_W'] = ranking['ROAD_RECORD'].apply(lambda x: x.split('-')[0]).astype('int')
    ranking['ROAD_L'] = ranking['ROAD_RECORD'].apply(lambda x: x.split('-')[1]).astype('int')
    ranking['ROAD_W_PCT'] = ranking['ROAD_W'] / ( ranking['ROAD_W'] + ranking['ROAD_L'] )

    # encode CONFERENCE as an integer (just using pandas - not importing sklearn for just one feature)
    ranking['CONFERENCE'] = ranking['CONFERENCE'].apply(lambda x: 0 if x=='East' else 1 ).astype('int') 

    # remove duplicates (there should only be one TEAM_ID per STANDINGSDATE)
    ranking = ranking[~ranking.duplicated(subset=['TEAM_ID','STANDINGSDATE'])]

    # drop unnecessary fields
    drop_fields = ['SEASON_ID', 'LEAGUE_ID', 'RETURNTOPLAY', 'TEAM', 'HOME_RECORD', 'ROAD_RECORD']
    ranking = ranking.drop(drop_fields,axis=1)

    return ranking


def process_games_details(details: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning on the games_details dataset.

    Args:
        details (pd.DataFrame): the raw games_details dataframe
    
    Returns:
        the cleaned games_details dataframe

    """

    
    # convert MIN:SEC to float
    df = details.loc[details['MIN'].str.contains(':',na=False)]
    df['MIN_whole'] = df['MIN'].apply(lambda x: x.split(':')[0]).astype("int8")
    df['MIN_seconds'] = df['MIN'].apply(lambda x: x.split(':')[1]).astype("int8")
    df['MIN'] = df['MIN_whole'] + (df['MIN_seconds'] / 60)

    details['MIN'].loc[details['MIN'].str.contains(':',na=False)] = df['MIN']
    details['MIN'] = details['MIN'].astype("float16")

    # convert negatives to positive
    details['MIN'].loc[details['MIN'] < 0] = -(details['MIN'])

    # update START_POSITION if did not play (MIN = NaN)
    details['START_POSITION'].loc[details['MIN'].isna()] = 'NP'

    # update START_POSITION if null
    details['START_POSITION'] = details['START_POSITION'].fillna('NS')

    # drop unnecessary fields
    drop_fields = ['COMMENT', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_NAME', 'NICKNAME'] 
    details = details.drop(drop_fields,axis=1)
    
    return details


def add_TARGET(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a TARGET column to the dataframe by copying HOME_TEAM_WINS.

    Args:
        df (pd.DataFrame): the dataframe to add the TARGET column to

    Returns:
        the games dataframe with a TARGET column

    """

    df['TARGET'] = df['HOME_TEAM_WINS']
    
    return df

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train and test sets.

    Splits the latest season as the test set and the rest as the train set.
    The second latest season included with the test set to allow for feature engineering.

    Args:
        df (pd.DataFrame): the dataframe to split

    Returns:
        the train and test dataframes

    """

    latest_season = df['SEASON'].unique().max()

    train = df[df['SEASON'] < (latest_season)]
    test = df[df['SEASON'] >= (latest_season - 1)]
    
    return train, test

