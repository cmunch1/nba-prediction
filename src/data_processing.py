
def process_games(games):
   
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


def process_ranking(ranking):

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


def process_games_details(details):
    
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


def add_TARGET(games_ranking):

    games_ranking['TARGET'] = games_ranking['HOME_TEAM_WINS']
    
    return games_ranking

def split_train_test(df):

    latest_season = df['SEASON'].unique().max()

    train = df[df['SEASON'] < (latest_season)]
    test = df[df['SEASON'] >= (latest_season - 1)]
    
    return train, test

