
def process_games(games):
   
    # remove preseason games (GAME_ID begins with a 1)
    games = games[games['GAME_ID'] > 20000000]

    # flag postseason games (GAME_ID begins with >2)
    games['PLAYOFF'] = (games['GAME_ID'] >= 30000000).astype('int8')

    # remove duplicates (each GAME_ID should be unique)
    games = games[~games.duplicated(subset=['GAME_ID'])]

    # drop unnecessary fields
    drop_fields = ['GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away']
    games = games.drop(drop_fields,axis=1)
    
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


def merge_games_ranking (games, ranking):
    
    # first merge ranking data for home team and then again for away team
    
    # rename columns for merging Home Team ranking data
    ranking = ranking.rename(columns={'STANDINGSDATE': 'GAME_DATE_EST', 'TEAM_ID': 'HOME_TEAM_ID'})
    games_ranking_home = pd.merge(games, ranking, how="left", on=["GAME_DATE_EST", "HOME_TEAM_ID"])

    # rename columns for merging Visitor Team ranking data
    ranking = ranking.rename(columns={'HOME_TEAM_ID': 'VISITOR_TEAM_ID'})
    games_ranking = pd.merge(games_ranking_home, ranking, how="left", on=["GAME_DATE_EST", "VISITOR_TEAM_ID"])
    
    return games_ranking


def add_TARGET(games_ranking):

    # all the data in games_ranking is post-game data after the game has already been won or lost
    # the win/lose TARGET needs to be shifted down to previous game so it is aligned with pre-game predictor data

    # add feature with default to 0
    games_ranking['TARGET'] = 0

    # sort games by the order in which they were played for each home team
    games_ranking = games_ranking.sort_values(by = ['HOME_TEAM_ID', 'GAME_ID'], axis=0, ascending=[False, False], ignore_index=True)

    # for each season and each team, shift HOME_TEAM_WINS down one to TARGET
    home_teams = games_ranking['HOME_TEAM_ID'].unique().tolist()
    seasons = games_ranking['SEASON'].unique().tolist()

    for season in seasons:
        for team in home_teams:
            games_ranking['TARGET'].loc[(games_ranking['SEASON'] == season) & (games_ranking['HOME_TEAM_ID'] == team)] = games_ranking['HOME_TEAM_WINS'].shift(periods=1)
            

    # remove games with null TARGET
    games_ranking = games_ranking[games_ranking['TARGET'].notna()]
    
    return games_ranking

def split_train_test(df):

    latest_season = df['SEASON'].unique().max()

    train = df[df['SEASON'] < (latest_season)]
    test = df[df['SEASON'] >= (latest_season - 1)]
    
    return train, test

