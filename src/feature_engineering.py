import pandas as pd
import os

# temporary hack for the way the streamlit app is handling imports
try :
    from src.constants import (
        LONG_INTEGER_FIELDS,
        SHORT_INTEGER_FIELDS,
        DATE_FIELDS,
        DROP_COLUMNS_NON_ROLLING,
        HOME_VISITOR_ROLL_LIST, #list of rolling periods to use when restricting to home or visitor role
        ALL_ROLL_LIST, #list of rolling periods to use when NOT restricting to home or visitor role
        FEATURE_LIST_HOME, # list of non-rolling features to create rolling features for home role
        FEATURE_LIST_AWAY, # list of non-rolling features to create rolling features for visitor role
    )
except:
    from constants import (
        LONG_INTEGER_FIELDS,
        SHORT_INTEGER_FIELDS,
        DATE_FIELDS,
        DROP_COLUMNS_NON_ROLLING,
        HOME_VISITOR_ROLL_LIST, #list of rolling periods to use when restricting to home or visitor role
        ALL_ROLL_LIST, #list of rolling periods to use when NOT restricting to home or visitor role
        FEATURE_LIST_HOME, # list of non-rolling features to create rolling features for home role
        FEATURE_LIST_AWAY, # list of non-rolling features to create rolling features for visitor role
    )


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to perform all the steps of feature engineering

    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe

    
    Feature engineering to add: 
        - rolling averages of key stats, 
        - win/lose streaks, 
        - home/away streaks, 
        - specific matchup (team X vs team Y) rolling averages and streaks
        - home team rolling stats minus visitor team rolling stats
        - rolling stats minus current league average
        
    Functions include:
        - fix_datatypes(): converts date to proper format and reduces memory footprint of ints and floats
        - add_date_features(): adds a feature for month number from game date 
        - remove_playoff_games(): playoff games may bias the statistics
        - add_rolling_home_visitor(): rolling avgs and streaks for home/visitor team when playing as home/visitor
        - process_games_consecutively(): separate home team stats from visitor team stats for each game and stack these together by game date
        - add_past_performance_all(): rolling avgs and streaks no matter if playing as home or visitor team
        - process_x_minus_league_avg: subtract league avg rolling stats from team's rolling stats
        - add_matchups(): rolling avgs and steaks for each time when Team A played Team B
        - combine_new_features(): combine back home team and visitor team features so each game has only one row again
        - process_x_minus_y(): subtract visitor team rolling stats from home rolling stats
    """
    
    
    df = fix_datatypes(df, DATE_FIELDS, SHORT_INTEGER_FIELDS, LONG_INTEGER_FIELDS)
    df = add_date_features(df)
    df = remove_playoff_games(df)
    df = add_rolling_home_visitor(df, "HOME", HOME_VISITOR_ROLL_LIST) # list of variety of rolling lengths
    df = add_rolling_home_visitor(df, "VISITOR", HOME_VISITOR_ROLL_LIST) # list of variety of rolling lengths
    
    df_consecutive = process_games_consecutively(df)
    df_consecutive = add_matchups(df_consecutive, HOME_VISITOR_ROLL_LIST) # list of variety of rolling lengths
    df_consecutive = add_past_performance_all(df_consecutive, ALL_ROLL_LIST) # list of variety of rolling lengths

    #add these features back to main dataframe
    df = combine_new_features(df,df_consecutive) 
        
    df = process_x_minus_y(df)
    
    return df


def fix_datatypes(df: pd.DataFrame, date_columns: list, short_integer_fields: list, long_integer_fields: list)-> pd.DataFrame:
    """
    Converts date to proper format and reduces memory footprint of ints and floats

    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe

    """

    for field in date_columns:
        df[field] = pd.to_datetime(df[field], utc=True)

    #convert long integer fields to int32 from int64
    for field in long_integer_fields:
        df[field] = df[field].astype('int32')

    # convert specific fields to int16 to avoid type issues with hopsworks.ai
    # Fill NaNs for short integer stat fields (e.g., new/unplayed games) prior to casting
    for field in short_integer_fields:
        print(f"Converting field {field} to int16", df[field].dtype, df[field].min(), df[field].max())
        df[field] = df[field].fillna(0)
        df[field] = df[field].astype('int16')
        
    #convert to positive. For some reason, some values have been saved as negative numbers
    for field in short_integer_fields:
        df[field] = df[field].abs()
    
    #convert the remaining int64s to int8
    for field in df.select_dtypes(include=['int64']).columns.tolist():
        df[field] = df[field].astype('int8')
        
    #convert float64s to float16s
    for field in df.select_dtypes(include=['float64']).columns.tolist():
        df[field] = df[field].astype('float16')
        
    return df


def add_date_features(df: pd.DataFrame)-> pd.DataFrame:
    """
    Creates new features from the game date, which will hopefully be more useful for the model
    
    Currently simply converts game date to just month and add as a feature. This limits cardinality of the date feature.

    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe
    """

    df['MONTH'] = df['GAME_DATE_EST'].dt.month
    
    return df


def remove_playoff_games(df: pd.DataFrame)-> pd.DataFrame:
    """
    Remove playoff games 

    Playoff games may bias the statistics because they are not played under the same conditions as regular season games and are played in a tournament format.

    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe
    """

    
    # Filter to only non-playoff games and then drop the PLAYOFF feature
    
    #df = df[df["PLAYOFF"] == 0] # keep playoff games for now so that the model can be demonstrated during the playoffs
    df = df.drop("PLAYOFF", axis = 1) 
    
    return df


def add_rolling_home_visitor(df: pd.DataFrame, location: str, roll_list: list)-> pd.DataFrame:
    """
    Add rolling avgs and win/lose streaks for home/visitor team when playing as home/visitor for a variety of rolling lengths

    This function also invokes another function to calculate the league average rolling stats for that moment in time and subtracts these from the team's rolling stats.

    Args:
        df (pd.DataFrame): the dataframe to process
        location (str): "HOME" or "VISITOR"
        roll_list (list): list of number of games for each rolling mean, e.g. [3, 5, 7, 10, 15] 

    Returns:
        the processed dataframe


    We are adding features that show how well the home team has done in its last home games and how well the visitor team has done in its last away games.
    We are also determining the current win streak for each team (negative if losing streak) when playing as home or visitor team.   
    
    """ 

    # compile stats for home or visitor team    
    location_id = location + "_TEAM_ID"

    # sort games by the order in which they were played for each home or visitor team
    df = df.sort_values(by = [location_id, 'GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
    
    # Win streak prior to current game (negative if a losing streak)
    # Define team win indicator based on location role
    if location == 'HOME':
        df['_team_won_role'] = df['HOME_TEAM_WINS']
    else:  # VISITOR
        df['_team_won_role'] = 1 - df['HOME_TEAM_WINS']

    # Use previous results only to avoid leakage into current row
    df['_w_prev'] = df.groupby([location_id])['_team_won_role'].shift(1)

    def _streak_from_prev(group: pd.DataFrame) -> pd.Series:
        r = group['_w_prev']
        # Identify runs of equal prev outcomes
        run_id = (r != r.shift()).cumsum()
        streak_len = run_id.groupby(run_id).cumcount() + 1
        # Where no previous result, streak is 0
        streak_len = streak_len.where(r.notna(), 0)
        # Negative for losing streaks
        signed = streak_len.where(r == 1, -streak_len)
        return signed.astype('int16')

    df[location + '_TEAM_WIN_STREAK'] = df.groupby([location_id], group_keys=False).apply(_streak_from_prev)
    # Clean up temp columns
    df = df.drop(columns=['_team_won_role', '_w_prev'])


    # rolling means 
    feature_list = FEATURE_LIST_HOME
    
    if location == 'VISITOR':
        feature_list = FEATURE_LIST_AWAY
    
      
    roll_feature_list = []
    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = location + '_' + feature + '_AVG_LAST_' + str(roll) + '_' + location
            if feature == 'HOME_TEAM_WINS': #remove the "HOME_" for better readability
                roll_feature_name = location + '_' + feature[5:] + '_AVG_LAST_' + str(roll) + '_' + location
            roll_feature_list.append(roll_feature_name)
            # use the correct team id for grouping depending on location (HOME vs VISITOR)
            df[roll_feature_name] = df.groupby([location_id])[feature].rolling(roll, closed= "left").mean().values
            
    
    
    # determine league avg for each stat and then subtract it from the each team's avg
    # as a measure of how well that team compares to all teams in that moment in time
    
    #remove win averages from roll list - the league average will always be 0.5 (half the teams win, half lose)
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('HOME_TEAM_WINS')]
    #print(location_id)
    df = process_x_minus_league_avg(df, roll_feature_list, location_id)
    
 
    return df


def process_games_consecutively(df_data: pd.DataFrame)-> pd.DataFrame:
    """
    
    Separate home team stats from visitor team stats for each game and stack these together by game date. 
    
    (Each game record will go from a single row, Home/Visitor combined, to two rows, one for home team and one for visitor)
    
    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe
    """
    
    # re-organize so that all of a team's games can be listed in chronological order whether HOME or VISITOR
    # this will facilitate feature engineering (winpct vs team X, 5-game winpct, current win streak, etc...)
    
    # before this step, the data is stored by game, and each game has 2 teams
    # this function will separate each teams stats so that each game has 2 rows (one for each team) instead of one combined row
    
    #this data will need to be re-linked back to the main dataframe after all processing is done,
    #joining TEAM1 to HOME_TEAM_ID for all records and then TEAM1 to VISITOR_TEAM_ID for all records
    
    #TEAM1 will be the key field. TEAM2 is used solely to process past team matchups

    # all the home games for each team will be selected and then stacked with all the away games
    
    df_home = pd.DataFrame()
    df_home['GAME_DATE_EST'] = df_data['GAME_DATE_EST']
    df_home['GAME_ID'] = df_data['GAME_ID']
    df_home['TEAM1'] = df_data['HOME_TEAM_ID']
    df_home['TEAM1_home'] = 1
    df_home['TEAM1_win'] = df_data['HOME_TEAM_WINS']
    df_home['TEAM2'] = df_data['VISITOR_TEAM_ID']
    df_home['SEASON'] = df_data['SEASON']
    
    df_home['PTS'] = df_data['PTS_home']
    df_home['FG_PCT'] = df_data['FG_PCT_home']
    df_home['FT_PCT'] = df_data['FT_PCT_home']
    df_home['FG3_PCT'] = df_data['FG3_PCT_home']
    df_home['AST'] = df_data['AST_home']
    df_home['REB'] = df_data['REB_home']
    
    # now for visitor teams  

    df_visitor = pd.DataFrame()
    df_visitor['GAME_DATE_EST'] = df_data['GAME_DATE_EST']
    df_visitor['GAME_ID'] = df_data['GAME_ID']
    df_visitor['TEAM1'] = df_data['VISITOR_TEAM_ID'] 
    df_visitor['TEAM1_home'] = 0
    df_visitor['TEAM1_win'] = df_data['HOME_TEAM_WINS'].apply(lambda x: 1 if x == 0 else 0)
    df_visitor['TEAM2'] = df_data['HOME_TEAM_ID']
    df_visitor['SEASON'] = df_data['SEASON']
    
    df_visitor['PTS'] = df_data['PTS_away']
    df_visitor['FG_PCT'] = df_data['FG_PCT_away']
    df_visitor['FT_PCT'] = df_data['FT_PCT_away']
    df_visitor['FG3_PCT'] = df_data['FG3_PCT_away']
    df_visitor['AST'] = df_data['AST_away']
    df_visitor['REB'] = df_data['REB_away']

    # merge dfs

    df = pd.concat([df_home, df_visitor])

    column2 = df.pop('TEAM1')
    column3 = df.pop('TEAM1_home')
    column4 = df.pop('TEAM2')
    column5 = df.pop('TEAM1_win')

    df.insert(2,'TEAM1', column2)
    df.insert(3,'TEAM1_home', column3)
    df.insert(4,'TEAM2', column4)
    df.insert(5,'TEAM1_win', column5)

    df = df.sort_values(by = ['TEAM1', 'GAME_ID'], axis=0, ascending=[True, True], ignore_index=True)

    return df


def add_matchups(df: pd.DataFrame, roll_list: list)-> pd.DataFrame:
    """
    Add rolling win pcts and win/lose steaks for each time when Team A played Team B for a variety of rolling windows

    Args:
        df (pd.DataFrame): the dataframe to process
        roll_list (list): list of number of games for each rolling mean, e.g. [3, 5, 7, 10, 15] 

    Returns:
        the processed dataframe
    """


    # group all the games that 2 teams played each other 
    # calculate home team win pct and the home team win/lose streak
    

    df = df.sort_values(by = ['TEAM1', 'TEAM2','GAME_DATE_EST'], axis=0, ascending=[True, True, True], ignore_index=True)

    for roll in roll_list:
        df['MATCHUP_WINPCT_' + str(roll)] = df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].rolling(roll, closed= "left").mean().values

    df['MATCHUP_WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'].shift() != df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].shift(2)).cumsum()).cumcount() + 1
   
    # if team1 lost the last game of the streak, then the streak must be a losing streak. make it negative
    df['MATCHUP_WIN_STREAK'].loc[df['TEAM1_win'].shift() == 0] = -1 * df['MATCHUP_WIN_STREAK']
  
    
    return df


def add_past_performance_all(df: pd.DataFrame, roll_list: list)-> pd.DataFrame:
    """
    Add rolling avgs, win/lose streak, and home/away streak no matter if playing as home or visitor team.

    Args:
        df (pd.DataFrame): the dataframe to process
        roll_list (list): list of number of games for each rolling mean, e.g. [3, 5, 7, 10, 15] 

    Returns:
        the processed dataframe
    """
       
    # add features showing how well each team has done in its last games
    # regardless whether they were at home or away
    
    # add rolling means and win streaks (negative number if losing streak)
    
    #this data will need to be re-linked back to the main dataframe after all processing is done,
    #joining TEAM1 to HOME_TEAM_ID for all records and then TEAM1 to VISITOR_TEAM_ID for all records
    
    #TEAM1 will be the key field. TEAM2 was used solely to process past team matchups


    df = df.sort_values(by = ['TEAM1','GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
  
    #streak of games won/lost, make negative is a losing streak
    df['WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'].shift() != df.groupby(['TEAM1'])['TEAM1_win'].shift(2)).cumsum()).cumcount() + 1   
    
    # if team1 lost the last game of the streak, then the streak must be a losing streak. make it negative
    df['WIN_STREAK'].loc[df['TEAM1_win'].shift() == 0]  = -1 * df['WIN_STREAK']
    
    #streak of games played at home/away, make negative if away streak
    df['HOME_AWAY_STREAK'] = df['TEAM1_home'].groupby((df['TEAM1_home'].shift() != df.groupby(['TEAM1'])['TEAM1_home'].shift(2)).cumsum()).cumcount() + 1
    
    # if team1 played the game of the streak away, then the streak must be an away streak. make it negative
    df['HOME_AWAY_STREAK'].loc[df['TEAM1_home'].shift() == 0]  = -1 * df['HOME_AWAY_STREAK']
    
    #rolling means 
   
    feature_list = ['TEAM1_win', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']
   
    #create new feature names based upon rolling period
    
    roll_feature_list =[]

    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = feature + '_AVG_LAST_' + str(roll) + '_ALL'
            roll_feature_list.append(roll_feature_name)
            df[roll_feature_name] = df.groupby(['TEAM1'])[feature].rolling(roll, closed= "left").mean().values

    # determine league avg for each stat and then subtract it from the each team's average
    # as a measure of how well that team compares to all teams in that moment in time
    
    #remove win averages from roll list - the league average will always be 0.5 (half the teams win, half lose)
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('TEAM1_win')]
    
    df = process_x_minus_league_avg(df, roll_feature_list, 'TEAM1')
    
    
    return df



def process_x_minus_league_avg(df: pd.DataFrame, feature_list: list, team_feature: str)-> pd.DataFrame:
    """
    Calculate the league average for every day of the season and then subtract the league average of each stat from the team's current stat for that day.

    This provides a measure of how good the team is compared to the the rest of the league at that moment in time.

    Args:
        df (pd.DataFrame): the dataframe to process
        feature_list (list): list of features to be used for subtraction, e.g. [PTS_AVG_LAST_5_ALL, REB_AVG_LAST_20_ALL]
        team_feature (str): the team's role (subset of data) that is being worked upon ("HOME_TEAM_ID", "VISITOR_TEAM_ID", or "TEAM1" for all roles)

    Returns:
        the processed dataframe

    """

    # create a temp dataframe so that every date can be front-filled
    # we need the current average for all 30 teams for every day during the season
    # whether that team played or not. 
    # We will front-fill from previous days to ensure that every day has stats for every team
    
 
    
    # create feature list for temp dataframe to hold league averages
    temp_feature_list = feature_list.copy()
    temp_feature_list.append(team_feature)
    temp_feature_list.append("GAME_DATE_EST")

    df_temp = df[temp_feature_list]

    
    # populate the dataframe with all days played and forward fill previous value if a particular team did not play that day
    # https://stackoverflow.com/questions/70362869
    df_temp = (df_temp.set_index('GAME_DATE_EST',)
            .groupby([team_feature])[feature_list]
            .apply(lambda x: x.asfreq('d', method = "ffill"))
            .reset_index()
            [temp_feature_list]
            )
    
    # find the average across all teams for each day
    df_temp = df_temp.groupby(['GAME_DATE_EST'])[feature_list].mean().reset_index()
    
    # rename features for merging
    df_temp = df_temp.add_suffix('_LEAGUE_AVG')
    temp_features = df_temp.columns
    
    # merge all-team averages with each record so that they can be subtracted
    df = df.sort_values(by = 'GAME_DATE_EST', axis=0, ascending= True, ignore_index=True)   
    df = pd.merge(df, df_temp, left_on='GAME_DATE_EST', right_on='GAME_DATE_EST_LEAGUE_AVG', how="left",)
    # subtract league average for each feature
    for feature in feature_list:
        df[feature + "_MINUS_LEAGUE_AVG"] = df[feature] - df[feature + "_LEAGUE_AVG"]

    # drop temp features that were only used for subtraction
    df = df.drop(temp_features, axis = 1) 
    
    return df


def combine_new_features(df: pd.DataFrame, df_consecutive: pd.DataFrame)-> pd.DataFrame:
    """
    Re-combine back the features created in the consecutive dataframe to the main dataframe.

    The consecutive dataframe was used to derive features regardless of whether the team was home or away, and now we need to add those features back to the main dataframe.

    Args:
        df (pd.DataFrame): the main dataframe where each row is a game with both a home team and a visitor team
        df_consecutive (pd.DataFrame): the dataframe where each row is a game with only one team (either home or visitor)

    Returns:
        the merged dataframe
    """

     
    # add back all the new features created in the consecutive dataframe to the main dataframe
    # all data for TEAM1 will be applied to the home team and then again to the visitor team
    # except for head-to-head MATCHUP data, which will only be applied to home team (redundant to include for both)
    # the letter '_x' will be appended to feature names when adding to home team
    # the letter '_y' will be appended to feature names when adding to visitor team
    # to match the existing convention in the dataset
    
    #first select out the new features
    all_features = df_consecutive.columns.tolist()
    link_features = ['GAME_ID', 'TEAM1', ]
    redundant_features = ['GAME_DATE_EST','TEAM1_home','TEAM1_win','TEAM2','SEASON','PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB',]
    matchup_features = [x for x in all_features if "MATCHUP" in x]
    ignore_features = link_features + redundant_features
    
    new_features = [x for x in all_features if x not in ignore_features]
    
    # first home teams
    
    df1 = df_consecutive[df_consecutive['TEAM1_home'] == 1]
    #add "_x" to new features
    df1.columns = [x + '_x' if x in new_features else x for x in df1.columns]
    #drop features that don't need to be merged
    df1 = df1.drop(redundant_features,axis=1)
    #change TEAM1 to HOME_TEAM_ID for easy merging
    df1 = df1.rename(columns={'TEAM1': 'HOME_TEAM_ID'})
    df = pd.merge(df, df1, how="left", on=["GAME_ID", "HOME_TEAM_ID"])
    
    #don't include matchup features for visitor team since they are equivalent for both home and visitor
    new_features = [x for x in new_features if x not in matchup_features]
    df_consecutive = df_consecutive.drop(matchup_features,axis=1)
    
    # next visitor teams
    
    df2 = df_consecutive[df_consecutive['TEAM1_home'] == 0]
    #add "_y" to new features
    df2.columns = [x + '_y' if x in new_features else x for x in df2.columns]
    #drop features that don't need to be merged
    df2 = df2.drop(redundant_features,axis=1)
    #change TEAM1 to VISITOR_TEAM_ID for easy merging
    df2 = df2.rename(columns={'TEAM1': 'VISITOR_TEAM_ID'})
    df = pd.merge(df, df2, how="left", on=["GAME_ID", "VISITOR_TEAM_ID"])
    
    return df


def process_x_minus_y(df: pd.DataFrame)-> pd.DataFrame:
    """
    Subtract visitor team rolling stats from home rolling stats.

    This may (or may not) be useful for the model to explicitly see the difference between the two teams. GBM models may be able to handle this automatically, but other models may not.
    
    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        the processed dataframe
    """

    # field_x - field_y
    
    # remove the current games stats since they are data leaks - we don't know these until after the game is played
    useful_features = remove_non_rolling(df)
    
    comparison_features = [x for x in useful_features if "_y" in x]
    
    #don't include redundant features. (x - league_avg) - (y - league_avg) = x-y
    comparison_features = [x for x in comparison_features if "_MINUS_LEAGUE_AVG" not in x]
    
    for feature in comparison_features:
        feature_base = feature[:-2] #remove "_y" from the end
        df[feature_base + "_x_minus_y"] = df[feature_base + "_x"] - df[feature_base + "_y"]
        
    #df = df.drop("CONFERENCE_x_minus_y") #category variable not meaningful?
        
    return df


def remove_non_rolling(df: pd.DataFrame) -> list:
    """
    Returns a list of columns in a dataframe with the current games stats removed, leaving only rolling averages and streaks

    Args:
        df (pd.DataFrame): the dataframe to process

    Returns:
        list: only the columns that are rolling averages and streaks
    """
 
    # remove non-rolling features - these are data leaks
    # they are stats from the actual game that decides winner/loser, 
    # but we don't know these stats before a game is played
    
    # These must be retained in the database to recalculate rolling avgs and streaks in the future,
    # so are filtered out as appropriate instead of deleted
    
   
    all_columns = df.columns.tolist()
    
    use_columns = [item for item in all_columns if item not in DROP_COLUMNS_NON_ROLLING]
    
    return use_columns
