import pandas as pd
import numpy as np

def fix_datatypes(df):
    
    '''
    Converts date to proper format and reduces memory footprint of ints and floats
    '''

    
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])

    long_integer_fields = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']

    #convert long integer fields to int32 from int64
    for field in long_integer_fields:
        df[field] = df[field].astype('int32')
    
    #convert the remaining int64s to int8
    for field in df.select_dtypes(include=['int64']).columns.tolist():
        df[field] = df[field].astype('int8')
        
    #convert float64s to float16s
    for field in df.select_dtypes(include=['float64']).columns.tolist():
        df[field] = df[field].astype('float16')
        
    return df

def add_date_features(df):
    
    '''
    Converts game date to month to limit cardinality
    '''
    
    import pandas as pd

    df['MONTH'] = df['GAME_DATE_EST'].dt.month
    
    return df

def remove_playoff_games(df):
    
    '''
    Remove playoff games 
    '''
    
    df = df[df["PLAYOFF"] == 0]
    df = df.drop("PLAYOFF", axis = 1)
    
    return df

def process_x_minus_y(df):
    #Subtract visitor teams stats from the home teams stats for key fields
    # field_x - field_y
    
    all_features = df.columns.tolist()
    comparison_features = [x for x in all_features if "_y" in x]
    
    #don't include redunant features. (x - league_avg) - (y - league_avg) = x-y
    comparison_features = [x for x in comparison_features if "_MINUS_LEAGUE_AVG" not in x]
    
    for feature in comparison_features:
        feature_base = feature[:-2] #remove "_y" from the end
        df[feature_base + "_x_minus_y"] = df[feature_base + "_x"] - df[feature_base + "_y"]
        
    #df = df.drop("CONFERENCE_x_minus_y") #category variable not meaningful?
        
    return df

def process_x_minus_league_avg(df, feature_list, team_feature):

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
    df_temp = (df_temp.set_index('GAME_DATE_EST')
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
    for feature in feature_list:
        df[feature + "_MINUS_LEAGUE_AVG"] = df[feature] - df[feature + "_LEAGUE_AVG"]

    # drop temp features that were only used for subtraction
    df = df.drop(temp_features, axis = 1)
    
    return df


def add_rolling_home_visitor(df, location, roll_list): 
    
    # location = "HOME" or "VISITOR"
    # roll_list = list of number of games for each rolling mean, e.g. [3, 5, 7, 10, 15]

    # new version 2022-10-31
    # now ignoring season boundaries and with longer rolling means 
    # AND create a field where the all-team average is subtracted from each field
    
    
    # add features showing how well the home team has done in its last home games 
    # and how well the visitor team has done in its last away games
    # add rolling means 
    # add win streaks (negative number if losing streak)
    # these are for the home teams last  *home* games
    # and for the visitor teams last *away* games
    
    location_id = location + "_TEAM_ID"

    # sort games by the order in which they were played for each home or visitor team
    df = df.sort_values(by = [location_id, 'GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
    
    # Win streak, negative if a losing streak
    df[location + '_TEAM_WIN_STREAK'] = df['HOME_TEAM_WINS'].groupby((df['HOME_TEAM_WINS'] != df.groupby([location_id])['HOME_TEAM_WINS'].shift()).cumsum()).cumcount() + 1
    df[location + '_TEAM_WIN_STREAK'].loc[df['HOME_TEAM_WINS'] == 0]  = df[location + '_TEAM_WIN_STREAK'].loc[df['HOME_TEAM_WINS'] == 0]  * -1

    # If visitor, the streak has opposite meaning (3 wins for home team is 3 losses in a row for visitor)
    if location == 'VISITOR':
        df[location + '_TEAM_WIN_STREAK'] = - df[location + '_TEAM_WIN_STREAK']  


    # rolling means
    feature_list = ['HOME_TEAM_WINS', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home']
    
    if location == 'VISITOR':
        feature_list = ['HOME_TEAM_WINS', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
    
      
    roll_feature_list = []
    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = location + '_' + feature + '_AVG_LAST_' + str(roll) + '_' + location
            if feature == 'HOME_TEAM_WINS': #remove the "HOME_" for better readability
                roll_feature_name = location + '_' + feature[5:] + '_AVG_LAST_' + str(roll) + '_' + location
            roll_feature_list.append(roll_feature_name)
            df[roll_feature_name] = df.groupby(['HOME_TEAM_ID'])[feature].rolling(roll).mean().values
            
    
    
    # determine league avg for each stat and then subtract it from the each team's avg
    # as a measure of how well that team compares to all teams in that moment in time
    
    #remove win averages from roll list - the league average will always be 0.5 (half the teams win, half lose)
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('HOME_TEAM_WINS')]
    
    df = process_x_minus_league_avg(df, roll_feature_list, location_id)
    
 
    return df

def process_games_consecutively(df_data):
    # re-organize so that all of a team's games can be listed in chronological order whether HOME or VISITOR
    # this will facilitate feature engineering (winpct vs team X, 5-game winpct, current win streak, etc...)
    
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

def add_matchups(df, roll_list):

    # new version 2022-10-30
    # now ignoring season boundaries and added roll parameters

    # group all the games that 2 teams played each other 
    # calculate home team win pct and the home team win/lose streak
    

    df = df.sort_values(by = ['TEAM1', 'TEAM2','GAME_DATE_EST'], axis=0, ascending=[True, True, True], ignore_index=True)

    for roll in roll_list:
        df['MATCHUP_WINPCT_' + str(roll)] = df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].rolling(roll).mean().values

    df['MATCHUP_WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'] != df.groupby(['TEAM1','TEAM2'])['TEAM1_win'].shift()).cumsum()).cumcount() + 1
    #make streak negative if a losing streak
    df['MATCHUP_WIN_STREAK'].loc[df['TEAM1_win'] == 0]  = df['MATCHUP_WIN_STREAK'].loc[df['TEAM1_win'] == 0]  * -1

    
    return df

def add_past_performance_all(df, roll_list):
    
    # roll_list = list of number of games for each rolling mean, e.g. [3, 5, 7, 10, 15]
    
    # new version 2022-11-03
    # now ignoring season boundaries and with longer rolling means (20 and 40 games)
    # AND create a field where the all-team average is subtracted from each field
   
    # add features showing how well each team has done in its last games
    # regardless whether they were at home or away
    # add rolling means for last 3, 5, 7, 10, 20, 40 games
    # add win streaks (negative number if losing streak)
    
    #this data will need to be re-linked back to the main dataframe after all processing is done,
    #joining TEAM1 to HOME_TEAM_ID for all records and then TEAM1 to VISITOR_TEAM_ID for all records
    
    #TEAM1 will be the key field. TEAM2 was used solely to process past team matchups


    df = df.sort_values(by = ['TEAM1','GAME_DATE_EST'], axis=0, ascending=[True, True,], ignore_index=True)
  
    #streak of games won/lost, make negative is a losing streak
    df['WIN_STREAK'] = df['TEAM1_win'].groupby((df['TEAM1_win'] != df.groupby(['TEAM1'])['TEAM1_win'].shift()).cumsum()).cumcount() + 1
    df['WIN_STREAK'].loc[df['TEAM1_win'] == 0]  = df['WIN_STREAK'].loc[df['TEAM1_win'] == 0]  * -1
    
    #streak of games played at home/away, make negative if away streak
    df['HOME_AWAY_STREAK'] = df['TEAM1_home'].groupby((df['TEAM1_home'] != df.groupby(['TEAM1'])['TEAM1_home'].shift()).cumsum()).cumcount() + 1
    df['HOME_AWAY_STREAK'].loc[df['TEAM1_home'] == 0]  = df['HOME_AWAY_STREAK'].loc[df['TEAM1_home'] == 0]  * -1
    
    #rolling means 
    
    feature_list = ['TEAM1_win', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']
   
    #create new feature names based upon rolling period
    
    roll_feature_list =[]

    for feature in feature_list:
        for roll in roll_list:
            roll_feature_name = feature + '_AVG_LAST_' + str(roll) + '_ALL'
            roll_feature_list.append(roll_feature_name)
            df[roll_feature_name] = df.groupby(['TEAM1'])[feature].rolling(roll).mean().values

    
    
    # determine league avg for each stat and then subtract it from the each team's average
    # as a measure of how well that team compares to all teams in that moment in time
    
    #remove win averages from roll list - the league average will always be 0.5 (half the teams win, half lose)
    roll_feature_list = [x for x in roll_feature_list if not x.startswith('TEAM1_win')]
    
    df = process_x_minus_league_avg(df, roll_feature_list, 'TEAM1')
    
    
    return df

def combine_new_features(df, df_consecutive):
     
    # add back all the new features created in the consecutive dataframe to the main dataframe
    # all data for TEAM1 will be applied to the home team and then again to the visitor team
    # except for head-to-head MATCHUP data, which will only be applied to home team (redundant to include for both)
    # the letter '_x' will be appeneded to feature names when adding to home team
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
    
    #don't include matchup features for visitor team since they are equivant for both home and visitor
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

def remove_non_rolling(df):
    
    drop_columns =[]
    
    all_columns = df.columns.tolist()
    
    drop_columns1 = ['HOME_TEAM_WINS', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home']
    drop_columns2 = ['PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
    
    drop_columns = drop_columns + drop_columns1
    drop_columns = drop_columns + drop_columns2 
    
    use_columns = [item for item in all_columns if item not in drop_columns]
    
    return df[use_columns]
