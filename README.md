# NBA Game Predictor Project

## The goal of this project is to develop an NBA game-winner predictor model capable of winning money in a sports betting situation better than typical online betting advice.

### Plan

The model will determine the probability for each game that the home team will win, and this will  be used in conjunction with a betting strategy that attempts to optimize win probability vs payout odds. Bets will be selected only for games that match a profitable criteria.

The model will be deployed online to predict winners of NBA games each day when the season is active. Performance metrics will also be charted as the season goes by. 

If the model proves successful, possible revenue options include:
 - Betting on games (where legal)
 - Providing the software as a service with paid ads or subscriptions

### DATA

Data from the 2013 thru 2021 season has been archived on Kaggle. The NBA provides an API to access current data each day as the season goes by. 

Currently available data includes:

 - games_details.csv .. (each-game player stats for everyone on the roster)
 - games.csv .......... (each-game team stats: final scores, points scored, field-goal & free-throw percentages, etc...)
 - players.csv ........ (index of players' names and teams)
 - ranking.csv ........ (incremental daily record of standings, games played, won, lost, win%, home record, road record)
 - teams.csv .......... (index of team info such as city and arena names and also head coach) 
 
 NOTES 
 - games and ranking will need to be linked by SEASON->SEASON_ID, GAME_DATE_EST->STANDINGSDATE(-1 day), HOME_TEAM_ID->TEAM_ID / VISITOR_TEAM_ID->TEAM_ID
 - games and game_details will need to be linked by GAME_ID->GAME_ID, HOME_TEAM_ID->TEAM_ID / VISITOR_TEAM_ID->TEAM_ID
 - just very basic stats are provided, but the data is there to generate much more, particulary running averages and other agreggates
 - some games during COVID were played in the "Bubble" - not true home arenas. These will need to be flagged at the least.
 - preseason games should be dropped. Post season probably as well.
 
 games.csv
 - 99 games from early 2003 are missing data; they seem fairly evenly distributed among the teams and probably not worth the effort to manually repopulate. Probably drop these.
 - GAME_ID format:
    - 1st digit: 1=pre-season, 2=regular season, >2 = post season,
    - 2nd & 3rd digit: last two digits of season year (eg 103######## represents preseason game in 2003)
 - redundant fields in games.csv: HOME_TEAM_ID / TEAM_ID_home and VISITOR_TEAM_ID / TEAM_ID_away
 - it appears that certain fields can be dropped: 'GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away'
 - some overtime games have as much as 168 points scored by a single team, but data does not indicate if overtime game or not.
 - outlier games may need to be flagged - overtime games, blow-outs, etc...
 - PTS, REB, AST have trended up the last several seasons, but Home win ratio is down
 - Strongest postive correlations: FG_PCT and AST to PTS.
 - Strongest negative correlation: REB_PCT_away to FG_PCT_home 
 - Strongest correlations to winning (in order): FG_PCT, PTS, FG3_PCT, AST, REB, FT_PCT (same ordering for either home or away)
 - When limiting winning correlations to just 2021 season or to just the last 5 seasons, then FG_PCT and PTS are reversed for away teams as is AST_away and REB_away (PTS, FG_PCT, FG3_PCT, REB, AST, FT_PCT for away teams).
 
 ranking.csv
 - this is primarily just supplemental data that needs to be integrated with the games data
 - RETURNTOPLAY field only used for a small portion of ranking stats (East conference March 2020 thru Dec 2020)
 - LEAGUE_ID always 0, can drop
 - SEASON_ID beginning with 1 appears to be preseason standings, while regular season standings start with a 2
 - My initial plan is to focus just on regular season games, so preseason standings can be removed from the updated dataset
 - Fields to be dropped: 'LEAGUE_ID', 'RETURNTOPLAY', 'TEAM'
 - HOME_RECORD and ROAD_RECORD each needs to be split into games won, games lost, and win percentage
 
 games_details.csv
 - this contains all individual players stats for each game
 - 105603 records have NaN for all stats - the player did not play that game
 - TEAM_ABBREVIATION, TEAM_CITY, PLAYER_NAME, NICKNAME are not needed and can be found in index tables if needed
 - COMMENT field will denote why player did not play, usually in a "XXX -" format (e.g. DNP - Injury/Illness) , but 1121 records do not follow the format
 - START_POSITION is null for both players that played but did not start and for players that did not play - maybe separate these two
 - MIN (minutes played) contains mixed formats: integers and MIN:SEC
 - MIN contains 12 records with negative values
 - 19 records have players over 60 minutes and seem to be overtime games
 - several other outlier stats have been verified as correct
 
 
 ### DATA PROCESSING SUMMARY

Games.csv and ranking.csv will be merged after initial data processing. Since all the features are *post* game data (final score, winning percentage after the game, etc...) they cannot be used as predictors for the current game. All the features will be used as predictors for the "next game", so the data will need to adjusted so that the TARGET (HOME_TEAM_WINS) is in the same row as the predictors. 

The easiest approach seems to be add a field called TARGET that denotes whether the home team won its *next* game or not.

Game_details.csv will intially be held in reserve for feature engineering. With a roster of 24 players per game and 21 features per player, the initial plan is NOT to add all these 500 features indiscriminately but to instead try to find useful features and incorporate these.

Scaling and power-transforms will not be used at this time since the plan is to use GBTs (gradient boosted trees) such as XGBoost where these tranforms are not needed. These transforms may be needed later for PCA and other techniques, though.


duplicates

 - both games.csv and ranking.csv contain several duplicated rows from Dec 2020 (covid season) that the pandas function *df.duplicated()* failed to detect in EDA. These will be filtered out using subsets instead of the entire dataframe.

 games.csv
 
 - delete preseason games (this will also take care of the null games from early 2003)
 - keep only games where GAME_STATUS_TEXT = 'Final' (for better utility in the future)
 - remove duplicated records 
 - flag postseason games 
 - drop 'GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away'

ranking.csv
 
 - drop preseason rankings (SEASON_ID begins with 1)
 - split HOME_RECORD into HOME_W, HOME_L, and HOME_W_PCT
 - split ROAD_RECORD into ROAD_W, ROAD_L, and ROAD_W_PCT
 - numericaly encode CONFERENCE (East or West)
 - remove duplicated records
 - drop 'SEASON_ID', 'LEAGUE_ID', 'RETURNTOPLAY', 'TEAM', 'HOME_RECORD', 'ROAD_RECORD'

 game_details.csv
 
 - fix mixed formats in MIN and convert to float
 - fix negatives in MIN
 - if MIN is null, edit START_POSITION to 'NP' (not played)
 - any START_POSITION remaining null, convert to NS (not start, but still played)
 - drop TEAM_ABBREVIATION, TEAM_CITY, PLAYER_NAME, NICKNAME, COMMENT
 
 Join games with ranking
 
  - LINK: games.GAME_DATE_EST, games.HOME_TEAM_ID, -> ranking.STANDINGSDATE, ranking.TEAM_ID 
  - ADD: CONFERENCE, G, W, L, W_PCT, HOME_W, HOME_L, HOME_W_PCT, ROAD_W, ROAD_L, ROAD_W_PCT
  - repeat with AWAY_TEAM_ID instead of HOME_TEAM_ID
  
 Add TARGET
 
  - Sort games by HOME_TEAM_ID and GAME_ID
  - for each SEASON and HOME_TEAM_ID, shift HOME_TEAM_WINS down to TARGET for previous game
  - remove games with null TARGETs (last game played each season by each team will have no null TARGET)
  
  ### Train / Test Split
  
  - Latest season is used as Test data and previous seasons are used as Train data
  
  ### Baseline Models
  
Simple If-Then Models

 - Team with best record wins (Accuracy = 0.56, AUC = 0.58 on Test data)
 - Home team always wins (Accuracy = 0.55, AUC = 0.50 on Test data)
 - Home team wins unless they have losing home record (Accuracy = 0.59, AUC = 0.57 on Test data)
 - Home team wins unless they have lost last 3 home games (Accuracy = 0.59, AUC = 0.55 on Test data)
 
ML Models

 - LightGBM (Accuracy = 0.59, AUC = 0.62 on Test data)
 - XGBoost (Accuracy = 0.56, AUC = 0.57 on Test data)