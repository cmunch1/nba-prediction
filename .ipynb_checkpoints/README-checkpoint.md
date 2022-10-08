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
 - postseason(and possibly preseason if present) games will need to be flagged (by date?)
 - 99 games from early 2003 are missing data; they seem fairly evenly distributed among the teams and probably not worth the effort to manually repopulate. Probably drop these.
 - redundant fields in games.csv: HOME_TEAM_ID / TEAM_ID_home and VISITOR_TEAM_ID / TEAM_ID_away
 - it appears that certain fields can be dropped: 'GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away'
 - some overtime games have as much as 168 points scored by a single team, but data does not indicate if overtime game or not.
 - outlier games may need to be flagged - overtime games, blow-outs, etc...
 - PTS, REB, AST have trended up the last several seasons, but Home win ratio is down
 - Strongest postive correlations: FG_PCT and AST to PTS.
 - Strongest negative correlation: REB_PCT_away to FG_PCT_home 
 - Strongest correlations to winning (in order): FG_PCT, PTS, FG3_PCT, AST, REB, FT_PCT (same ordering for either home or away)
 - When limiting winning correlations to just 2021 season or to just the last 5 seasons, then FG_PCT and PTS are reversed for away teams as is AST_away and REB_away (PTS, FG_PCT, FG3_PCT, REB, AST, FT_PCT for away teams). 
