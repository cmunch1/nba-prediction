########### FEATURE ENGINEERING CONSTANTS ##############

LONG_INTEGER_FIELDS = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']
SHORT_INTEGER_FIELDS = ['PTS_home', 'AST_home', 'REB_home', 'PTS_away', 'AST_away', 'REB_away']
DATE_FIELDS = ['GAME_DATE_EST']
DROP_COLUMNS = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', ]

# non-rolling features; used to generate rolling features
FEATURE_LIST_HOME = ['HOME_TEAM_WINS', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home']
FEATURE_LIST_AWAY = ['HOME_TEAM_WINS', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']

DROP_COLUMNS_NON_ROLLING = FEATURE_LIST_HOME + FEATURE_LIST_AWAY 

# lengths of rolling averages and streaks to calculate for each team
# we will try a variety of lengths to see which works best
HOME_VISITOR_ROLL_LIST = [3, 7, 10]  #lengths to use when restricting to home or visitor role
ALL_ROLL_LIST = [3, 7, 10, 15] #lengths to use when NOT restricting to home or visitor role



########### MODEL TRAINING CONSTANTS ##############

CATEGORY_COLUMNS = ['SEASON', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID' ]



########### WEB SCRAPING CONSTANTS ##############

DAYS = 2 #number of days back to scrape for games, set to >1 to catch up in case of a failed run

OFF_SEASON_START = 7 #month that the off-season starts, typically July
REGULAR_SEASON_START = 10 #month that the regular season starts, typically October
PLAYOFFS_START = 4 #month that the playoffs start, typically April

# columns from the nba.com boxscore table to dropped, either not used or already renamed to match our schema
NBA_COM_DROP_COLUMNS = ['Team', 'MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', '+/-',]



########## STREAMLIT CONSTANTS ############
# dictionary to convert team ids to team names
NBA_TEAMS_NAMES = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612754: "Indiana Pacers",
    1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}