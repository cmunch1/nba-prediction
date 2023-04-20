import os

import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import json
import time
from datetime import timedelta, datetime
import xgboost as xgb

from pathlib import Path


print(f"Current directory: {Path.cwd()}")
print(f"Home directory: {Path.home()}")
print(f"Parent directory: {Path.cwd().parent}")

from hopsworks_utils import (
    convert_feature_names,
)

from feature_engineering import (
    fix_datatypes,
    remove_non_rolling,
)


# Load hopsworks API key from .env file

from dotenv import load_dotenv

load_dotenv()

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Set environment variable HOPSWORKS_API_KEY')



def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True )

def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model from the Hopsworks Model Registry."""

    mr = project.get_model_registry()
    # get best model based on custom metrics
    model = mr.get_best_model(model_name,
                                evaluation_metric,
                                sort_metrics_by)
    
    # download model from Hopsworks
    #model_dir = model.download()
    #print(model_dir)
    model_dir  = Path.cwd() / "models"
    with open(model_dir / "model.pkl", 'rb') as f:
        loaded_model = joblib.load(f)


    return loaded_model


# dictionary to convert team ids to team names

nba_team_names = {
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

# Streamlit app
st.title('NBA Prediction Project')
st.write(36 * "-")
st.write("")
st.write("This app uses a machine learning model to predict the winner of NBA games.")
st.write("")
st.write("Note: NBA season and postseason usually runs annually from October to June. There will be no games to predict outside of this time period.")

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")
fancy_header('\n📡 Connecting to Hopsworks Feature Store...')


# Connect to Hopsworks Feature Store and get Feature Group
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

rolling_stats_fg = fs.get_feature_group(
    name="rolling_stats",
    version=2,
)

st.write("Successfully connected!✔️")
progress_bar.progress(20)



# Get data from Feature Store
st.write(36 * "-")
fancy_header('\n☁️ Retrieving data from Feature Store...')


# pull games just for this season to get current games waiting prediction and additional games so that we show past performance
current_season = datetime.today().year
if datetime.today().month < 10:
    current_season = current_season - 1

ds_query = rolling_stats_fg.filter(rolling_stats_fg.season == current_season)
df_current_season = ds_query.read()

# get games for today that have not been played yet
df_todays_matches = df_current_season[df_current_season['pts_home'] == 0]

# select games that have been played
df_current_season = df_current_season[df_current_season['pts_home'] != 0]

# select last 25 games from the season
df_current_season = df_current_season.sort_values(by=['game_id'], ascending=False).head(25)


# if no games are scheduled for today, stop the app
if df_todays_matches.shape[0] == 0:
    progress_bar.progress(100)
    st.write()
    fancy_header('\n 🤷‍♂️ No games scheduled for today! 🤷‍♂️')
    st.write()
    st.write("Try again tomorrow!")
    st.write()
    st.write("NBA season and postseason usually runs from October to June.")
    st.stop()

today = datetime.today().strftime('%Y-%m-%d')
st.write("Successfully retrieved games for " + today + "!✔️")
progress_bar.progress(40)


# Prepare data for prediction
st.write(36 * "-")
fancy_header('\n☁️ Processing Data for prediction...')

# convert feature names back to mixed case
df_todays_matches = convert_feature_names(df_todays_matches)
df_current_season = convert_feature_names(df_current_season)

# Add a column that displays the matchup using the team names 
# this will make the display more meaningful
df_todays_matches['MATCHUP'] = df_todays_matches['VISITOR_TEAM_ID'].map(nba_team_names) + " @ " + df_todays_matches['HOME_TEAM_ID'].map(nba_team_names)
df_current_season['MATCHUP'] = df_current_season['VISITOR_TEAM_ID'].map(nba_team_names) + " @ " + df_current_season['HOME_TEAM_ID'].map(nba_team_names)

# fix date and other types
long_integer_fields = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']
short_integer_fields = ['PTS_home', 'AST_home', 'REB_home', 'PTS_away', 'AST_away', 'REB_away']
date_fields = ['GAME_DATE_EST']

df_todays_matches = fix_datatypes(df_todays_matches ,date_fields, short_integer_fields, long_integer_fields)
df_current_season = fix_datatypes(df_current_season ,date_fields, short_integer_fields, long_integer_fields)

# remove features not used by model
drop_columns = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', ] 
df_todays_matches = df_todays_matches.drop(drop_columns, axis=1)

# remove stats from today's games - these are blank (the game hasn't been played) and are not used by the model
use_columns = remove_non_rolling(df_todays_matches)
X = df_todays_matches[use_columns]

# MATCHUP is just for informational display, not used by model
X = X.drop('MATCHUP', axis=1) 


st.write(df_todays_matches['MATCHUP'])

st.write("Successfully processed!✔️")
progress_bar.progress(60)


# Load model from Hopsworks Model Registry
st.write(36 * "-")
fancy_header(f"Loading Best Model...")

model = get_model(project=project,
                  model_name="xgboost",
                  evaluation_metric="AUC",
                  sort_metrics_by="max")

st.write("Successfully loaded!✔️")
progress_bar.progress(70)



# Predict winning probabilities of home team
st.write(36 * "-")
fancy_header(f"Predicting Winning Probabilities...")


preds = model.predict_proba(X)[:,1]

df_todays_matches['HOME_TEAM_WIN_PROBABILITY'] = preds

st.dataframe(df_todays_matches[['MATCHUP', 'HOME_TEAM_WIN_PROBABILITY']])

progress_bar.progress(85)

# Show past performance
st.write(36 * "-")
fancy_header(f"Winning Probabilities and Results from last 25 games...")

X = df_current_season.drop(drop_columns, axis=1)
X = X[use_columns]
X = X.drop('MATCHUP', axis=1)
preds = model.predict_proba(X)[:,1]

df_current_season['HOME_TEAM_WIN_PROBABILITY'] = preds

#rename TARGET to HOME_WINS
df_current_season = df_current_season.rename(columns={'TARGET': 'HOME_WINS'})

# add column to show if prediction was correct
df_current_season['HOME_TEAM_WIN_PROBABILITY_INT'] = df_current_season['HOME_TEAM_WIN_PROBABILITY'].round().astype(int)
df_current_season['CORRECT_PREDICTION'] = df_current_season['HOME_TEAM_WIN_PROBABILITY_INT'] == df_current_season['HOME_WINS'] 

#sort by game date
df_current_season = df_current_season.sort_values(by=['GAME_DATE_EST'], ascending=False)

# format date
df_current_season["GAME_DATE_EST"] = df_current_season["GAME_DATE_EST"].dt.strftime('%Y-%m-%d')

st.dataframe(df_current_season[['GAME_DATE_EST','MATCHUP', 'HOME_TEAM_WIN_PROBABILITY', 'HOME_WINS', 'CORRECT_PREDICTION']])

# Show accuracy
st.write("Accuracy: " + str(df_current_season['CORRECT_PREDICTION'].sum() / df_current_season.shape[0]))


progress_bar.progress(100)


st.button("Re-run")
