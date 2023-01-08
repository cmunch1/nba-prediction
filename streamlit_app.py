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

from src.hopsworks_utils import (
    convert_feature_names,
)

from src.feature_engineering import (
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
    
    model = joblib.load("model.pkl")

    return model


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
    1610612746: "Indiana Pacers",
    1610612747: "LA Clippers",
    1610612748: "Los Angeles Lakers",
    1610612749: "Memphis Grizzlies",
    1610612750: "Miami Heat",
    1610612751: "Milwaukee Bucks",
    1610612752: "Minnesota Timberwolves",
    1610612753: "Brooklyn Nets",
    1610612754: "New York Knicks",
    1610612755: "Orlando Magic",
    1610612756: "Philadelphia 76ers",
    1610612757: "Phoenix Suns",
    1610612758: "Portland Trail Blazers",
    1610612759: "Sacramento Kings",
    1610612760: "San Antonio Spurs",
    1610612761: "Oklahoma City Thunder",
    1610612762: "Toronto Raptors",
    1610612763: "Utah Jazz",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}

# Streamlit app
st.title('NBA Prediction Project')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")
fancy_header('\n📡 Connecting to Hopsworks Feature Store...')


# Connect to Hopsworks Feature Store and get Feature Group
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

rolling_stats_fg = fs.get_feature_group(
    name="rolling_stats",
    version=1,
)

st.write("Successfully connected!✔️")
progress_bar.progress(20)



# Get data from Feature Store
st.write(36 * "-")
fancy_header('\n☁️ Retrieving data from Feature Store...')

# filter new games that are scheduled for today
# these are games where no points have been scored yet
ds_query = rolling_stats_fg.filter(rolling_stats_fg.pts_home == 0)
df_todays_matches = ds_query.read()

st.write("Successfully retrieved!✔️")
progress_bar.progress(40)



# Prepare data for prediction
st.write(36 * "-")
fancy_header('\n☁️ Processing Data for prediction...')

# convert feature names back to mixed case
df_todays_matches = convert_feature_names(df_todays_matches)

# Add a column that displays the matchup using the team names 
# this will make the display more meaningful
df_todays_matches['MATCHUP'] = df_todays_matches['VISITOR_TEAM_ID'].map(nba_team_names) + " @ " + df_todays_matches['HOME_TEAM_ID'].map(nba_team_names)

# fix date and other types
df_todays_matches = fix_datatypes(df_todays_matches)

# remove features not used by model
drop_columns = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', ] 
df_todays_matches = df_todays_matches.drop(drop_columns, axis=1)

# remove stats from today's games - these are blank (the game hasn't been played) and are not used by the model
use_columns = remove_non_rolling(df_todays_matches)
X = df_todays_matches[use_columns]

# MATCHUP is just for informational display, not used by model
X = X.drop('MATCHUP', axis=1) 

X_dmatrix = xgb.DMatrix(X) # convert to DMatrix for XGBoost

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
progress_bar.progress(80)



# Predict winning probabilities of home team
st.write(36 * "-")
fancy_header(f"Predicting Winning Probabilities...")

# https://discuss.streamlit.io/t/frustrated-streamlit-frontend-disconnects-from-apps-with-long-running-processes/11612/22?page=2

preds = model.predict(X_dmatrix)

df_todays_matches['HOME_TEAM_WIN_PROBABILITY'] = preds

st.dataframe(df_todays_matches[['MATCHUP', 'HOME_TEAM_WIN_PROBABILITY']])

progress_bar.progress(100)
st.button("Re-run")
