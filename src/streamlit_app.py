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


from hopsworks_utils import (
    convert_feature_names,
)

from feature_engineering import (
    fix_datatypes,
    remove_non_rolling,
)

from constants import (
    LONG_INTEGER_FIELDS,    
    SHORT_INTEGER_FIELDS,   
    DATE_FIELDS,            
    DROP_COLUMNS,
    NBA_TEAMS_NAMES,
    FEATURE_GROUP_VERSION
)


print(f"Current directory: {Path.cwd()}")
print(f"Home directory: {Path.home()}")
print(f"Parent directory: {Path.cwd().parent}")


# Load hopsworks API key from .env file

from dotenv import load_dotenv

load_dotenv()

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Set environment variable HOPSWORKS_API_KEY')


######################## Helper functions ########################

def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True )

def process_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for prediction.

    """
    
    # convert feature names back to mixed case
    df = convert_feature_names(df)
    
    # fix date and other types
    df = fix_datatypes(df, DATE_FIELDS, SHORT_INTEGER_FIELDS, LONG_INTEGER_FIELDS)

    # Add a column that displays the matchup using the team names 
    # this will make the display more meaningful
    df['MATCHUP'] = df['VISITOR_TEAM_ID'].map(NBA_TEAMS_NAMES) + " @ " + df['HOME_TEAM_ID'].map(NBA_TEAMS_NAMES)
        
    return df

def remove_unused_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that are not used in the model.

    """
    
    # remove stats from today's games - these are blank (the game hasn't been played) and are not used by the model
    use_columns = remove_non_rolling(df)
    X = df[use_columns]

    # drop columns not used in model
    X = X.drop(DROP_COLUMNS, axis=1)

    # MATCHUP is just for informational display, not used by model
    X = X.drop('MATCHUP', axis=1) 
    
    return X

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



############################### Streamlit app ##############################

st.title('NBA Prediction Project')
st.write(36 * "-")
st.write("This app uses a machine learning model to predict the winner of NBA games.")
st.write("")
st.write("This streamlit app demonstrates on-demand retrieval of data from the Hopsworks Feature Store, loading a model from the Hopsworks Model Registry, and making predictions.")
st.write(" - For the 2022-23 regular season (not playoffs), the current model would have an accuracy of 0.615.")
st.write(" - One of the best publicly available models achieved an accuracy of 0.656.")
st.write(" - A simple baseline model of 'home team always wins' would have an accuracy of 0.58.")
#st.write("*** THE CURRENT MODEL APPEARS TO SUCK FOR PLAYOFF GAMES. I'M WORKING ON IT. ***")
st.write("")
st.write("Note: NBA season and postseason usually runs annually from October to June. There will be no games to predict outside of this time period.")

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")
fancy_header('\n📡 Connecting to Hopsworks Feature Store...')


########### Connect to Hopsworks Feature Store and get Feature Group

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

rolling_stats_fg = fs.get_feature_group(
    name="rolling_stats",
    version=FEATURE_GROUP_VERSION,
)

st.write("Successfully connected!✔️")
progress_bar.progress(20)


########### Get data from Feature Store

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
#df_todays_matches = pd.DataFrame() # uncomment this line to test no games scheduled for today

# select games that have been played
df_current_season = df_current_season[df_current_season['pts_home'] != 0]

# select last 25 games from the season
df_current_season = df_current_season.sort_values(by=['game_id'], ascending=False).head(25)


# if no games are scheduled for today, write a message 
if df_todays_matches.shape[0] == 0:
    progress_bar.progress(40)
    st.write()
    fancy_header('\n 🤷‍♂️ No games scheduled for today! 🤷‍♂️')
    st.write()
    st.write("No games for today. Check out the last 25 games below")
    st.write()
    st.write("NBA season and postseason usually runs from October to June.")
    no_games = True
else:
    no_games = False
    today = datetime.today().strftime('%Y-%m-%d')
    st.write("Successfully retrieved games!✔️")
    progress_bar.progress(40)
    

########### Prepare data for prediction

st.write(36 * "-")
fancy_header('\n☁️ Processing Data for prediction...')

if no_games == False:
    df_todays_matches = process_for_prediction(df_todays_matches)
    df_todays_matches = df_todays_matches.reset_index(drop=True)
    df_todays_matches["GAME_DATE_EST"] = df_todays_matches["GAME_DATE_EST"].dt.strftime('%Y-%m-%d')
    st.write(df_todays_matches[['GAME_DATE_EST','MATCHUP']])
    st.write("Successfully processed data!✔️")

progress_bar.progress(60)


###########  Load model from Hopsworks Model Registry

st.write(36 * "-")
fancy_header(f"Loading Best Model...")

model = get_model(project=project,
                  model_name="xgboost",
                  evaluation_metric="AUC",
                  sort_metrics_by="max")

st.write("Successfully loaded!✔️")
progress_bar.progress(70)


########### Predict winning probabilities of home team

if no_games == False:
    
    st.write(36 * "-")
    fancy_header(f"Predicting Winning Probabilities...")

    X = remove_unused_features(df_todays_matches)

    preds = model.predict_proba(X)[:,1]

    df_todays_matches['HOME_TEAM_WIN_PROBABILITY'] = preds

    df_todays_matches = df_todays_matches.reset_index(drop=True)
    st.dataframe(df_todays_matches[['GAME_DATE_EST', 'MATCHUP', 'HOME_TEAM_WIN_PROBABILITY']])

progress_bar.progress(85)


########### Show past performance

st.write(36 * "-")
fancy_header(f"Preparing Winning Probabilities and Results from last 25 games...")

df_current_season = process_for_prediction(df_current_season)

X = remove_unused_features(df_current_season)

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

# clean up display
df_current_season = df_current_season.rename(columns={'GAME_DATE_EST': 'GAME_DATE', 'HOME_TEAM_WIN_PROBABILITY': 'HOME_WIN_PROB', 'CORRECT_PREDICTION': 'CORRECT'})
df_current_season = df_current_season.reset_index(drop=True)
st.dataframe(df_current_season[['GAME_DATE','MATCHUP', 'HOME_WIN_PROB', 'HOME_WINS', 'CORRECT']])

# Show accuracy
st.write("Accuracy: " + str(df_current_season['CORRECT'].sum() / df_current_season.shape[0]))


progress_bar.progress(100)


st.button("Re-run")
