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
    model_dir = model.download()
    model = joblib.load(model_dir + "/model.pkl")

    return model


st.title('NBA Prediction Project')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
st.write(36 * "-")
fancy_header('\n📡 Connecting to Hopsworks Feature Store...')

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

rolling_stats_fg = fs.get_feature_group(
    name="rolling_stats",
    version=1,
)


st.write("Successfully connected!✔️")
progress_bar.progress(20)

st.write(36 * "-")
fancy_header('\n☁️ Getting data from Feature Store...')

# filter new games that are scheduled for today
# the pipeline has saved these with a game_id starting at 20000001
ds_query = rolling_stats_fg.filter(rolling_stats_fg.game_id < 20000100)
df_todays_matches = ds_query.read()

progress_bar.progress(40)

# prepare data for prediction

df_todays_matches = convert_feature_names(df_todays_matches)

df_todays_matches = fix_datatypes(df_todays_matches)

drop_columns = ['TARGET', 'GAME_DATE_EST', 'GAME_ID', ] 
df_todays_matches = df_todays_matches.drop(drop_columns, axis=1)

use_columns = remove_non_rolling(df_todays_matches)

X = df_todays_matches[use_columns]


print(use_columns)

X_dmatrix = xgb.DMatrix(X)


progress_bar.progress(60)

st.write(36 * "-")
fancy_header(f"🗺 Predicting Win Probabilities...")



model = get_model(project=project,
                  model_name="xgboost",
                  evaluation_metric="AUC",
                  sort_metrics_by="max")

preds = model.predict(X_dmatrix)


print(preds)


df = pd.DataFrame(data=preds, columns=[f"Predictions for Today's Games"])

st.sidebar.write(df)
progress_bar.progress(100)
st.button("Re-run")