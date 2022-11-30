import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import json
import time
from datetime import timedelta, datetime
from branca.element import Figure


def fancy_header(text, font_size=24):
    res = f'<span style="color:#ff5f27; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True )

def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model.pkl"
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
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

project = hopsworks.login()
fs = project.get_feature_store()
feature_view = fs.get_feature_view(
    name = 'air_quality_fv',
    version = 1
)

st.write("Successfully connected!✔️")
progress_bar.progress(20)

st.write(36 * "-")
fancy_header('\n☁️ Getting batch data from Feature Store...')

start_date = datetime.now() - timedelta(days=1)
start_time = int(start_date.timestamp()) * 1000

X = feature_view.get_batch_data(start_time=start_time)
progress_bar.progress(50)

latest_date_unix = str(X.date.values[0])[:10]
latest_date = time.ctime(int(latest_date_unix))

st.write(f"⏱ Data for {latest_date}")

X = X.drop(columns=["date"]).fillna(0)

data_to_display = decode_features(X, feature_view=feature_view)

progress_bar.progress(60)

st.write(36 * "-")
fancy_header(f"🗺 Processing the map...")








progress_bar.progress(80)
st.sidebar.write("-" * 36)


model = get_model(project=project,
                  model_name="gradient_boost_model",
                  evaluation_metric="f1_score",
                  sort_metrics_by="max")

preds = model.predict(X)



next_day_date = datetime.today() + timedelta(days=1)
next_day = next_day_date.strftime ('%d/%m/%Y')
df = pd.DataFrame(data=preds, index=cities, columns=[f"AQI Predictions for {next_day}"], dtype=int)

st.sidebar.write(df)
progress_bar.progress(100)
st.button("Re-run")