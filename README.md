# NBA Game Predictor Project

## The goal of this project is to develop an NBA game-winner predictor model that can be used to develop profitable betting strategies. Initially, the focus will be on the predictor model, and then later, betting strategy models may be explored.

### Plan

Gradient boosted tree models (Xgboost and LightGBM) will be utilized to determine the probability that the home team will win each game. The probability of winning will be important in developing betting strategies because such strategies will not bet on every game, just on games with better expected values. The model will be deployed online using a streamlit app to predict and report wining probabilities every day. https://cmunch1-nba-prediction-streamlit-app-fs5l47.streamlit.app/


### Overview

 - Historical game data is retrieved from Kaggle.
 - EDA, Data Processing, and Feature Engineering are used to develop best model in either XGboost or LightGBM.
 - Data and model is added to serverless Feature Store and Model Registry
 - Model is deployed online as a Streamlit app
 - Pipelines are setup to:
 -- Scrape new data from NBA website and add to Feature Store every day using Github Actions
 -- Retrain model and tune hyperparameters

 Tools Used:

 - Pandas - data manipulation
 - XGboost - modeling
 - LightGBM - modeling
 - Optuna - hyperparamter tuning
 - Neptune.ai - experiment tracking
 - Selenium - data scraping and processing
 - ScrapingAnt - data scraping
 - BeautifulSoup - data processing of scraped data
 - Hopsworks.ai - Feature Store and Model Registry
 - Github Actions - running notebooks to scrape new data, predict winning probabilities, and retrain models
 - Streamlit - user interface


### Structure

Jupyter Notebooks were used for initial development and testing and are labeled 01 through 10 in the main directory. Notebooks 01 thru 06 are primarily just historical records and notes for the development process.

Key functions were moved to .py files in src directory once the functions were stable.

Notebooks 07, 09, and 10 are used in production.


### Data

Data from the 2013 thru 2021 season has been archived on Kaggle. New data is scraped from NBA website. 

Currently available data includes:

 - games_details.csv .. (each-game player stats for everyone on the roster)
 - games.csv .......... (each-game team stats: final scores, points scored, field-goal & free-throw percentages, etc...)
 - players.csv ........ (index of players' names and teams)
 - ranking.csv ........ (incremental daily record of standings, games played, won, lost, win%, home record, road record)
 - teams.csv .......... (index of team info such as city and arena names and also head coach) 
 
 NOTES 
 - games.csv is the primary data source and will be the only data used initially
 - games_details.csv details individual player stats for each game and may be added to the model later
 - ranking.csv data is essentially cumulative averages from the beginning of the season and is not really needed as these and other rolling averages can be calculated from the games.csv data 


**New Data**

New data is scraped from https://www.nba.com/stats/teams/boxscores.

 
**Data Leakage**

The data for each game are stats for the *completed* game. We want to predict the winner *before* the game is played, not after. The model should only use data that would be available before the game is played. Our model features will primarily be rolling stats for the previous games (e.g. average assists for previous 5 games) while excluding the current game.

If the goal is simply to predict which stats are important for winning games, then the model can be trained on the entire dataset. However, if the goal is to predict the winner of a game like we are trying to do, then the model must be trained on data that would only be available before the game is played.

### Train / Test Split
  
  - Latest season is used as Test data and previous seasons are used as Train data
  
### Baseline Models
  
Simple If-Then Models

 - Home team always wins (Accuracy = 0.59, AUC = 0.50 on Train data, Accuracy = 0.49, AUC = 0.50 on Test data
 
ML Models

 - LightGBM (Accuracy = 0.58, AUC = 0.64 on Test data)
 - XGBoost (Accuracy = 0.59, AUC = 0.61 on Test data)

### Feature Engineering

 - Covert game date to month only
 - Compile rolling means for various time periods for each team as home team and as visitor team 
 - Compile current win streak for each team as home team and as visitor team
 - Compile head-to-head matchup data for each team pair 
 - Compile rolling means for various time periods for each team regardless of home or visitor status
 - Compile current win streak for each team regardless of home or visitor status
 
### Model Testing

  Both LightGBM and XGBoost are used for testing.

  Notebook 07 integrates Neptune.ai for experiment tracking and Optuna for hyperparameter tuning.

  Experiment tracking logs can be viewed here: https://app.neptune.ai/cmunch1/nba-prediction/experiments?split=tbl&dash=charts&viewId=979e20ed-e172-4c33-8aae-0b1aa1af3602

### Production Features Pipeline

Notebook 09 is run from a Github Actions every morning.

- It scrapes the stats from the previous day's games and adds them to the Feature Store.
- It scrapes the upcoming game matchups for the current day and adds them to the Feature Store so that the streamlit app can use these to make it's daily predictions.

09a uses ScrapingAnt to scrape the data, while 09b uses Selenium. 

 - The Selenium notebook worked fine when ran locally, but there were issues when running the notebook in Github Actions, likely due to the ip address and anti-bot measures on the NBA website (which would require a proxy server to address)
 - ScrapingAnt is a cloud-based scraper with a Python API than handles the proxy server issues. An account is required, but the free account is sufficient for this project.

### Model Training Pipeline

Notebook 10 retrieves data from the Feature Store, trains the model, and adds the model to the Model Registry.

### Streamlit App

The streamlit app is deployed at streamlit.io and can be accessed here: https://cmunch1-nba-prediction-streamlit-app-fs5l47.streamlit.app/

It uses the model in the Model Registry to predict the win probability of the home team for the current day's upcoming games.