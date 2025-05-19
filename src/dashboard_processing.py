"""
NBA Prediction Data Processor for Dashboards

This module processes NBA game data and generates prediction outputs
suitable for visualization in dashboarding tools.
It replaces the streamlit-specific functionality while maintaining
the core data preparation and prediction logic.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import helper modules 
try:
    from src.feature_engineering import fix_datatypes, remove_non_rolling
    from src.constants import (
        LONG_INTEGER_FIELDS,    
        SHORT_INTEGER_FIELDS,   
        DATE_FIELDS,            
        DROP_COLUMNS,
        NBA_TEAMS_NAMES,
        FEATURE_GROUP_VERSION
    )
except ImportError as e:
    logger.error(f"Error importing helper modules: {e}")
    raise

# Constants
DATAPATH = Path('data')
MODEL_PATH = Path('models')

class NBADataProcessor:
    """Processes NBA game data for prediction and visualization in Tableau."""
    
    def __init__(self, data_path=DATAPATH, model_path=MODEL_PATH):
        """Initialize the data processor with paths to data and models."""
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        logger.info(f"Initialized NBADataProcessor with data_path={data_path}, model_path={model_path}")
    
    def load_data(self, filepath='games_engineered.csv'):
        """Load the game data from CSV."""
        full_path = self.data_path / filepath
        logger.info(f"Loading data from {full_path}")
        
        try:
            df = pd.read_csv(full_path)
            logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_model(self, model_file='model.pkl'):
        """Load the prediction model."""
        full_path = self.model_path / model_file
        logger.info(f"Loading model from {full_path}")
        
        try:
            with open(full_path, 'rb') as f:
                self.model = joblib.load(f)
            logger.info(f"Successfully loaded model")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def process_for_prediction(self, df):
        """Prepare data for prediction by fixing data types and adding matchup information."""
        logger.info("Processing data for prediction")
        
        # Fix date and other types
        df = fix_datatypes(df, DATE_FIELDS, SHORT_INTEGER_FIELDS, LONG_INTEGER_FIELDS)
        
        # Add a column that displays the matchup using the team names
        df['MATCHUP'] = df['VISITOR_TEAM_ID'].map(NBA_TEAMS_NAMES) + " @ " + df['HOME_TEAM_ID'].map(NBA_TEAMS_NAMES)
        
        return df
    
    def remove_unused_features(self, df):
        """Remove features that are not used in the model."""
        logger.info("Removing unused features")
        
        # Remove stats from today's games - these are blank (the game hasn't been played) and are not used by the model
        use_columns = remove_non_rolling(df)
        X = df[use_columns]
        
        # Drop columns not used in model
        X = X.drop(DROP_COLUMNS, axis=1)
        
        # MATCHUP is just for informational display, not used by model
        if 'MATCHUP' in X.columns:
            X = X.drop('MATCHUP', axis=1)
        
        return X
    
    def make_predictions(self, df):
        """Make predictions for home team win probability."""
        logger.info("Making predictions")
        
        if self.model is None:
            logger.warning("Model not loaded, loading now...")
            self.load_model()
        
        X = self.remove_unused_features(df)
        
        try:
            preds = self.model.predict_proba(X)[:, 1]
            logger.info(f"Successfully made predictions for {len(preds)} games")
            return preds
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def prepare_todays_games(self):
        """Prepare prediction data for today's games."""
        logger.info("Preparing today's games data")
        
        # Load data
        df_all = self.load_data()
        
        # Select current season
        current_season = datetime.today().year
        if datetime.today().month < 10:
            current_season = current_season - 1
        
        logger.info(f"Filtering for season: {current_season}")
        df_current_season = df_all[df_all['SEASON'] == current_season]
        
        # Get games for today that have not been played yet
        df_todays_matches = df_current_season[df_current_season['PTS_home'] == 0]
        
        if df_todays_matches.shape[0] == 0:
            logger.info("No games scheduled for today")
            return None
        
        # Process data
        df_todays_matches = self.process_for_prediction(df_todays_matches)
        
        # Make predictions
        preds = self.make_predictions(df_todays_matches)
        
        # Add predictions to dataframe
        df_todays_matches['HOME_TEAM_WIN_PROBABILITY'] = preds
        
        # Format date for display
        if 'GAME_DATE_EST' in df_todays_matches.columns:
            if pd.api.types.is_datetime64_any_dtype(df_todays_matches['GAME_DATE_EST']):
                df_todays_matches["GAME_DATE_EST"] = df_todays_matches["GAME_DATE_EST"].dt.strftime('%Y-%m-%d')
            else:
                # Convert string to datetime if needed
                df_todays_matches["GAME_DATE_EST"] = pd.to_datetime(df_todays_matches["GAME_DATE_EST"]).dt.strftime('%Y-%m-%d')

        # Clean up display for game data
        df_todays_matches = df_todays_matches.rename(columns={
            'GAME_DATE_EST': 'GAME_DATE', 
            'HOME_TEAM_WIN_PROBABILITY': 'HOME_WIN_PROB', 
            'CORRECT_PREDICTION': 'CORRECT'
        })
        
        logger.info(f"Prepared predictions for {df_todays_matches.shape[0]} games today")
        return df_todays_matches
    
    def calculate_daily_running_accuracy(self, games_df):
        """
        Calculate running accuracy for each team, home teams, away teams, and all teams combined
        on a daily basis.
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            DataFrame with team accuracy and running accuracy metrics
        """
        logger.info("Calculating daily running accuracy metrics")
        
        # Ensure the dataframe is sorted by date
        games_df = games_df.copy()
        
        # Convert GAME_DATE to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(games_df['GAME_DATE']):
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        # Sort by date
        games_df = games_df.sort_values('GAME_DATE')
        
        # Create copies of the dataframe for home and away team calculations
        home_games = games_df.copy()
        away_games = games_df.copy()
        
        # Rename columns for consistency when we combine the data
        home_games = home_games.rename(columns={
            'HOME_TEAM_ID': 'TEAM_ID',
            'HOME_WINS': 'ACTUAL_WIN',
            'HOME_WIN_PROB': 'PREDICTED_WIN_PROB'
        })
        home_games['IS_HOME'] = True
        
        # For away teams, we need to invert some values
        away_games = away_games.rename(columns={
            'VISITOR_TEAM_ID': 'TEAM_ID'
        })
        # Away team wins when home team loses
        away_games['ACTUAL_WIN'] = 1 - away_games['HOME_WINS']
        # Away team win probability is inverse of home team win probability
        away_games['PREDICTED_WIN_PROB'] = 1 - away_games['HOME_WIN_PROB']
        away_games['IS_HOME'] = False
        
        # Select only the columns we need
        columns_to_keep = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'ACTUAL_WIN', 
                          'PREDICTED_WIN_PROB', 'CORRECT', 'IS_HOME']
        
        home_games = home_games[columns_to_keep]
        away_games = away_games[columns_to_keep]
        
        # Combine home and away data
        all_team_games = pd.concat([home_games, away_games], ignore_index=True)
        
        # Convert team IDs to team names
        all_team_games['TEAM_NAME'] = all_team_games['TEAM_ID'].map(NBA_TEAMS_NAMES)
        
        # Sort by date first, then by team
        all_team_games = all_team_games.sort_values(['GAME_DATE', 'TEAM_NAME'])
        
        # Create daily running accuracy dataframes
        
        # 1. Team-specific daily running accuracy
        team_daily_accuracy = self._calculate_team_daily_accuracy(all_team_games)
        
        # 2. Home/Away split daily running accuracy
        home_away_daily_accuracy = self._calculate_home_away_daily_accuracy(all_team_games)
        
        # 3. Overall daily running accuracy
        overall_daily_accuracy = self._calculate_overall_daily_accuracy(all_team_games)
        
        # Create a lookup map for each team's running accuracy on each date
        team_accuracy_lookup = self._create_team_accuracy_lookup(team_daily_accuracy)
        home_accuracy_lookup = self._create_role_accuracy_lookup(home_away_daily_accuracy, is_home=True)
        away_accuracy_lookup = self._create_role_accuracy_lookup(home_away_daily_accuracy, is_home=False)
        overall_accuracy_lookup = self._create_overall_accuracy_lookup(overall_daily_accuracy)
        
        # Return the lookup dictionaries that will be used to add accuracy to the games dataframe
        return {
            'team_accuracy': team_accuracy_lookup,
            'home_accuracy': home_accuracy_lookup,
            'away_accuracy': away_accuracy_lookup,
            'overall_accuracy': overall_accuracy_lookup
        }
    
    def _calculate_team_daily_accuracy(self, all_team_games):
        """Calculate daily running accuracy for each team."""
        logger.info("Calculating team-specific daily running accuracy")
        
        # Create an empty list to store team data
        team_daily_records = []
        
        # Process each team separately
        for team_id in all_team_games['TEAM_ID'].unique():
            team_data = all_team_games[all_team_games['TEAM_ID'] == team_id].copy()
            
            # Sort by date
            team_data = team_data.sort_values('GAME_DATE')
            
            # Calculate running correct predictions and games played
            team_data['TEAM_GAMES_PLAYED'] = range(1, len(team_data) + 1)
            team_data['TEAM_CORRECT_CUMULATIVE'] = team_data['CORRECT'].cumsum()
            team_data['TEAM_RUNNING_ACCURACY'] = team_data['TEAM_CORRECT_CUMULATIVE'] / team_data['TEAM_GAMES_PLAYED']
            
            # Keep only necessary columns
            team_daily = team_data[['GAME_DATE', 'TEAM_ID', 'TEAM_NAME', 'TEAM_RUNNING_ACCURACY']].copy()
            
            team_daily_records.append(team_daily)
        
        # Combine all team records
        if team_daily_records:
            return pd.concat(team_daily_records, ignore_index=True)
        else:
            return pd.DataFrame(columns=['GAME_DATE', 'TEAM_ID', 'TEAM_NAME', 'TEAM_RUNNING_ACCURACY'])
    
    def _calculate_home_away_daily_accuracy(self, all_team_games):
        """Calculate daily running accuracy for home and away teams."""
        logger.info("Calculating home/away daily running accuracy")
        
        # Create empty lists to store home and away records
        home_daily_records = []
        away_daily_records = []
        
        # Sort data by date
        all_team_games = all_team_games.sort_values('GAME_DATE')
        
        # Process home games
        home_games = all_team_games[all_team_games['IS_HOME'] == True].copy()
        # Calculate running correct predictions and games played for home games
        running_correct_home = 0
        running_games_home = 0
        
        # Process away games
        away_games = all_team_games[all_team_games['IS_HOME'] == False].copy()
        # Calculate running correct predictions and games played for away games
        running_correct_away = 0
        running_games_away = 0
        
        # Get unique dates in sorted order
        dates = sorted(all_team_games['GAME_DATE'].unique())
        
        for date in dates:
            # Process home games for this date
            home_games_on_date = home_games[home_games['GAME_DATE'] == date]
            if not home_games_on_date.empty:
                running_correct_home += home_games_on_date['CORRECT'].sum()
                running_games_home += len(home_games_on_date)
                home_accuracy = running_correct_home / running_games_home if running_games_home > 0 else 0
                
                home_daily_records.append({
                    'GAME_DATE': date,
                    'IS_HOME': True,
                    'ROLE': 'HOME',
                    'ROLE_RUNNING_ACCURACY': home_accuracy,
                    'ROLE_GAMES_PLAYED': running_games_home,
                    'ROLE_CORRECT_PREDICTIONS': running_correct_home
                })
            
            # Process away games for this date
            away_games_on_date = away_games[away_games['GAME_DATE'] == date]
            if not away_games_on_date.empty:
                running_correct_away += away_games_on_date['CORRECT'].sum()
                running_games_away += len(away_games_on_date)
                away_accuracy = running_correct_away / running_games_away if running_games_away > 0 else 0
                
                away_daily_records.append({
                    'GAME_DATE': date,
                    'IS_HOME': False,
                    'ROLE': 'AWAY',
                    'ROLE_RUNNING_ACCURACY': away_accuracy,
                    'ROLE_GAMES_PLAYED': running_games_away,
                    'ROLE_CORRECT_PREDICTIONS': running_correct_away
                })
        
        # Combine home and away records
        daily_records = home_daily_records + away_daily_records
        
        if daily_records:
            return pd.DataFrame(daily_records)
        else:
            return pd.DataFrame(columns=[
                'GAME_DATE', 'IS_HOME', 'ROLE', 'ROLE_RUNNING_ACCURACY', 
                'ROLE_GAMES_PLAYED', 'ROLE_CORRECT_PREDICTIONS'
            ])

    def _calculate_overall_daily_accuracy(self, all_team_games):
        """Calculate overall daily running accuracy across all teams and games."""
        logger.info("Calculating overall daily running accuracy")
        
        # Create an empty list to store daily records
        daily_records = []
        
        # Sort data by date
        all_team_games = all_team_games.sort_values('GAME_DATE')
        
        # Since each game appears twice (once for each team), we need to deduplicate
        # to get actual game counts and correct prediction counts
        
        # Get unique game IDs and dates
        unique_games = all_team_games[['GAME_ID', 'GAME_DATE']].drop_duplicates()
        
        # Initialize counters for running totals
        running_correct = 0
        running_games = 0
        
        # Get unique dates in sorted order
        dates = sorted(unique_games['GAME_DATE'].unique())
        
        for date in dates:
            # Get games for this date
            games_on_date = unique_games[unique_games['GAME_DATE'] == date]
            game_ids_on_date = games_on_date['GAME_ID'].unique()
            
            # Get all rows for these games
            games_data = all_team_games[all_team_games['GAME_ID'].isin(game_ids_on_date)]
            
            # Each game appears twice, so we need to count unique games
            # We can use either home or away team data, but not both
            home_team_games = games_data[games_data['IS_HOME'] == True]
            
            # Add to running totals
            running_correct += home_team_games['CORRECT'].sum()
            running_games += len(home_team_games)
            
            # Calculate running accuracy
            running_accuracy = running_correct / running_games if running_games > 0 else 0
            
            daily_records.append({
                'GAME_DATE': date,
                'OVERALL_RUNNING_ACCURACY': running_accuracy,
                'OVERALL_GAMES_PLAYED': running_games,
                'OVERALL_CORRECT_PREDICTIONS': running_correct
            })
        
        if daily_records:
            return pd.DataFrame(daily_records)
        else:
            return pd.DataFrame(columns=[
                'GAME_DATE', 'OVERALL_RUNNING_ACCURACY', 
                'OVERALL_GAMES_PLAYED', 'OVERALL_CORRECT_PREDICTIONS'
            ])    
        
    def _create_team_accuracy_lookup(self, team_daily_accuracy):
            """Create a lookup dictionary for team running accuracy by date and team ID."""
            lookup = {}
            
            for _, row in team_daily_accuracy.iterrows():
                team_id = row['TEAM_ID']
                game_date = row['GAME_DATE']
                accuracy = row['TEAM_RUNNING_ACCURACY']
                
                if team_id not in lookup:
                    lookup[team_id] = {}
                
                lookup[team_id][game_date] = accuracy
            
            return lookup
    
    def _create_role_accuracy_lookup(self, home_away_daily_accuracy, is_home=True):
        """Create a lookup dictionary for home/away running accuracy by date."""
        lookup = {}
        
        role_data = home_away_daily_accuracy[home_away_daily_accuracy['IS_HOME'] == is_home]
        
        for _, row in role_data.iterrows():
            game_date = row['GAME_DATE']
            accuracy = row['ROLE_RUNNING_ACCURACY']
            lookup[game_date] = accuracy
        
        return lookup
    
    def _create_overall_accuracy_lookup(self, overall_daily_accuracy):
        """Create a lookup dictionary for overall running accuracy by date."""
        lookup = {}
        
        for _, row in overall_daily_accuracy.iterrows():
            game_date = row['GAME_DATE']
            accuracy = row['OVERALL_RUNNING_ACCURACY']
            lookup[game_date] = accuracy
        
        return lookup
    
    def _add_running_accuracy_to_games(self, games_df, accuracy_lookups):
        """Add running accuracy metrics to the games dataframe."""
        logger.info("Adding running accuracy metrics to games dataframe")
        
        # Create a copy of the dataframe
        df = games_df.copy()
        
        # Ensure GAME_DATE is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Add team-specific running accuracy
        df['HOME_TEAM_RUNNING_ACCURACY'] = df.apply(
            lambda row: accuracy_lookups['team_accuracy'].get(row['HOME_TEAM_ID'], {}).get(row['GAME_DATE'], None), 
            axis=1
        )
        
        df['VISITOR_TEAM_RUNNING_ACCURACY'] = df.apply(
            lambda row: accuracy_lookups['team_accuracy'].get(row['VISITOR_TEAM_ID'], {}).get(row['GAME_DATE'], None), 
            axis=1
        )
        
        # Add home/away role running accuracy
        df['HOME_ROLE_RUNNING_ACCURACY'] = df['GAME_DATE'].map(accuracy_lookups['home_accuracy'])
        df['AWAY_ROLE_RUNNING_ACCURACY'] = df['GAME_DATE'].map(accuracy_lookups['away_accuracy'])
        
        # Add overall running accuracy
        df['OVERALL_RUNNING_ACCURACY'] = df['GAME_DATE'].map(accuracy_lookups['overall_accuracy'])
        
        return df
    
    def calculate_team_accuracy(self, games_df):
        """
        Calculate prediction accuracy for each team.
        Note: This method is maintained for backward compatibility.
        The new method calculate_daily_running_accuracy() should be used for
        calculating running accuracy metrics.
        """
        logger.info("Calculating team-specific prediction accuracy (legacy method)")
        
        # Create copies of the dataframe for home and away team calculations
        home_games = games_df.copy()
        away_games = games_df.copy()
        
        # Rename columns for consistency when we combine the data
        home_games = home_games.rename(columns={
            'HOME_TEAM_ID': 'TEAM_ID',
            'HOME_WINS': 'ACTUAL_WIN',
            'HOME_WIN_PROB': 'PREDICTED_WIN_PROB'
        })
        home_games['IS_HOME'] = True
        
        # For away teams, we need to invert some values
        away_games = away_games.rename(columns={
            'VISITOR_TEAM_ID': 'TEAM_ID'
        })
        # Away team wins when home team loses
        away_games['ACTUAL_WIN'] = 1 - away_games['HOME_WINS']
        # Away team win probability is inverse of home team win probability
        away_games['PREDICTED_WIN_PROB'] = 1 - away_games['HOME_WIN_PROB']
        away_games['IS_HOME'] = False
        
        # Select only the columns we need
        columns_to_keep = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'ACTUAL_WIN', 
                          'PREDICTED_WIN_PROB', 'CORRECT', 'IS_HOME']
        
        home_games = home_games[columns_to_keep]
        away_games = away_games[columns_to_keep]
        
        # Combine home and away data
        all_team_games = pd.concat([home_games, away_games], ignore_index=True)
        
        # Convert team IDs to team names
        all_team_games['TEAM_NAME'] = all_team_games['TEAM_ID'].map(NBA_TEAMS_NAMES)
        
        # Sort by team and date
        all_team_games = all_team_games.sort_values(['TEAM_NAME', 'GAME_DATE'])
        
        # Calculate running accuracy for each team
        team_accuracy = []
        
        for team in all_team_games['TEAM_NAME'].unique():
            team_data = all_team_games[all_team_games['TEAM_NAME'] == team].copy()
            
            # Calculate cumulative correct predictions and games played
            team_data['GAMES_PLAYED'] = range(1, len(team_data) + 1)
            team_data['CORRECT_CUMULATIVE'] = team_data['CORRECT'].cumsum()
            team_data['ACCURACY_CUMULATIVE'] = team_data['CORRECT_CUMULATIVE'] / team_data['GAMES_PLAYED']
            
            # Calculate overall team accuracy
            total_correct = team_data['CORRECT'].sum()
            total_games = len(team_data)
            overall_accuracy = total_correct / total_games if total_games > 0 else 0
            
            # Add overall team accuracy as a column
            team_data['TEAM_OVERALL_ACCURACY'] = overall_accuracy
            
            team_accuracy.append(team_data)
        
        # Combine all team data
        team_accuracy_df = pd.concat(team_accuracy, ignore_index=True)
        
        logger.info(f"Calculated team accuracy for {len(team_accuracy_df['TEAM_NAME'].unique())} teams")
        
        return team_accuracy_df
    
    def prepare_processed_games_data(self, num_recent_games=25):
        """
        Process completed games data and return both detailed game data and summary statistics.
        
        Args:
            num_recent_games: Number of recent games to flag as "recent"
            
        Returns:
            tuple: (processed_games_df, season_summary_df, team_accuracy_df)
        """
        logger.info(f"Preparing processed games data")
        
        # Load data
        df_all = self.load_data()
        
        # Select current season
        current_season = datetime.today().year
        if datetime.today().month < 10:
            current_season = current_season - 1
        
        logger.info(f"Filtering for season: {current_season}")
        df_current_season = df_all[df_all['SEASON'] == current_season]
        
        # Select games that have been played
        df_completed_games = df_current_season[df_current_season['PTS_home'] != 0]
        
        # Process data
        df_completed_games = self.process_for_prediction(df_completed_games)
        
        # Make predictions
        preds = self.make_predictions(df_completed_games)
        
        # Add predictions to dataframe
        df_completed_games['HOME_TEAM_WIN_PROBABILITY'] = preds
        
        # Rename TARGET to HOME_WINS
        df_completed_games = df_completed_games.rename(columns={'TARGET': 'HOME_WINS'})
        
        # Add column to show if prediction was correct
        df_completed_games['HOME_TEAM_WIN_PROBABILITY_INT'] = df_completed_games['HOME_TEAM_WIN_PROBABILITY'].round().astype(int)
        df_completed_games['CORRECT_PREDICTION'] = df_completed_games['HOME_TEAM_WIN_PROBABILITY_INT'] == df_completed_games['HOME_WINS']
        
        # Format date for display
        if 'GAME_DATE_EST' in df_completed_games.columns:
            if pd.api.types.is_datetime64_any_dtype(df_completed_games['GAME_DATE_EST']):
                # Remove timezone and format as YYYY-MM-DD
                df_completed_games["GAME_DATE_EST"] = pd.to_datetime(df_completed_games["GAME_DATE_EST"]).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        # Clean up display for game data
        processed_games = df_completed_games.rename(columns={
            'GAME_DATE_EST': 'GAME_DATE', 
            'HOME_TEAM_WIN_PROBABILITY': 'HOME_WIN_PROB', 
            'CORRECT_PREDICTION': 'CORRECT'
        })
        
        # Add a flag for recent games
        processed_games['RECENT_FLAG'] = False
        recent_indices = processed_games.sort_values(
            by=['GAME_DATE', 'GAME_ID'], 
            ascending=[False, False]
        ).head(num_recent_games).index
        processed_games.loc[recent_indices, 'RECENT_FLAG'] = True
        
        # Calculate running accuracy metrics
        # Make sure GAME_DATE is in datetime format
        processed_games['GAME_DATE'] = pd.to_datetime(processed_games['GAME_DATE'])
        
        # Calculate running accuracy lookup tables
        accuracy_lookups = self.calculate_daily_running_accuracy(processed_games)
        
        # Add running accuracy metrics to the games dataframe
        processed_games_with_accuracy = self._add_running_accuracy_to_games(processed_games, accuracy_lookups)
        
        # For backward compatibility, also calculate team accuracy using the old method
        legacy_team_accuracy = self.calculate_team_accuracy(processed_games)
        
        # Calculate summary statistics
        total_games = processed_games.shape[0]
        correct_predictions = processed_games['CORRECT'].sum()
        accuracy = correct_predictions / total_games if total_games > 0 else 0
        
        summary = {
            'SEASON': current_season,
            'TOTAL_GAMES': total_games,
            'CORRECT_PREDICTIONS': correct_predictions,
            'ACCURACY': accuracy,
            'HOME_TEAM_WINS': processed_games['HOME_WINS'].sum(),
            'HOME_TEAM_WIN_PCT': processed_games['HOME_WINS'].mean() if total_games > 0 else 0,
        }
        
        logger.info(f"Processed {total_games} completed games with {correct_predictions} correct predictions")
        
        return processed_games_with_accuracy, pd.DataFrame([summary]), legacy_team_accuracy
    
    # Backward compatibility methods
    def prepare_recent_games(self, num_games=25):
        """Prepare data for recently completed games (backward compatibility method)."""
        games_df, _, _ = self.prepare_processed_games_data(num_recent_games=num_games)
        # Filter to only include the recent games
        recent_games = games_df[games_df['RECENT_FLAG']].copy()
        logger.info(f"Prepared data for {recent_games.shape[0]} recent games (via compatibility method)")
        return recent_games
    
    def prepare_season_summary(self):
        """Prepare summary statistics for the current season (backward compatibility method)."""
        _, summary_df, _ = self.prepare_processed_games_data()
        logger.info(f"Season summary prepared (via compatibility method)")
        return summary_df
    
    def filter_dashboard_columns(self, games_df):
        """
        Filter the games dataframe to keep only columns needed for the dashboard.
        
        Args:
            games_df: DataFrame with full game data
            
        Returns:
            DataFrame with only the columns needed for the dashboard
        """
        logger.info("Filtering columns for dashboard")
        
        # Define columns to keep for the dashboard
        dashboard_columns = [
            # Game identification
            'GAME_ID', 
            'GAME_DATE', 
            'SEASON',
            'GAME_STATUS',  # 'Upcoming' or 'Completed'
            
            # Team information
            'HOME_TEAM_ID', 
            'VISITOR_TEAM_ID',
            'MATCHUP',  # Display-friendly matchup text
            
            # Game results (for completed games)
            'HOME_WINS',  # 1 if home team won, 0 if away team won
            'PTS_home',   # Home team points
            'PTS_away',   # Away team points
            
            # Prediction information
            'HOME_WIN_PROB',  # Probability that home team wins
            'CORRECT',        # Whether prediction was correct (for completed games)
            
            # Running accuracy metrics
            'HOME_TEAM_RUNNING_ACCURACY',    # Running accuracy for home team
            'VISITOR_TEAM_RUNNING_ACCURACY', # Running accuracy for visiting team
            'HOME_ROLE_RUNNING_ACCURACY',    # Running accuracy for all home teams
            'AWAY_ROLE_RUNNING_ACCURACY',    # Running accuracy for all away teams
            'OVERALL_RUNNING_ACCURACY',      # Running accuracy for all teams
            
            # Flag for recent games
            'RECENT_FLAG'  # True for most recent games
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in dashboard_columns if col in games_df.columns]
        
        # Create a copy with only the needed columns
        df_filtered = games_df[available_columns].copy()
        
        # Add team name columns based on ID mapping
        df_filtered['HOME_TEAM_NAME'] = df_filtered['HOME_TEAM_ID'].map(NBA_TEAMS_NAMES)
        df_filtered['VISITOR_TEAM_NAME'] = df_filtered['VISITOR_TEAM_ID'].map(NBA_TEAMS_NAMES)
        
        # For upcoming games, ensure prediction-related columns exist
        if 'CORRECT' not in df_filtered.columns:
            df_filtered['CORRECT'] = None
        
        if 'HOME_WINS' not in df_filtered.columns:
            df_filtered['HOME_WINS'] = None
        
        # Calculate score difference for completed games
        if 'PTS_home' in df_filtered.columns and 'PTS_away' in df_filtered.columns:
            mask = (df_filtered['GAME_STATUS'] == 'Completed') & (~df_filtered['PTS_home'].isna()) & (~df_filtered['PTS_away'].isna())
            df_filtered.loc[mask, 'SCORE_DIFF'] = df_filtered.loc[mask, 'PTS_home'] - df_filtered.loc[mask, 'PTS_away']
        
        # Calculate prediction confidence (how far from 0.5 the probability is)
        if 'HOME_WIN_PROB' in df_filtered.columns:
            df_filtered['PREDICTION_CONFIDENCE'] = abs(df_filtered['HOME_WIN_PROB'] - 0.5)
        
        logger.info(f"Filtered dataframe from {len(games_df.columns)} to {len(df_filtered.columns)} columns")
        
        return df_filtered
    
    def _calculate_weekly_averages(self, completed_games):
        """
        Calculate 7-day rolling accuracy averages for all games.
        Starting with the first game of the season, creates weekly buckets
        and calculates average accuracy for each 7-day period.
        
        Args:
            completed_games: DataFrame with completed games data
            
        Returns:
            DataFrame with weekly average metrics
        """
        logger.info("Calculating weekly (7-day) average accuracy metrics")
        
        # Ensure games are sorted by date
        games_df = completed_games.copy()
        games_df = games_df.sort_values('GAME_DATE')
        
        # Get the first and last game dates
        if games_df.empty:
            logger.warning("No completed games data to calculate weekly averages")
            return pd.DataFrame()
        
        first_date = games_df['GAME_DATE'].min()
        last_date = games_df['GAME_DATE'].max()
        
        # Generate all 7-day periods starting from the first game
        weekly_periods = []
        current_date = first_date
        
        while current_date <= last_date:
            period_end = current_date + pd.Timedelta(days=6)
            weekly_periods.append((current_date, period_end))
            current_date = current_date + pd.Timedelta(days=7)
        
        # Calculate accuracy for each weekly period
        weekly_metrics = []
        
        for start_date, end_date in weekly_periods:
            # Get games in this period
            period_games = games_df[(games_df['GAME_DATE'] >= start_date) & 
                                    (games_df['GAME_DATE'] <= end_date)]
            
            if not period_games.empty:
                # Calculate accuracy for this period
                correct_count = period_games['CORRECT'].sum()
                total_games = len(period_games)
                period_accuracy = correct_count / total_games if total_games > 0 else 0
                
                # Only add if we have games in this period
                if total_games > 0:
                    weekly_metrics.append({
                        'GAME_DATE': start_date,  # Use the first day of the period
                        'PERIOD_END': end_date,   # Store the last day of the period
                        'METRIC_TYPE': 'OVERALL_7_DAY_AVG',
                        'METRIC_VALUE': period_accuracy,
                        'GAMES_IN_PERIOD': total_games
                    })
        
        return pd.DataFrame(weekly_metrics)

            
    def export_data_for_dashboard(self, output_dir=DATAPATH):
        """Export datasets to CSV files for dashboard consumption."""
        logger.info(f"Exporting data for dashboards to {output_dir}")

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare today's games
        todays_games = self.prepare_todays_games()

        # Process completed games data using the combined function
        completed_games, season_summary, _ = self.prepare_processed_games_data(num_recent_games=25)

        # Create a consolidated games dataframe
        consolidated_games = []

        # Add today's games (if any) with a game status flag
        if todays_games is not None and not todays_games.empty:
            if 'GAME_DATE' in todays_games.columns and not pd.api.types.is_datetime64_any_dtype(todays_games['GAME_DATE']):
                todays_games['GAME_DATE'] = pd.to_datetime(todays_games['GAME_DATE'])
                
            # Get the latest date in completed_games to add running accuracy metrics to today's games
            if not completed_games.empty:
                latest_date = completed_games['GAME_DATE'].max()
                
                # Add running accuracy for today's games using the latest values from completed games
                if not completed_games.empty:
                    # Get the latest home team running accuracy
                    for team_id in todays_games['HOME_TEAM_ID'].unique():
                        latest_team_games = completed_games[completed_games['HOME_TEAM_ID'] == team_id]
                        if not latest_team_games.empty:
                            latest_accuracy = latest_team_games.loc[latest_team_games['GAME_DATE'].idxmax(), 'HOME_TEAM_RUNNING_ACCURACY']
                            todays_games.loc[todays_games['HOME_TEAM_ID'] == team_id, 'HOME_TEAM_RUNNING_ACCURACY'] = latest_accuracy
                    
                    # Get the latest visitor team running accuracy
                    for team_id in todays_games['VISITOR_TEAM_ID'].unique():
                        latest_team_games = completed_games[completed_games['VISITOR_TEAM_ID'] == team_id]
                        if not latest_team_games.empty:
                            latest_accuracy = latest_team_games.loc[latest_team_games['GAME_DATE'].idxmax(), 'VISITOR_TEAM_RUNNING_ACCURACY']
                            todays_games.loc[todays_games['VISITOR_TEAM_ID'] == team_id, 'VISITOR_TEAM_RUNNING_ACCURACY'] = latest_accuracy
                    
                    # Add overall and role-based running accuracy
                    # Find the latest date in completed games
                    latest_date_idx = completed_games['GAME_DATE'].idxmax()
                    if not pd.isna(latest_date_idx):
                        latest_home_accuracy = completed_games.loc[latest_date_idx, 'HOME_ROLE_RUNNING_ACCURACY']
                        latest_away_accuracy = completed_games.loc[latest_date_idx, 'AWAY_ROLE_RUNNING_ACCURACY']
                        latest_overall_accuracy = completed_games.loc[latest_date_idx, 'OVERALL_RUNNING_ACCURACY']
                        
                        todays_games['HOME_ROLE_RUNNING_ACCURACY'] = latest_home_accuracy
                        todays_games['AWAY_ROLE_RUNNING_ACCURACY'] = latest_away_accuracy
                        todays_games['OVERALL_RUNNING_ACCURACY'] = latest_overall_accuracy
            
            todays_games['GAME_STATUS'] = 'Upcoming'
            consolidated_games.append(todays_games)

        # Add completed games with game status flag
        if not completed_games.empty:
            completed_games['GAME_STATUS'] = 'Completed'
            consolidated_games.append(completed_games)

        # Combine into one dataframe
        if consolidated_games:
            all_games_df = pd.concat(consolidated_games, ignore_index=True)
            
            # Apply column filtering for dashboard
            filtered_games_df = self.filter_dashboard_columns(all_games_df)

            # Export consolidated games file (filtered for dashboard)
            filtered_games_df.to_csv(output_path / 'games_dashboard.csv', index=False)
            logger.info(f"Exported filtered games data to {output_path / 'games_dashboard.csv'}")
            
            # Export running accuracy metrics for visualization
            self._export_running_accuracy_metrics(all_games_df, output_path)
        else:
            logger.warning("No game data to export")

        # Export season summary
        season_summary.to_csv(output_path / 'season_summary_stats.csv', index=False)
        logger.info(f"Exported season summary to {output_path / 'season_summary_stats.csv'}")

        return {
            'all_games': 'games_dashboard.csv' if consolidated_games else None,
            'season_summary': 'season_summary_stats.csv',
            'running_accuracy': 'running_accuracy_metrics.csv' if consolidated_games else None
            }
    
    def _export_running_accuracy_metrics(self, games_df, output_path):
        """Export daily running accuracy metrics for visualization."""
        logger.info("Exporting running accuracy metrics")
        
        # Ensure GAME_DATE is datetime
        games_df = games_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(games_df['GAME_DATE']):
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        # Filter to completed games only
        completed_games = games_df[games_df['GAME_STATUS'] == 'Completed'].copy()
        
        if completed_games.empty:
            logger.warning("No completed games data to export running accuracy metrics")
            return
        
        # Extract unique dates and running accuracy metrics
        accuracy_metrics = []
        
        # Get unique dates in sorted order
        dates = sorted(completed_games['GAME_DATE'].unique())
        
        for date in dates:
            # Get the games for this date
            date_games = completed_games[completed_games['GAME_DATE'] == date]
            
            # If no games on this date, skip
            if date_games.empty:
                continue
            
            # Get a sample game for this date to extract overall and role-based metrics
            sample_game = date_games.iloc[0]
            
            # Add overall and role-based accuracy for this date
            accuracy_metrics.append({
                'GAME_DATE': date,
                'METRIC_TYPE': 'OVERALL',
                'METRIC_VALUE': sample_game['OVERALL_RUNNING_ACCURACY']
            })
            
            accuracy_metrics.append({
                'GAME_DATE': date,
                'METRIC_TYPE': 'HOME_ROLE',
                'METRIC_VALUE': sample_game['HOME_ROLE_RUNNING_ACCURACY']
            })
            
            accuracy_metrics.append({
                'GAME_DATE': date,
                'METRIC_TYPE': 'AWAY_ROLE',
                'METRIC_VALUE': sample_game['AWAY_ROLE_RUNNING_ACCURACY']
            })
            
            # Add team-specific metrics
            for _, game in date_games.iterrows():
                # Add home team accuracy
                accuracy_metrics.append({
                    'GAME_DATE': date,
                    'TEAM_ID': game['HOME_TEAM_ID'],
                    'TEAM_NAME': NBA_TEAMS_NAMES.get(game['HOME_TEAM_ID'], ''),
                    'METRIC_TYPE': 'TEAM',
                    'METRIC_VALUE': game['HOME_TEAM_RUNNING_ACCURACY']
                })
                
                # Add visitor team accuracy
                accuracy_metrics.append({
                    'GAME_DATE': date,
                    'TEAM_ID': game['VISITOR_TEAM_ID'],
                    'TEAM_NAME': NBA_TEAMS_NAMES.get(game['VISITOR_TEAM_ID'], ''),
                    'METRIC_TYPE': 'TEAM',
                    'METRIC_VALUE': game['VISITOR_TEAM_RUNNING_ACCURACY']
                })
        
        # Calculate and add 7-day average metrics
        weekly_avg_metrics = self._calculate_weekly_averages(completed_games)
        
        # Create dataframe and export
        if accuracy_metrics:
            accuracy_df = pd.DataFrame(accuracy_metrics)
            
            # Remove duplicate team entries for each date
            if 'TEAM_ID' in accuracy_df.columns:
                team_metrics = accuracy_df[accuracy_df['METRIC_TYPE'] == 'TEAM'].copy()
                team_metrics = team_metrics.drop_duplicates(subset=['GAME_DATE', 'TEAM_ID'])
                
                # Get non-team metrics
                non_team_metrics = accuracy_df[accuracy_df['METRIC_TYPE'] != 'TEAM'].copy()
                
                # Combine unique team metrics with non-team metrics
                accuracy_df = pd.concat([team_metrics, non_team_metrics], ignore_index=True)
            
            # Add the weekly average metrics
            if not weekly_avg_metrics.empty:
                # Only keep essential columns for the export
                weekly_metrics_export = weekly_avg_metrics[['GAME_DATE', 'METRIC_TYPE', 'METRIC_VALUE']].copy()
                accuracy_df = pd.concat([accuracy_df, weekly_metrics_export], ignore_index=True)
                
                # Log some info about the weekly metrics
                logger.info(f"Added {len(weekly_avg_metrics)} weekly average metrics")
            
            # Export to CSV
            accuracy_df.to_csv(output_path / 'running_accuracy_metrics.csv', index=False)
            logger.info(f"Exported running accuracy metrics to {output_path / 'running_accuracy_metrics.csv'}")
if __name__ == "__main__":
    # Example usage
    processor = NBADataProcessor()
    exported_files = processor.export_data_for_dashboard()
    
    print("\nData Processing Complete!")
    print(f"Files exported for dashboard:")
    for key, value in exported_files.items():
        if value:
            print(f"- {key}: {value}")