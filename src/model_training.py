import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import (
    CalibrationDisplay,
)



def fix_datatypes(df, date_columns, long_integer_columns):
    
    for field in date_columns:
        df[field] = pd.to_datetime(df[field])
 

    #convert long integer fields to int32 from int64
    for field in long_integer_columns:
        df[field] = df[field].astype('int32')
    
    #convert the remaining int64s to int8
    for field in df.select_dtypes(include=['int64']).columns.tolist():
        df[field] = df[field].astype('int8')
        
    #convert float64s to float16s
    for field in df.select_dtypes(include=['float64']).columns.tolist():
        df[field] = df[field].astype('float16')
        
    return df

def encode_categoricals(df, category_columns, MODEL_NAME, ENABLE_CATEGORICAL):
    
    # To use special category feature capabalities in XGB and LGB, categoricals must be ints from 0 to N-1
    # Conversion can be accomplished by simple subtraction for several features
    # (these category capabilities may or may not be used, but encoding does not hurt anything)
    
    first_team_ID = df['HOME_TEAM_ID'].min()
    first_season = df['SEASON'].min()
   
    # subtract lowest value from each to create a range of 0 thru N-1
    df['HOME_TEAM_ID'] = (df['HOME_TEAM_ID'] - first_team_ID).astype('int8') #team ID - 1610612737 = 0 thru 29
    df['VISITOR_TEAM_ID'] = (df['VISITOR_TEAM_ID'] - first_team_ID).astype('int8') 
    df['SEASON'] = (df['SEASON'] - first_season).astype('int8')
    
    # if xgb experimental categorical capabilities are to be used, then features must be of category type
    if MODEL_NAME == "xgboost":
        if ENABLE_CATEGORICAL:
            for field in category_columns:
                df[field] = df[field].astype('category')

    return df

def plot_calibration_curve(clf_list, X_train, y_train, X_test, y_test, n_bins=10):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            n_bins=n_bins,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(f"Calibration plots (bins = {n_bins})")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=n_bins,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()