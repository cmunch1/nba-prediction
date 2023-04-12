import pandas as pd
import numpy as np

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import (
    CalibrationDisplay,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def encode_categoricals(df: pd.dataframe, category_columns: list, MODEL_NAME: str, ENABLE_CATEGORICAL: bool) -> pd.dataframe:
    """
    Encode categorical features as integers for use in XGBoost and LightGBM

    Args:
        df (pd.DataFrame): the dataframe to process
        category_columns (list): list of columns to encode as categorical
        MODEL_NAME (str): the name of the model being used
        ENABLE_CATEGORICAL (bool): whether or not to enable categorical features in the model
    
    Returns:
        the dataframe with categorical features encoded
    

    """

    # To use special category feature capabilities in XGB and LGB, categorical features must be ints from 0 to N-1
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


def plot_calibration_curve(clf_list: list, X_train: pd.dataframe, y_train: pd.dataframe, X_test: pd.dataframe, y_test: pd.dataframe, n_bins: int = 10) -> None:
    """
    Plots calibration curves for a list of classifiers vs ideal probability distribution

    Args:
        clf_list (list): the classifiers to plot
        X_train (pd.dataframe): training data
        y_train (pd.dataframe): labels for training data
        X_test (pd.dataframe): test data
        y_test (pd.dataframe): labels for test data
        n_bins (int, optional): how many bins to use for calibration. Defaults to 10.

    Returns:
        None

    FROM: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    """
    

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

    return


def calculate_classification_metrics(clf_list: list, X_train: pd.dataframe, y_train: pd.dataframe, X_test: pd.dataframe, y_test: pd.dataframe) -> tuple[pd.dataframe, list]:
    """
    Calculates classification metrics for a list of classifiers and returns the fitted models as well. Brier score, log loss, precision, recall, f1, and roc_auc are calculated.

    Args:
        clf_list (list): the classifiers to calculate metrics for
        X_train (pd.dataframe): training data
        y_train (pd.dataframe): labels for training data
        X_test (pd.dataframe): test data
        y_test (pd.dataframe): labels for test data

    Returns:
       tuple: (dataframe) of the metrics and (list) containing the fitted models and names of the models as strings

    FROM: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    """

    # this is 

    scores = defaultdict(list)

    for i, (clf, name) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        scores["Classifier"].append(name)

        for metric in [brier_score_loss, log_loss]:
            score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            scores[score_name].append(metric(y_test, y_prob[:, 1]))

        for metric in [precision_score, recall_score, f1_score, roc_auc_score]:
            score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
            scores[score_name].append(metric(y_test, y_pred))

        score_df = pd.DataFrame(scores).set_index("Classifier")
        score_df.round(decimals=3)

        # update clf_list with the trained model
        clf_list[i] = (clf, name)


    return score_df, clf_list