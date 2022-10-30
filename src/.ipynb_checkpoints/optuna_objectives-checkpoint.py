def XGB_objective(trial, train, target, STATIC_PARAMS, ENABLE_CATEGORICAL, NUM_BOOST_ROUND, OPTUNA_CV, OPTUNA_FOLDS, SEED):
    
    import numpy as np
    import xgboost as xgb
    
    from sklearn.model_selection import (
        StratifiedKFold, 
        TimeSeriesSplit,
    )
    
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
    )

    train_oof = np.zeros((train.shape[0],))
    
    train_dmatrix = xgb.DMatrix(train, target,
                         feature_names=train.columns,
                        enable_categorical=ENABLE_CATEGORICAL)
    
    xgb_params= {       
                'num_round': trial.suggest_int('num_round', 2, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1E-3, 1),
                'max_bin': trial.suggest_int('max_bin', 2, 1000),
                'max_depth': trial.suggest_int('max_depth', 1, 8),
                'alpha': trial.suggest_float('alpha', 1E-16, 12),
                'gamma': trial.suggest_float('gamma', 1E-16, 12),
                'reg_lambda': trial.suggest_float('reg_lambda', 1E-16, 12),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 1E-16, 1.0),
                'subsample': trial.suggest_float('subsample', 1E-16, 1.0), 
                'min_child_weight': trial.suggest_float('min_child_weight', 1E-16, 12),
                'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 15),       
                }
    
    xgb_params = xgb_params | STATIC_PARAMS
        
   #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "evaluation-auc")
    
    if OPTUNA_CV == "StratifiedKFold": 
        kf = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    elif OPTUNA_CV == "TimeSeriesSplit":
        kf = TimeSeriesSplit(n_splits=OPTUNA_FOLDS)
    

    for f, (train_ind, val_ind) in (enumerate(kf.split(train, target))):

        train_df, val_df = train.iloc[train_ind], train.iloc[val_ind]
        
        train_target, val_target = target[train_ind], target[val_ind]

        train_dmatrix = xgb.DMatrix(train_df, label=train_target,enable_categorical=ENABLE_CATEGORICAL)
        val_dmatrix = xgb.DMatrix(val_df, label=val_target,enable_categorical=ENABLE_CATEGORICAL)


        model =  xgb.train(xgb_params, 
                           train_dmatrix, 
                           num_boost_round = NUM_BOOST_ROUND,
                           #callbacks=[pruning_callback],
                          )

        temp_oof = model.predict(val_dmatrix)

        train_oof[val_ind] = temp_oof

        #print(roc_auc_score(val_target, temp_oof))
    
    val_score = roc_auc_score(target, train_oof)
    
    return val_score


def LGB_objective(trial, train, target, category_columns, STATIC_PARAMS, ENABLE_CATEGORICAL, NUM_BOOST_ROUND, OPTUNA_CV, OPTUNA_FOLDS, SEED):

    import numpy as np
    import lightgbm as lgb
    from lightgbm import (
        early_stopping,
        log_evaluation,
    )
    
    from sklearn.model_selection import (
        StratifiedKFold, 
        TimeSeriesSplit,
    )
    
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
    )
    
    train_oof = np.zeros((train.shape[0],))
    
    
    lgb_params= {
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 0.5),
                "max_depth": trial.suggest_categorical('max_depth', [5,10,20,40,100, -1]),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200000),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
                "cat_smooth" : trial.suggest_int('min_data_per_groups', 1, 100)
                }

    lgb_params = lgb_params | STATIC_PARAMS
        
    #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    
    if OPTUNA_CV == "StratifiedKFold": 
        kf = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    elif OPTUNA_CV == "TimeSeriesSplit":
        kf = TimeSeriesSplit(n_splits=OPTUNA_FOLDS)
    

    for f, (train_ind, val_ind) in (enumerate(kf.split(train, target))):

        train_df, val_df = train.iloc[train_ind], train.iloc[val_ind]
        
        train_target, val_target = target[train_ind], target[val_ind]

        train_lgbdataset = lgb.Dataset(train_df, label=train_target,categorical_feature=category_columns)
        val_lgbdataset = lgb.Dataset(val_df, label=val_target, reference = train_lgbdataset, categorical_feature=category_columns)


        model =  lgb.train(lgb_params, 
                           train_lgbdataset,
                           valid_sets=val_lgbdataset,
                           #num_boost_round = NUM_BOOST_ROUND,
                           callbacks=[#log_evaluation(LOG_EVALUATION),
                                      early_stopping(EARLY_STOPPING,verbose=False),
                                      #pruning_callback,
                                    ]               
                           #verbose_eval= VERBOSE_EVAL,
                          )

        temp_oof = model.predict(val_df)

        train_oof[val_ind] = temp_oof

        #print(roc_auc_score(val_target, temp_oof))
    
    val_score = roc_auc_score(target, train_oof)
    
    return val_score