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

