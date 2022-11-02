def fix_datatypes(df):
    
    '''
    Converts date to proper format and reduces memory footprint of ints and floats
    '''
    
    import pandas as pd
    
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])

    long_integer_fields = ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON']

    #convert long integer fields to int32 from int64
    for field in long_integer_fields:
        df[field] = df[field].astype('int32')
    
    #convert the remaining int64s to int8
    for field in df.select_dtypes(include=['int64']).columns.tolist():
        df[field] = df[field].astype('int8')
        
    #convert float64s to float16s
    for field in df.select_dtypes(include=['float64']).columns.tolist():
        df[field] = df[field].astype('float16')
        
    return df

def add_date_features(df):
    
    '''
    Converts game date to month to limit cardinality
    '''
    
    import pandas as pd

    df['MONTH'] = df['GAME_DATE_EST'].dt.month
    
    return df