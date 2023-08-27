import pandas as pd
import numpy as np
import os
from src.LSTM import LSTM
from config.config import LSTM_DATA_DIR, OBSERVACIONS_FILTRAT_DIR, OBSERVACIONS_DIR



def parse_station(file, only_testing = True, model_with_flow = False, transfer_learning = False, replace_simulated_by_nans = True, observacions_filtrades = True, nse_error = False):

    observations_file = file.replace('_stats', '')

    df = pd.read_csv(os.path.join(LSTM_DATA_DIR / file))

    df = df.rename(columns = {'datetime': 'ds', 'Flow': 'y'})

    if observacions_filtrades:
        df_observations = pd.read_csv(os.path.join(OBSERVACIONS_FILTRAT_DIR / observations_file), index_col = 0, parse_dates = True)
    else:
        df_observations = pd.read_csv(os.path.join(OBSERVACIONS_DIR / observations_file), index_col = 0, parse_dates = True)
    
    if replace_simulated_by_nans:
        df['y'] = df_observations['Flow'].astype('float32').values  #Calculate nash only for observations

    


    n = len(df)
    test_df = df[int(n*0.85):]

    lstm_model_no_flow = LSTM(station_name = file, model_with_flow=model_with_flow, transfer_learning=transfer_learning, nse_error = nse_error)
    if only_testing:
        lstm_fitted = lstm_model_no_flow.fit(test_df)
    else:
        lstm_fitted = lstm_model_no_flow.fit(df)
        
    return lstm_fitted


