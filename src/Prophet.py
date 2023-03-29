from prophet import Prophet as Prophetfb
import numpy as np
import src.ForecastingModel as ForecastingModel
from src.fittedModel import ProphetFittedModel
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

class Prophet(ForecastingModel.ForecastingModel):

    def __init__(self, changepoint_prior_scale = 0.05, seasonality_prior_scale =  10):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        m = Prophetfb(interval_width=0.98, 
                      yearly_seasonality = True, 
                      changepoint_prior_scale=self.changepoint_prior_scale, 
                      seasonality_prior_scale=self.seasonality_prior_scale)
        

        self.m = m

        
    """
    Public methods
    """
    def fit(self, df):

        #check if df has 2 columns named ds and y
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise Exception('Dataframe does not have ds and y columns')
        
        _df = df.copy()[['ds', 'y']]

        #check if ds is datetime
        if not np.issubdtype(_df['ds'].dtype, np.datetime64):
            #check if can be converted to datetime
            try:
                _df['ds'] = pd.to_datetime(_df['ds'])
            except:
                raise Exception('ds column is not datetime')
        
        #check if y is numeric
        if not np.issubdtype(_df['y'].dtype, np.number):
            raise Exception('y column is not numeric')
               

        return ProphetFittedModel(self.m.fit(_df), _df)
    

    


    """
    def fit_predict(self, df, changepoint_prior_scale = 0.05, seasonality_prior_scale =  10):
        
        m = Prophetfb(interval_width=0.98, 
                      yearly_seasonality = True, 
                      changepoint_prior_scale=changepoint_prior_scale, 
                      seasonality_prior_scale=seasonality_prior_scale)
        
        m = m.fit(df)
        forecast = m.predict(df)
        forecast['y'] = df['y'].reset_index(drop = True)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']]

        forecast['error'] = forecast['y'] - forecast['yhat']
        forecast['uncertainty'] = forecast['yhat_upper'] - forecast['yhat_lower']
        forecast['anomaly'] = forecast.apply(lambda x: 1 if(np.abs(x['error']) > 1*x['uncertainty']) else 0, axis = 1)
        
        return forecast
    """
    

