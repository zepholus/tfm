from prophet import Prophet as Prophetfb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from prophet.diagnostics import cross_validation
from src.metrics import nash, pbias

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)



class ProphetFittedModel:


    def __init__(self, m, df):

        #check if m is Prophet
        if not isinstance(m, Prophetfb):
            raise Exception('m is not Prophet')
        

        """
        percentage_non_nan = int(len(df.dropna())*0.8) #70% of the non nan values for training

        #get last day of training
        date_split = df.iloc[percentage_non_nan-1]['ds']

        len_train = len(df[df['ds'] < date_split])
        test_len = len(df) - len_train #get the length of the testing df

        k = 10 #number of folds

        df_cv = cross_validation(m, initial=f"{len_train} days", period=f'{test_len / k} days', horizon = f'{test_len / k} days')
        """

        #predict on the whole df
        forecast = m.predict(df)
        forecast['y'] = df['y'].reset_index(drop = True)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y']]

        forecast['error'] = forecast['y'] - forecast['yhat']
        forecast['uncertainty'] = forecast['yhat_upper'] - forecast['yhat_lower']
        forecast['anomaly'] = forecast.apply(lambda x: 1 if(np.abs(x['error']) > 1*x['uncertainty']) else 0, axis = 1)

        self.m = m
        self.df = df
        self.df_cv = forecast


    def predict(self, df):

        #check if df has 1 column named ds
        if 'ds' not in df.columns:
            raise Exception('Dataframe does not have ds column')
        
        #check if ds is datetime
        if not np.issubdtype(df['ds'].dtype, np.datetime64):
            raise Exception('ds column is not datetime')
        
        _df = df.copy()[['ds']]
    
        return self.m.predict(_df)
    
    def nash(self):

        df = self.df_cv[['ds', 'y', 'yhat']].dropna()
        return nash(df['y'], df['yhat'])
    
    def pbias(self):
        df = self.df_cv[['ds', 'y', 'yhat']].dropna()
        return pbias(df['y'], df['yhat'])
    
    def a(self, a):
        return 2*a
    
    
    def plot(self, anomalies = None):


        fig_definition = [
            go.Scatter(
                name='Observated flow',
                x=self.df.ds,
                y=self.df.y,
                mode='lines',
                line=dict(color='rgb(255, 165, 0)'),
            ),
            go.Scatter(
                name='Forecasted flow',
                x=self.df_cv.ds,
                y=self.df_cv.yhat,
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='Upper Bound',
                x=self.df_cv['ds'],
                y=self.df_cv['yhat_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=self.df_cv['ds'],
                y=self.df_cv['yhat_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ]

        if anomalies is not None:
            fig_definition.append(
                go.Scatter(
                    name='Anomalies',
                    x=anomalies.ds,
                    y=anomalies.y,
                    mode='markers',
                    marker=dict(color='rgb(255, 0, 0)'),
                )
            )
        
        fig = go.Figure(fig_definition)
        

        fig.update_layout(
            yaxis_title='Flow (m3/s)',
            title=f"Observations vs Forecast -- Nash: {self.nash():.2f}",
            hovermode="x"
        )
        
        
        fig.show()

    

        


