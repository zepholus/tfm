from prophet import Prophet as Prophetfb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from prophet.diagnostics import cross_validation
from src.metrics import nash, pbias
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)



class ProphetFittedModel:


    def __init__(self, m, df, station_name):

        #check if m is Prophet
        if not isinstance(m, Prophetfb):
            raise Exception('m is not Prophet')
        
        #In case we were doing predictions on the future, which we are not
        """
        df_non_nan = df.dropna()
        percentage_non_nan = int(len(df_non_nan)*0.8) #80% of the non nan values for training

        #get last day of training
        date_split = df_non_nan.iloc[percentage_non_nan-1]['ds']

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
        self.station_name = station_name
    

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

            #convert ds column to datetime
            _anomalies = anomalies.copy()
            _anomalies['ds'] = pd.to_datetime(_anomalies['ds'])

            predicted_anomalies = self.df_cv[self.df_cv['anomaly'] == 1]

            #inner join in order to get the anomalies that are in the predicted anomalies
            well_detected_anomalies = _anomalies.merge(predicted_anomalies, on = 'ds', how = 'inner')

            #find anomalies predicted but not observed
            bad_detected_anomalies = predicted_anomalies[~predicted_anomalies['ds'].isin(well_detected_anomalies['ds'])]

            
            fig_definition.append(
                go.Scatter(
                    name='Observed anomalies',
                    x=_anomalies.ds,
                    y=_anomalies.y,
                    mode='markers',
                    marker=dict(color='rgb(255, 0, 0)'),
                )
               
            )
            
            fig_definition.append(
                go.Scatter(
                    name='Well predicted anomalies',
                    x=well_detected_anomalies.ds,
                    y=well_detected_anomalies.y_x,
                    mode='markers',
                    marker=dict(color='green'),
                ) 
            )
            fig_definition.append(
                go.Scatter(
                    name='Bad predicted anomalies',
                    x=bad_detected_anomalies.ds,
                    y=bad_detected_anomalies.y,
                    mode='markers',
                    marker=dict(color='purple'),
                ) 
            )
            
        
        fig = go.Figure(fig_definition)
        

        fig.update_layout(
            #yaxis_title='Flow (m3/s)',
            #title=f"Observations vs Forecast of station {self.station_name}-- Nash: {self.nash():.2f}",
            #hovermode="x",
            showlegend=False
        )
        fig.write_image("C:\\Users\\joans\\OneDrive\\Escriptori\\master\\tfm\\tfm\\figures\\prophet\\"+self.station_name+".svg")

        fig.show()

    def get_statistics_anomaly_prediction(self, anomalies):
        _anomalies = anomalies.copy()
        _anomalies = _anomalies.dropna()
        
        #anomalies ds to datetime
        _anomalies['ds'] = pd.to_datetime(_anomalies['ds'])

        _forecast = self.df_cv[['ds', 'y', 'anomaly']].copy()
        _forecast = _forecast.dropna()  #account only non null observations 
        _forecast = _forecast[['ds', 'anomaly']].copy()
        
        _forecast['observed_anomaly'] = 0

        #if anomaly in anomalies, observed_anomaly = 1
        for index, row in _forecast.iterrows():
            if row['ds'] in _anomalies['ds'].values:
                _forecast.at[index, 'observed_anomaly'] = 1


        y_true = _forecast['observed_anomaly']
        y_pred = _forecast['anomaly']

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    

        


