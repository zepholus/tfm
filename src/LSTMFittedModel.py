import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from src.metrics import nash, pbias
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from src.FittedModel import FittedModel





class LSTMFittedModel(FittedModel):
    def __init__(self, df, station_name):
        self.df = df
        self.station_name = station_name
        self.df_cv = df #for compatibility with prophet
    
    
    def plot(self, anomalies = None, save_html = None):
        FittedModel.plot(self, anomalies, 'lstm', save_html)
    """
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
            yaxis_title='Flow (m3/s)',
            title=f"Observations vs Forecast of station {self.station_name}-- Nash: {self.nash():.2f}",
            hovermode="x",
            showlegend=True
        )
        #fig.write_image("C:\\Users\\joans\\OneDrive\\Escriptori\\master\\tfm\\tfm\\figures\\prophet\\"+self.station_name+".svg")

        fig.show()
    """








        


