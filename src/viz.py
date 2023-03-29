import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go




def observation_vs_prediction(self, observation, prediction, title='Observation vs Prediction'):
    
    #fig = px.line(forecasted, x='ds', y='y', title='aa')
    #fig.add_scatter(x=forecasted['ds'], y=forecasted['yhat'])
    fig = px.line(self.df, x='ds', y='y', title=title)

        
    fig = go.Figure([

        go.Scatter(
            name='Observated flow',
            x=observation['ds'],
            y=observation['y'],
            mode='lines',
            line=dict(color='rgb(255, 165, 0)'),
        ),
        go.Scatter(
            name='Forecasted flow',
            x=prediction.df['ds'],
            y=prediction.df['y'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        """
            go.Scatter(
                name='Upper Bound',
                x=self.df['ds'],
                y=self.df['yhat_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=self.df['ds'],
                y=self.df['yhat_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        """
        ])
        """
        fig.update_layout(
            yaxis_title='Wind speed (m/s)',
            title='Continuous, variable value error bars',
            hovermode="x"
        )
        """
        fig.show()
