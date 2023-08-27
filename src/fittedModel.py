from abc import ABC
import plotly.graph_objs as go
import pandas as pd
from src.metrics import nash, pbias
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


 
class FittedModel(ABC):
        
    def nash(self):
        df = self.df_cv[['ds', 'y', 'yhat']].dropna()
        return nash(df['y'], df['yhat'])
    
    def get_df(self):
        return self.df_cv
    
    def pbias(self):
        df = self.df_cv[['ds', 'y', 'yhat']].dropna()
        return pbias(df['y'], df['yhat'])


    def plot(self, anomalies = None, model_name = None, save_html = None):


        fig_definition = [
            go.Scatter(
                name='Observated flow',
                x=self.df_cv.ds,
                y=self.df_cv.y,
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
        if model_name == 'prophet':
            fig_definition.append(
                go.Scatter(
                    name='Upper Bound',
                    x=self.df_cv['ds'],
                    y=self.df_cv['yhat_upper'],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
            )
            fig_definition.append(
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
            )

        if anomalies is not None:

            #convert ds column to datetime
            _anomalies = anomalies.copy()
            _anomalies['ds'] = pd.to_datetime(_anomalies['ds'])

            predicted_anomalies = self.df_cv[self.df_cv['anomaly'] == 1]



            #delete rows from anomalies that are not in df_cv
            _anomalies = _anomalies[_anomalies['ds'].isin(self.df_cv['ds'])]



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

        #save svg
        #fig.write_image("C:\\Users\\joans\\OneDrive\\Escriptori\\master\\tfm\\tfm\\figures\\prediccions_lstm\\"+self.station_name+".svg")

        if save_html:
            fig.write_html(f'{save_html}\\{self.station_name}.html')

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

        
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except:
            tn, fp, fn, tp = 0, 0, 0, 0

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }







    

        


