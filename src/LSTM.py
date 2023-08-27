import numpy as np
import src.ForecastingModel as ForecastingModel
from src.LSTMFittedModel import LSTMFittedModel
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import joblib
from config.config import LSTM_CALIBRATED_DIR, AUTOENCODER_DIR, OBSERVACIONS_DIR, OBSERVACIONS_FILTRAT_DIR
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import os
import warnings 
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler


def flatten(X):
    '''
    Flatten a 3D array.
    
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def scale_data_anomaly_detection(data_all_stations):
    #for scaling, we use same scaler for both observations and predictions. That way, we can compare them in same axis. What we want to compare between timeseries is the shape, not the values
    scaled_data = []
    for sample in data_all_stations:    

        data_for_scaler = sample.reshape(-1)
        data_for_scaler = np.array([[value, value] for value in data_for_scaler])

        scaler = MinMaxScaler()
        scaler.fit(data_for_scaler)
        transformed_train = scaler.transform(sample)

        scaled_data.append(transformed_train)
        
    return np.array(scaled_data)


class LSTM(ForecastingModel.ForecastingModel):

    def __init__(self, station_name = "default_name", model_with_flow = False, transfer_learning = False, nse_error = False):
       
        self.station_name = station_name
        self.model_with_flow = model_with_flow

        self.autoencoder_model = load_model(AUTOENCODER_DIR / 'lstm_autoencoder.h5')
        self.window_lenght_anomaly = 14
        self.loss_threshold = 0.04
        #self.loss_threshold = 0


        if model_with_flow:
            self.n_input = 30
            self.scaler = joblib.load(LSTM_CALIBRATED_DIR / 'lstm_scaler_3.pkl')
            self.model = load_model(LSTM_CALIBRATED_DIR / 'lstm_hypermodel_3.h5')

        elif transfer_learning:
            self.n_input = 150

            #Entrenat primer amb dades swat, fine tuned amb observacions
            self.scaler = joblib.load(LSTM_CALIBRATED_DIR / 'lstm_scaler_5.pkl')
            self.model = load_model(LSTM_CALIBRATED_DIR / 'lstm_transfer_learning.h5')
        
        elif nse_error:

            print('aaaa')
            self.n_input = 150
            
            #Entrenat amb observacions i buits amb swat
            self.scaler = joblib.load(LSTM_CALIBRATED_DIR / 'lstm_scaler.pkl')
            self.model = load_model(LSTM_CALIBRATED_DIR / 'lstm_hypermodel.h5')



        else:
            self.n_input = 150
            
            #Entrenat amb observacions i buits amb swat
            self.scaler = joblib.load(LSTM_CALIBRATED_DIR / 'lstm_scaler.pkl')
            self.model = load_model(LSTM_CALIBRATED_DIR / 'lstm_hypermodel_log_mse.h5')

        

    """
    Public methods
    """
    def fit(self, df):


        
        _df = df.copy()

        #check if ds is datetime
        if not np.issubdtype(_df['ds'].dtype, np.datetime64):
            #check if can be converted to datetime
            try:
                _df['ds'] = pd.to_datetime(_df['ds'])
            except:
                raise Exception('ds column is not datetime')

        df_train = _df.copy()
        df_train = df_train.drop('ds', axis=1)

        if self.model_with_flow:
            X_train = df_train.values
    
        else:
            #Create the generator
            X_train = df_train.drop('y', axis=1).values

        
        y_train = df_train['y'].values
        

        #Scale the data
        X_train = self.scaler.transform(X_train)
        
        generator = TimeseriesGenerator(X_train, y_train, length=self.n_input, batch_size=1)

        inputs = np.array([sample[0][0] for sample in generator])
        predictions = self.model.predict(inputs)
        predictions = [prediction[0] for prediction in predictions]

        _df1 = _df[self.n_input:].copy()

        _df1['yhat'] = predictions
        _df2 = _df1[['ds', 'y', 'yhat']].copy()

        #return _df1[['ds', 'y', 'yhat']]

        #Calculate anomalies
        #df_flow_scaled =  self.autoencoder_scaler.transform(_df2[['y', 'yhat']].values)
        df_flow = _df2[['y', 'yhat']].values
        window = self.window_lenght_anomaly
        flow_strided = np.lib.stride_tricks.as_strided(df_flow, shape=(df_flow.shape[0] - window + 1, window, df_flow.shape[1]), strides=(df_flow.strides[0], df_flow.strides[0], df_flow.strides[1]))
        flow_strided = scale_data_anomaly_detection(flow_strided)

        #scaler = MinMaxScaler()
        #flow_strided = np.array([scaler.fit_transform(sample) for sample in flow_strided])
        
        
        flow_pred = self.autoencoder_model.predict(flow_strided, verbose=0)
        test_mae_loss = np.mean(np.abs(flatten(flow_pred) - flatten(flow_strided)), axis=1)

        #drop first window-1 values
        _df2 = _df2[window-1:].copy()
        _df2['loss'] = test_mae_loss
        _df2['anomaly'] = _df2['loss'] > self.loss_threshold

        return LSTMFittedModel(_df2[['ds', 'y', 'yhat', 'anomaly', 'loss']], self.station_name)
    

    


