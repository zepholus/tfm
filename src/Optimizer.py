from src.Prophet import Prophet
import numpy as np
import pandas as pd
import itertools




class ProphetOptimizer:


    def __init__(self, df):
        self.df = df

    def optimize(self):
        
        param_grid = {  
            'changepoint_prior_scale': np.linspace(0.001, 0.5, 10),
            'seasonality_prior_scale': np.linspace(0.01, 10, 10),
        }
        
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        errors_df = []

        for params in all_params:
            changepoint_prior_scale = params['changepoint_prior_scale']
            seasonality_prior_scale = params['seasonality_prior_scale']
            
            m = Prophet(changepoint_prior_scale, seasonality_prior_scale)
            fitted_model = m.fit(self.df)

            nash = fitted_model.nash()
            pbias = fitted_model.pbias()

            params['nash'] = nash
            params['pbias'] = pbias

            errors_df.append(params)
        
        return pd.DataFrame(all_params).sort_values(by=['nash', 'pbias'], ascending=False)




    

        


