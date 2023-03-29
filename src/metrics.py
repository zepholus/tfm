import pandas as pd
import numpy as np






#1 is better
def nash(observations, predictions):

    x = observations
    y = predictions

    nse = 1 - (sum((x - y)**2) / sum((x - x.mean())**2))
    return nse
    

#0 is better
def pbias(y, yhat):
    pbias = 100 * (sum(yhat) - sum(y)) / sum(y)
    return pbias