import pandas as pd
import numpy as np
import sys





#1 is better
def nash(observations, predictions):

    x = observations
    y = predictions

    #nash sutcliffe efficiency
    if (sum((x - x.mean())**2)) == 0:
        return -9999

    nse = 1 - (sum((x - y)**2) / (sum((x - x.mean())**2)))


    return nse
    

#0 is better
def pbias(y, yhat):
    pbias = 100 * (sum(yhat) - sum(y)) / sum(y)
    return pbias


