# tfm
Summary

## Setup Python environment
To create the environment with all packages needed, simply run
```
conda env create -f environment.yml
conda activate tfm
pip install -e .
```

## Run locally

Before starting to do anything, make sure you have activated the conda environment.

```
conda activate tfm
```

## Content of the repository
- ```data/``` contains all the data needed for training and evaluating your models
- ```src/``` contains the entire code for executing the notebooks
- ```notebooks/``` contains the notebooks that guide to the results of the study
    - ```anomaly_lstm/``` contains the notebooks for reading and organising the required data for training the LSTM autoencoder for anomaly detection, and displaying its results
    - ```lstm/``` contains the notebooks for reading and organising the required data for training and testing the different versions of the LSTM neural network developed for the river flow prediction (required for the anomaly detection using LSTM autoencoder)
    - ```prophet/``` contains the notebooks for reading and organising the required data for the anomaly detection using Prophet library
