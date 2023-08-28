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
- ```documents/``` contains the PDF documents and presentation generated for the current thesis
- ```figures/```  contains all the generated figures that have been utilized in the various presented documents. 
- ```notebooks/``` contains the notebooks that guide to the results of the study
    - ```anomaly_lstm/``` contains the notebooks for reading and organising the required data for training the LSTM autoencoder for anomaly detection, and displaying its results
    - ```lstm/``` contains the notebooks for reading and organising the required data for training and testing the different versions of the LSTM neural network developed for the river flow prediction (required for the anomaly detection using LSTM autoencoder)
    - ```prophet/``` contains the notebooks for reading and organising the required data for the anomaly detection using Prophet library
    - ```filtrar_estacions_aforament.ipynb``` : in this notebook, the original flow data is labeled into good samples and anomaly
    - ```plot_estacions_meteo.ipynb``` : in this notebook, we plot the location of the metereological stations into a map
    - ```prepare_swat_predictions.ipynb``` : in this notebook, the original predictions of SWAT are read and organized to facilitate their utilization in the code
