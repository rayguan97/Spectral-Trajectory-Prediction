# Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clusteringin Graph-LSTMs


<p align="center">
<img src="figures/predict.png" width="400">
<img src="figures/results.gif" width="400">
</p>


## Instruction

Python version: 3.7

### Installation

1. Create a conda environement<br>
  `conda env create -f env.yml`

2. To activate the environment:<br>
  `conda activate sc-glstm`

3. Download resources <br>
  `python setup.py`

### Usage

* To run our one & two stream model:<br>
  1. `cd ours/`<br>
  2. `python main.py`
  3. To change between one stream to two stream, simply change the variable `s1` in main.py between True and False.
  4. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run EncDec comparison methods:<br>
  1. `cd comparison_methods/EncDec/`<br>
  2. `python main.py`
  3. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run GRIP comparison methods:<br>
  1. `cd comparison_methods/GRIP/`<br>
  2. `python main.py`
  3. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run TraPHic/SC-LSTM comparison methods:<br>
  1. `cd comparison_methods/traphic_sconv/`
  2. `python main.py`
  3. To change the model and methods, change `DATASET` and `PREDALGO` variable in main.py.

Note: During evaluation of the trained_models, the best results may be different from reported error due to different batch normalization applied to the network. To obtain the same number, we may have to mannually change the network.

### Data preparation steps
Important steps if you plan to prepare the data from the raw data

#### Formatting the dataset after downloading from the official website
* Run `data_processing/format_apolloscape.py` to format the downloaded apolloscape data into our desired representation
* Run `data_processing/format_lyft.py` to format the downloaded lyft data into our desired representation
* Use functions in `data_processing/generate_data.py` to format the downloaded Argoverse trajectory data into our desired representation

#### For preparing the formatted data into the data structures which our model requires
* Use `data_processing/data_stream.py` to generate input data for stream1 and stream2. 
* Use `generate_adjacency()` function in `data_processing/behaviors.py` to generate adjacency matrices.
* Must use `add_behaviors_stream2()` function in `data_processing/behaviors.py` to add behavior labels to the stream2 data before supplying the data to the network.

### Plotting

* use the `plot_behaviors()` function in `data_processing/behaviors.py` to plot the behaviors of the agents.

## Our network

<p align="center">
<img src="figures/network.png">
</p>

## Comparison with other models
![comparison of our methods with other methods](figures/compare.png)

## Results

### Trajectory Prediction Result
![Trajectory Prediction Result](figures/spectral_cluster_regularization.png)

### Behavior prediction results
<p align="center">
  <img src="figures/behaviors.png">
</p>


