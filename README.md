# DeepPhasePick
DeepPhasePick is a method for detecting and picking seismic phases from local earthquakes based on highly optimized convolutional and recurrent deep neural networks.

## 1. Install

...

### 2. Sample Data

#### 2.1 Continuous seismic data

DeepPhasePick is applied on continuous seismic data stored in the archived directory structure in

**archive/YY/NET/STA/CH**

Where YY is year, NET is the network code, STA is the station code and CH is
the channel code (for example HH).
This archive can be updated with new data...

### 3. Optimized trained Models

Best performing trained models and other relevant results obtained from the hyperparameter optimization are stored in the following directories:

**detection** -> phase detection related results,

**picking/P** -> P-phase picking related results, and


**picking/S** -> S-phase picking related results.

### 4. DeepPhasePick Worflow

DeepPhasePick worflow is controlled by several dictionaries defined in the run\_dpp.py script.
Using the parameters defined in these dictionaries, the script call the functions sotred in dpp.py following the steps described below...

1) Reading trained models and other relevant results from hyperparameter
optimization.

Parameters used here should not be changed by the user, since...

2) Definition of user-defined parameters...

2.1) Parameters used for detection and picking of phases.

dct\_param -> dictionary defining how waveform data is to be preprocessed.

dct\_trigger -> dictionary defining how predicted probabity time series are

2.2) Parameters defining continuous waveform data on which DeepPhasePick is applied.

dct\_sta -> dictionary defining the archived waveform data.

dct\_fmt ->  dictionary defining some formatting for plotting prediction

dct\_time -> dictionary defining the time over which prediction is made.

3) Run DeepPhasePick on continuous waveform data.

3.1) Perform phase detection, for predicting preliminary phase picks.

3.2) Perform phase picking, for predicting refined phase picks, and optionally plotting them / saving some relevant statistics.

3.3) Plotting of continuous waveform including predicted P, S phases, and corresponding predicted probability time series.



Please let us know of any bugs found in the code.


### Citation:

If you use this algorithm for research purpose, please cite it as follows:

- Soto, H., & Schurr, B. (2020). DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local Earthquakes based on highly
optimizedConvolutional and Recurrent Deep Neural Networks. EarthArXiv preprint DOI: XX (https://eartharxiv.org/repository/view/1752/).
