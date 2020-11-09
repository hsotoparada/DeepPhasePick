# DeepPhasePick

DeepPhasePick is a method for automatically detecting and picking seismic phases from local earthquakes based on highly optimized deep neural networks.
In a first stage, phase detection is performed by a Convolutional Neural Network (CNN) on three-component seismograms.
P- and S-picking is then conducted by two Long-Short Term Memory (LSTM) Recurrent Neural Networks (RNN) on the vertical and the two-horizontal components, respectively.
CNN and LSTM networks have been trained using >30,000 seismic records extracted from manually-picked event waveforms originating from northern Chile.
DeepPhasePick additionally computes uncertainties of the predicted phase time onsets, based on the Monte Carlo Dropout (MCD) technique as an approximation of Bayesian inference.
Predicted phase time onsets and associated uncertainties generated by DeepPhasePick can be used to feed a phase associator algorithm in an automatic earthquake location procedure.

## 1. Install

...

## 2. Seismic Data

DeepPhasePick method is applied on three-component MiniSEED seismic waveforms, which must be stored in a commonly used archive directory structured as:

**archive/YY/NET/STA/CH**

Here **YY** is year, **NET** is the network code, **STA** is the station code and **CH** is the channel code (e.g., HH) of the seismic streams.
Example data is included in the **archive** directory, where new data, on which DeepPhasePick will be applied, can be added by users.

## 3. Optimized trained Models

Best performing trained models and other relevant results obtained from the hyperparameter optimization are stored in the following directories:

**detection** -> results related to the phase detection stage,

**picking/P** -> results related to the P-phase picking stage, and

**picking/S** -> results related to the S-phase picking stage.

## 4. DeepPhasePick worflow

The DeepPhasePick workflow is controlled by the parameters in several dictionaries defined in the **run\_dpp.py** script.
These parameters are used by **run\_dpp.py** to perform the steps described below.

(1) Reading of optimized models trained for phase detection and picking tasks, and other relevant results from the hyperparameter optimization.

Parameters used in this part of the script are fixed and should not be changed by the user.

(2) Definition of user-defined parameters used for detection and picking of phases.

(2.1) Parameters used for detection and picking of phases.

**dct\_param** -> dictionary defining how waveform data is to be preprocessed.
The parameters in this dictionary are defined based on the optimized hyperparameters of the trained models, then it is recommended not to modify them.

**dct\_trigger** -> dictionary defining how predicted probabity time series are used to obtain preliminary and refined phase onsets.
Here user-defined parameters are included in the following three nested dictionaries:

**dct\_trigger['detec\_thres']**: dictionary containing user-defined parameters defining how the preliminary onsets are obtained in the phase detection stage.

**dct\_trigger['detec\_cond']**: dictionary containing user-defined parameters applied in optional conditions for improving phase detections, by keeping or removing presumed false preliminary onsets.

**dct\_trigger['mcd']**: dictionary containing user-defined parameters controlling how Monte Carlo Dropout MCD technique is applied in the phase picking stage.

(2.2) Parameters defining continuous waveform data on which DeepPhasePick is applied.

**dct\_sta** -> dictionary defining the archived waveform data.

**dct\_fmt** ->  dictionary defining some formatting for plotting prediction.

**dct\_time** -> dictionary defining the time over which predictions are made.

**dct\_out** -> dictionary defining DeepPhasePick output options..

User-defined parameters in **dct\_sta**, **dct\_fmt**, and **dct\_time** dictionaries should be included within nested dictionaries defined by the label **dct\_out['flag_data']**
that describes the seismic data on which DeepPhasePick is applied. DeepPhasePick results will be stored in the directory **dct\_out['opath']/dct\_out['flag_data']**.

(3) Run DeepPhasePick on continuous waveform data.

(3.1) Perform phase detection: prediction of preliminary phase picks.

(3.2) Perform phase picking: prediction of refined phase picks, optionally plotting them and saving some relevant statistics.

(3.3) Plotting of continuous waveform with predicted P and S phases, and corresponding predicted probability time series.


## Citation:

If you use this algorithm for research purpose, please cite it as follows:

- Soto, H., & Schurr, B. (2020). DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local Earthquakes based on highly
optimizedConvolutional and Recurrent Deep Neural Networks. EarthArXiv preprint DOI: XX (https://eartharxiv.org/repository/view/1752/).

Please let us know of any bugs found in the code.


## Thanks:

The development of DeepPhasePick method has received financial support from

-  The HAzard and Risk Team (HART) initiative of the GFZ German Research Centre for Geosciences in collaboration with the Institute of GeoSciences, Energy, Water
and Environment of the Polytechnic University Tirana, Albania and the KIT Karlsruhe Institute of Technology.

