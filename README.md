# DeepPhasePick

DeepPhasePick (DPP) is a method for automatically detecting and picking seismic phases from local earthquakes based on highly optimized deep neural networks.
The method work in a pipeline, where in a first stage phase detection is performed by a Convolutional Neural Network (CNN) on three-component seismograms.
Then P- and S-picking is conducted by two Long-Short Term Memory (LSTM) Recurrent Neural Networks (RNN) on the vertical and the two-horizontal components, respectively.
The CNN and LSTM networks have been trained using >30,000 seismic records extracted from manually-picked event waveforms originating from northern Chile.
DPP additionally computes uncertainties of the predicted phase time onsets, based on the Monte Carlo Dropout (MCD) technique as an approximation of Bayesian inference.
Predicted phase time onsets and associated uncertainties generated by DPP can be used to feed a phase associator algorithm as part of an automatic earthquake location procedure.

## 1. Install

You can directly clone the public repository:

    git clone https://github.com/hsotoparada/DeepPhasePick

...

## 2. DPP Worflow

### 1. DPP configuration

Before running DPP, the method needs to be configured by creating an instance of the class Config(), for example using:

    import config, data, model, util
    dpp_config = config.Config()

Then, parameters controlling different stages in the method can be configured as described below.

Parameters determining the selected waveform data on which DeepPhasePick is
applied are defined using `dpp_config.set_data()`.

For example, to select the waveforms from stations `PB01` and `PB02` (network `CX`), and channel `HH` which are stored in
the archive directory `data`, and save the results in directory `out`, run:

    dpp_config.set_data(stas=['PB01', 'PB02'], net='CX', ch='HH', archive='data', opath='out')

Parameters controlling how seismic waveforms are processed before phase detection stage are defined using `dpp_config.set_data_params()`.

For example, the following will apply a highpass filter (> .5 Hz) and resample the data to 100 Hz (if data is not already sampled at that sampling rate) before running the detection:

    dpp_config.set_data_params(samp_freq=100., st_filter='highpass', filter_opts={'freq': .5})

Note that, since the models in DPP were trained using non-filtered data, this may cause numerous false positive predictions.

DPP will be applied on the selected seismic data (see function `set_data()`) in the time windows defined using `dpp_config.set_time()`.

For example, to create 30-min (1800-seconds) time windows in the period between
`2015-04-03T00:00:00` and `2015-04-03T02:00:00` (2 hours), use:

    dpp_config.set_time(dt_iter=1800., tstart="2015-04-03T00:00:00", tend="2015-04-03T02:00:00")

Note that the windows created will have the same duration except for the last window, which will be filled with the remainder data until `tend` in case
`dt_iter + tstart(last window) > tend`.

Parameters determining how predicted discrete probability time series are computed when running phase detection on seismic waveforms are defined using `dpp_config.set_trigger()`.

For example, the following will compute the discrete probability time series every 20 samples, using a probability threshold of 0.95 for P- and S-phases:

    dpp_config.set_trigger(n_shift=20, pthres_p=[0.95, 0.001], pthres_s=[0.95, 0.001])

Parameters controlling the optional conditions applied for refining preliminary picks obtained from phase detection are defined using `dpp_config.set_picking()`.

For example, the following will remove preliminary picks which are presumed false positive, by applying all of the four optional conditions described in the
Text S1 in the Supplementary Material of Soto and Schurr (2021).
This is the default and recommended option, especially when dealing with very noise waveforms or filtered seismic waveforms, which may increase the number of presumed false positives.

Then refined pick onsets and their time uncertainties will be computed by applying 20 iterations of Monte Carlo Dropout.

    dpp_config.set_picking(run_mcd=True, mcd_iter=20)

More details on the arguments accepted by each of these configuration functions can be seen from the corresponding function documentation.

Note that, instead of configuring DPP by using the functions describe above, each set of parameters can be passed as a dictionary to `config.Config()`.
See the class documentation to use this approach.

### 2. Seismic Data

DPP method is applied on three-component MiniSEED seismic waveforms.

To read the seismic waveforms into DPP an instance of the class Data() needs to be created, for example using:

    dpp_data = data.Data()

Then, the data can be read into DPP for example from a local archive directory using:

    dpp_data.read_from_archive(dpp_config)

The local archive needs to have the following commonly used structure:

**archive/YY/NET/STA/CH**

Here **YY** is year, **NET** is the network code, **STA** is the station code and **CH** is the channel code (e.g., HHZ.D) corresponding to the seismic streams.
An example of archive data is included in `sample_data/archive`.

Alternatively, waveforms can be read from a local directory with no specific structure. For example using:

    dpp_data.read_from_directory(dpp_config)


### 3. Phase Detection and Picking

In order to run the phase detection and picking stages, an instance of the class Model() needs to be created, for example using:

    dpp_model = model.Model()

This reads the optimized trained models into DPP.
By default, the trained model weights and other relevant model information obtained from the hyperparameter optimization are read from the following directories:

**models/detection/20201002**, which contains files related to the optimized phase detection model described in Soto and Schurr (2021),

**models/picking/20201002_1/P**, which contains files related to the optimized P-phase picking model described in Soto and Schurr (2021), and

**models/picking/20201002_1/S**, which contains files related to the optimized S-phase picking model described in Soto and Schurr (2021).

The strings `20201002` and  `20201002_1` indicates the version of the models, which is defined by the optional parameters `version_det`, `version_pick_P`, `version_pick_S`
that can be passed to Model().
<br> This allows the integration of new versions of trained models in the future, which can be used as an alternative.
<br> Note that a new version (`20201002_2`) of the P- and S-phase picking models is also available. These upgraded picking models were trained using 2x and 3x
the number of shifted versions of the seismic records originally used in Soto and Schurr (2021). Hence enhancing the performance of the phase picking.

See the class Model() documentation for details on other allowed optional parameters.

Once the models are read into DPP, model information can be retrieved for example by using:

    print(dpp_model.model_detection['best_model'].summary())
    print(dpp_model.model_picking_P['best_model'].summary())
    print(dpp_model.model_picking_S['best_model'].summary())

Then, to run the phase detection on the selected seismic waveforms use:

    dpp_model.run_detection(dpp_config, dpp_data)

This will compute discrete class probability time series from predictions, which are used to obtain preliminary phase picks.

The optional parameter `save_dets = True` (default is `False`) will save a dictionary containing the class probabilities and preliminary picks to `opath/*/pick_stats` if needed for further use.
Here `opath` is the output directory defined in the DPP configuration (see function `set_data()`).

The optional parameter `save_data = True` (default is `False`) will save a dictionary containing the seismic waveform data used for phase detection to the same directory.

Next the phase picking can be run to refine the preliminary picks, using:

    dpp_model.run_picking(dpp_config, dpp_data)

The optional parameter `save_plots = True` (default is `True`) will save figures of individual predicted phase onsets to `opath/*/pick_plots` if `run_mcd=True`.
These figures are similar to the subplots in Figure 3 (Soto and Schurr, 2021).

The optional parameter `save_picks = True` (default is `False`) will save a dictionary containing relevant information on preliminary and refined phase picks to `opath/*/pick_stats`.

The optional parameter `save_stats = True` (default is `True`) will save statistics of predicted phase onsets to the output file `opath/*/pick_stats/pick_stats`.
If `run_mcd=False`, the ouput file will contain the following 4 columns:

`station, phase (P or S), pick number, detection probability, tons (preliminary; UTC)`

If `run_mcd=True`, the output file will contain the previous columns plus the following additional columns with the results from the MCD iterations:

`tons (refined; UTC), tons (preliminary; within picking window), tons (refined; within picking window),
tons_err (before onset), tons_err (after onset), pick class, pb, pb_std`


### 4. Plotting predicted P and S phases

To plot the continuous waveforms together with the predicted P and S phases, use:

    util.plot_predicted_wf_phases(dpp_config, dpp_data, dpp_model)

Three additional optional function parameters allow to modify the plots content.
The parameter `plot_comps` defines which seismogram components are plotted.
The parameter `plot_probs` defines the probability time series of which classes are plotted.
Finally, the parameter `shift_probs` controlls if the plotted probability time series are shifted in time,
according to the optimized hyperparamter values defining the picking window for each class.

For example, the following will plot the predicted picks, the vertical ('Z') and north ('N') seismogram components,
and the probability time series for P- and S-phase classes shifted in time as described above.

    util.plot_predicted_wf_phases(dpp_config, dpp_data, dpp_model, plot_comps=['Z','N'], plot_probs=['P','S'], shift_probs=True)

Please check a more detailed information about all the allowed parameters in the documentation for this function.


## Citation:

If you use this algorithm for research purpose, please cite it as follows:

- Soto, H., & Schurr, B. (2020). DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local Earthquakes based on highly
optimizedConvolutional and Recurrent Deep Neural Networks. EarthArXiv preprint: https://eartharxiv.org/repository/view/1752/.

Please let me know of any bugs found in the code.


## Thanks:

The development of DeepPhasePick method has received financial support from

-  The HAzard and Risk Team (HART) initiative of the GFZ German Research Centre for Geosciences in collaboration with the Institute of GeoSciences, Energy, Water
and Environment of the Polytechnic University Tirana, Albania and the KIT Karlsruhe Institute of Technology.

