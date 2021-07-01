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

To set the parameters selecting the waveform data on which DeepPhasePick is applied, use:

    dpp_config.set_data()

For example, to select the waveforms from stations `PB01` and `PB02` (network `CX`), and channel `HH` which are stored in
the archive directory `data`, and save the results in directory `out`, run:

    dpp_config.set_data(
        stas=['PB01', 'PB02'],
        net='CX',
        ch='HH',
        archive='data',
        opath='out'
    )

To set the parameters defining how seismic waveforms is processed before phase detection, use:

    dpp_config.set_data_params()

For example, the following will apply a highpass filter (> .5 Hz) and resample the data to 100 Hz (if that is not already the data sampling rate) before running the detection:

    dpp_config.set_data_params(
        samp_freq=100.,
        st_filter=True,
        filter_type='highpass',
        filter_fmin=.2
    )

DPP will be applied on the selected seismic data (see function `set_data()`) in the time windows defined using:

    dpp_config.set_time()

For example, to create 30-min (1800-seconds) time windows in the period between
`2015-04-03T00:00:00` and `2015-04-03T02:00:00` (2 hours), use:

    dpp_config.set_time(dt_iter=1800., tstart="2015-04-03T00:00:00", tend="2015-04-03T02:00:00")

Note that the windows will have the same duration except for the last window, which will be filled with the remainder data until `tend` in case
`dt_iter + tstart(last window) > tend`.

To set the parameters defining how predicted discrete probability time series are computed when running phase detection on seismic waveforms, use:

    dpp_config.set_trigger()

For example, the following will compute the discrete probability time series in phase detection every 20 samples, using a probability threshold of 0.95 for P- and S-phases:

    dpp_config.set_trigger(n_shift=20, pthres_p=[0.95, 0.001], pthres_s=[0.95, 0.001])

To set the parameters applied in optional conditions for refining preliminary picks obtained from phase detection, use:

    dpp_config.set_picking()

For example, the following will remove preliminary picks which are presumed false positive, by applying all of the four optional conditions described in the
Supplementary Material of Soto and Schurr (2021). This is the default and recommended option, especially when dealing with very noise waveforms.
Please refer to the Text S1 in the above-mentioned Supplementary Material to see how the different user-defined parameters are used to remove preliminary picks.

Then refined pick onsets and their time uncertainties will be computed by applying 20 iterations of Monte Carlo Dropout.

    dpp_config.set_picking(
        op_conds=['1','2','3','4'],
        dt_ps_max=25.,
        dt_sdup_max=2.,
        dt_sp_near=1.5,
        tp_th_add=1.5,
        run_mcd=True
        mcd_iter=20,
    )

More details on the arguments accepted by each of these functions can be seen from the corresponding function documentation.


### 2. Seismic Data

DPP method is applied on three-component MiniSEED seismic waveforms.

To read the seismic waveforms into DPP an instance of the class Data() needs to be created, for example using:

    dpp_data = data.Data()

Then, the data can be read into DPP from a local archive directory using:

    dpp_data.read_from_archive(dpp_config)

The local archive needs to have the following commonly used structure:

**archive/YY/NET/STA/CH**

Here **YY** is year, **NET** is the network code, **STA** is the station code and **CH** is the channel code (e.g., HH) of the seismic streams.
Example data is included in the **archive** directory, where new data, on which DeepPhasePick will be applied, can be added by users.

Alternatively, waveforms can be read from a local directory with no specific structure. For example using:

    dpp_data.read_from_directory(dpp_config)


### 3. Phase Detection and Picking

In order to run the phase detection and picking stages, an instance of the class Model() needs to be created, for example using:

    dpp_model = model.Model()

This reads the optimized trained models into DPP.
By default, the trained model weights and other relevant model information obtained from the hyperparameter optimization are read from the following directories:

**detection/20201002**, which contains the files related to the optimized phase detection model,

**picking/20201002/P**, which contains the files related to the optimized P-phase picking model, and

**picking/20201002/S**, which contains the files related to the optimized S-phase picking model.

Here `20201002` is a string indicating the version of each model, defined by the optional arguments `version_det`, `version_pick_P`, `version_pick_S` of Model().

New trained models (with their corresponding versions) will be added in the near future, which can be accessed by modifying this argument.

See the class documentation for details on other optional arguments.

Once the models are read into DPP, model information can be retrieved for example by using:

    print(dpp_model.model_detection['best_model'].summary())
    print(dpp_model.model_picking_P['best_model'].summary())
    print(dpp_model.model_picking_S['best_model'].summary())

Then, to run the phase detection on the selected seismic waveforms, use:

    dpp_model.run_detection(dpp_config, dpp_data)

This will compute discrete class probability time series from predictions, which are used to obtain preliminary phase picks.

Use the optional argument `save_dets = True` (by default is `False`) to save a dictionary containing the class probabilities and preliminary picks for further use.

Use the optional argument `save_data = True` (by default is `False`) to save a dictionary containing the seismic waveform data used for phase detection if needed.

Next the phase picking can be run to refine the preliminary picks, using:

    dpp_model.run_picking(dpp_config, dpp_data)

Use the optional argument `save_plots = True` (by default is `True`) to create figures of individual predicted phase onsets.

Use the optional argument `save_stats = True` (by default is `True`) to save statistics of predicted phase onsets.

Use the optional argument `save_picks = True` (by default is `False`) to save a dictionary containing relevant information on preliminary and refined phase picks.


### 4. Plotting results

...

### 4. DeepPhasePick worflow -- OLD

(3.3) Plotting continuous waveform with predicted P and S phases, and corresponding predicted probability time series.

Two type of output plots can be generated.
The function **plot_predicted_wf_phases** creates plots including predicted phase onsets on seismic waveforms.
The function **plot_predicted_wf_phases_prob** additionally includes predicted probability time series.
The function **plotly_predicted_wf_phases** (requires the installation of the library plotly), creates interactive plots of the predicted phase onsets
on seismic waveforms and predicted probability time series, where predicted picks can be visualized in more detailed.
In all the above plots, the phase onsets shown are the refined picks.
Moreover, the user-defined parameter **dct\_out['plot_comps']** controls which seismogram components are plotted.

## Citation:

If you use this algorithm for research purpose, please cite it as follows:

- Soto, H., & Schurr, B. (2020). DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local Earthquakes based on highly
optimizedConvolutional and Recurrent Deep Neural Networks. EarthArXiv preprint: https://eartharxiv.org/repository/view/1752/.

Please let me know of any bugs found in the code...


## Thanks:

The development of DeepPhasePick method has received financial support from

-  The HAzard and Risk Team (HART) initiative of the GFZ German Research Centre for Geosciences in collaboration with the Institute of GeoSciences, Energy, Water
and Environment of the Polytechnic University Tirana, Albania and the KIT Karlsruhe Institute of Technology.

