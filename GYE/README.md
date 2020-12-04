# Greater Yellowstone Ecosystem
SVD project and training data for the study area (Greater Yellowstone Ecosystem, USA) used for the paper:

"Widespread regeneration failure in forests of Greater Yellowstone under scenarios of future climate and fire", by Werner Rammer, Kristin H. Braziunas, Winslow D. Hansen, Zak Ratajczak, A. Leroy Westerling, Monica G. Turner, Rupert Seidl


## SVD-project "GYE"

The folder `project` contains all files for running SVD for the GYE:

* configuration file (`config_full_cc_202004.conf`) for SVD
* pre-trained DNNs (`dnn/frozen_graph36.pb`), and DNN metadata (`dnn/graph_meta_v36.txt`)
* spatial data of the project area (site data, initial state, DEM) (`gis` folder). *Note*: to circumvent GitHub's size policy some files in the `gis`  folder are zipped. Unzip in the same folder before running SVD!
* climate data for driving the simulations for several climate scenarios (folder `climate`). Switch scenarios with the `climate.file` and `climate.sequence` settings in the configuration file.

Model code and executable can be found at https://github.com/SVDmodel/SVD
Note that the last version includes a CPU-only version of TensorFlow. Therefore, running SVD should be rather straightforward (under Windows): just start the `SVDUI.exe` from the `executable` folder.

In a nutshell, to run the model, open the `config_full_cc_202004.conf` in the SVD executable, and click "Run". Detailed instructions for using the SVD model
are available at https://svdmodel.github.io/SVD/#/svdUI

## Training script and data

The folder `training` contains `regen_fail_v36_final.py`, which is the Python script and: 

* loads the training data and sets up the data pipeline (using the Tensorflow `Dataset` API)
* defines the DNN structure with Keras and Tensorflow
* runs the DNN training 
* saves the trained DNN as a frozen graph (later used by the SVD model)


### Network training

The DNN training facilitates the [TensorFlow](tensorflow.org) framework and the 
high-level [Keras](https://www.tensorflow.org/guide/keras) library.

In order to perform network training, TensorFlow needs to [installed](tensorflow.org/install) on your machine.



