# Nationalpark Kalkalpen
SVD project and training data for the study area (Kalkalpen Nationalpark) used for the paper:

"A scalable model of vegetation transitions using deep neural networks", by Werner Rammer and Rupert Seidl.


## SVD-project "NPKA"

The folder `project` contains all files for running SVD for the Nationalpark Kalkalpen:

* configuration file (`config.conf`) for SVD
* pre-trained DNNs (`dnn/graph*.pb`), and DNN metadata (`dnn/graph_meta.txt`)
* spatial data of the project area (site data, DEM) (`gis` folder)
* climate data for driving the simulations (folder `climate`)

Model code and executable can be found at https://github.com/SVDmodel/SVD

In a nutshell, to run the model, open the `config.conf` in the SVD executable, and click "Run". Detailed instructions for using the SVD model
are available at https://svdmodel.github.io/SVD/#/svdUI

## Training script and data

The folder `training` contains:

* `npka.py`, `npka_noDB_noD.py`: Python scripts for training a DNN with TensorFlow
* additional python scripts (`climate.py`, `examples.py`, `iLand_env.py`, `utils.py`)
* Training data (`data` folder) that was generated by the process based model [iLand](iland.boku.ac.at); 
the data was derived from multiple PBM runs for the Nationalpark Kalkalpen under 
different climate scenarios (3.4 GB in total)
* additional site data (constant environmental factors for soil) and a table of the used vegetation states

### Network training

The DNN training facilitates the [TensorFlow](tensorflow.org) framework and the 
high-level [Keras](https://www.tensorflow.org/guide/keras) library.

In order to perform network training, TensorFlow needs to [installed](tensorflow.org/install) on your machine.

A good starting point is the `npka.py` script. 

Training is known to work for TensorFlow versions 1.04 to 1.12.
