# -*- coding: utf-8 -*-
"""
This is the Python code used for the DNN training for the paper
"Widespread regeneration failure in forests of Greater Yellowstone under scenarios of future climate and fire", 
by Werner Rammer, Kristin H. Braziunas, Winslow D. Hansen, Zak Ratajczak, A. Leroy Westerling, Monica G. Turner, Rupert Seidl

The script loads training data (generated with the model iLand),
creates a DNN structure, runs the training, and saves a "frozen" DNN model.


@author: wrammer
"""

import tensorflow as tf

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Embedding

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



# split
filenames = ["train1_00", "train1_01", "train1_02", "train1_03", "train1_04", "train1_05", "train1_06", "train1_07", "train1_08", "train1_09", "train1_10", "train1_11", "train1_12", "train1_13", "train1_14", "train1_15", "train1_16", "train1_17", "train1_18", "train1_19", "train1_20", "train1_21", "train1_22", "train1_23", "train1_24", "train1_25", "train1_26", "train1_27", "train1_28", "train1_29", "train1_30", "train1_31", "train1_32", "train1_33", "train1_34", "train1_35", "train1_36", "train1_37", "train1_38", "train1_39", "train1_40", "train1_41", "train2_00", "train2_01", "train2_02", "train2_03", "train2_04", "train2_05", "train2_06", "train2_07", "train2_08", "train2_09", "train2_10", "train2_11", "train2_12", "train2_13", "train2_14", "train2_15", "train2_16", "train2_17", "train2_18", "train2_19", "train2_20", "train2_21", "train2_22", "train2_23", "train2_24", "train2_25", "train2_26", "train2_27"]
filenames_eval = ["eval1_00", "eval1_01", "eval1_02", "eval1_03", "eval1_04", "eval1_05", "eval1_06", "eval1_07", "eval1_08", "eval1_09", "eval1_10", "eval1_11", "eval1_12", "eval1_13", "eval1_14", "eval1_15", "eval1_16", "eval1_17", "eval1_18", "eval1_19", "eval1_20", "eval1_21", "eval1_22", "eval1_23", "eval1_24", "eval1_25", "eval1_26", "eval1_27", "eval1_28", "eval1_29", "eval1_30", "eval1_31", "eval1_32", "eval1_33", "eval1_34", "eval1_35", "eval1_36", "eval1_37", "eval1_38", "eval1_39", "eval1_40", "eval1_41", "eval1_42", "eval1_43", "eval1_44", "eval1_45", "eval1_46", "eval1_47", "eval1_48", "eval1_49", "eval1_50", "eval1_51", "eval2_00", "eval2_01", "eval2_02", "eval2_03", "eval2_04", "eval2_05", "eval2_06", "eval2_07", "eval2_08", "eval2_09", "eval2_10", "eval2_11", "eval2_12", "eval2_13", "eval2_14", "eval2_15", "eval2_16", "eval2_17", "eval2_18", "eval2_19", "eval2_20", "eval2_21", "eval2_22", "eval2_23", "eval2_24", "eval2_25", "eval2_26", "eval2_27", "eval2_28", "eval2_29", "eval2_30", "eval2_31", "eval2_32", "eval2_33", "eval2_34"]

filenames = [ "../SVD1R/split/"+x for x in filenames ]
filenames_eval = [ "../SVD1R/split/"+x for x in filenames_eval ]

from random import shuffle
shuffle(filenames)
shuffle(filenames_eval)

"""
#################################################################################
# each training file is in a CSV format and has 248 columns:
# 0..5: state, restime, ruId, distance, nitrogen, sand  
# 246,247: targetState, targetTime
# 6-245: climate 120x temp (6:125), 120x precip (126:245)
# 248: soil depth (cm)
#################################################################################
"""

record_defaults = [tf.float32], [tf.float32], [tf.int32], [tf.float32]*3, [tf.float32] * 240, [tf.int32]*2, [tf.float32]
record_defaults = [item for sublist in record_defaults for item in sublist] # https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python


"""
Data pipeline
=============

The pipeline is as follows:
    
    * The files are read by CsvDataset(); one training example per line
    * the `_parse_function()` is called for each example; the function extracts
    the bits from the CSV file, reshapes and transforms the data and returns a dictionary
    * The data is cached (`cache()`) to speed up training (parsing each line takes a while)
    * the data stream is shuffled randomly (with a fairly large window), and 
    * batched


"""


def _parse_function(*example):
     ex = ({'state': tf.cast(tf.reshape( (example[0]-1),[1]), dtype=tf.int32),
           'restime': tf.cast(tf.reshape(example[1]/10, [1]), dtype=tf.float32),
           'distance': tf.cast(tf.reshape(example[3]/1000,[1]), dtype=tf.float32),
           'site': tf.cast(tf.reshape( tf.stack([ example[4]/100, example[5]/100 , example[248]/100]), [3]), dtype=tf.float32), # nitrogen, sand
           'climate': tf.concat([tf.reshape(tf.transpose([ x/10. for x in example[6:126] ]), [ 10, 12]),
                                tf.reshape(tf.transpose([x/20. for x in example[126:246]]), [ 10, 12])], 1),
           },
           { 'targetState': tf.reshape(example[246]-1, [1]),
            'targetTime': tf.reshape(example[247]-1, [1])})
     return ex



dataset_f = tf.contrib.data.CsvDataset(filenames, record_defaults) 
dataset = dataset_f.map(_parse_function, num_parallel_calls=3).cache("/tmp/tf_cache/cachet").apply(tf.contrib.data.shuffle_and_repeat(256000)).batch(256)

dataset_eval_f = tf.contrib.data.CsvDataset(filenames_eval, record_defaults) 
dataset_eval = dataset_eval_f.map(_parse_function, num_parallel_calls=2).cache("/tmp/tf_cache/cachee").apply(tf.contrib.data.shuffle_and_repeat(25600)).batch(256)

"""
The DNN
=======

We uses Keras on top of Tensorflow to define the DNN.

The network has five different inputs (corresponding to the _parse_function() above):
    * the current state (`state`)
    * the residence time the cell already is in the state (`restime`)
    * the distance to the closest potential seed source (`distance`)
    * three site variables (plant available nitrogen (site fertility), soil depth, fraction of sand, `site`)
    * climate variables (`climate`): monthly means of temperature and precipitation for the next 10 years (10 x 12 x 2 = 240 values)

The network has also two outputs:
    
    * The next state (or the same if no state change occurs): `targetState`
    * The time of the state change (years from now, 10 if no change happens): `targetTime`
    
    

"""



from tensorflow.keras.layers import Input,  concatenate
from tensorflow.keras.layers import Dense, Dropout,  Flatten, Reshape
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

### define network

NClasses = 22 # There are 22 distinct classes in the GYE training data set
NTimeSteps = 10 # Predictions are made for a window of 10 years

# tf.keras.backend.clear_session()

# Inputs
sinput = Input(shape=(1,), dtype="int32",  name='state') 
timeinput = Input(shape=(1,), name='restime')
climinput = Input(shape=(10,24,), name='climate')
siteinput = Input(shape=(3,), name='site')
distanceinput = Input(shape=(1, ), name='distance')

# For the state we use an Embedding layer with 6 output dimmensions
sinput_em = Embedding(output_dim=6, input_dim=NClasses, input_length=1, name="stateem")(sinput)
sinput_em = Flatten()(sinput_em)

#  The climate runs through a fully connected layer, and gets flattened
clim = Dense(64, activation="relu")(climinput)
clim = Reshape( (640,) )(clim)
clim = Dropout(rate=0.25)(clim)

# The climate signal is compressed to 16 values
envx = Dense(64, activation="relu", name="envx")(clim)
clim = Dropout(rate=0.25)(clim)
envx = Dense(64, activation="relu", name="envx3")(envx)
envx = Dense(16, activation="relu", name="envx2")(envx)


# The different inputs are joined together
# 1x state, 1x time,  16x climate, 1x dist, 3x site 
minput = concatenate([sinput_em, timeinput,  envx, distanceinput, siteinput])

# two separate branches start from this layer, 
# both ending in a Softmax layer for the final classification:
# 1 layer (fully connected layers, dropout) for the prediction of transition time (`targetTime`)
# 1 layer (FC layers, dropout) for the prediction of `targetState`

# target time branch
c_time = Dense(256, activation="relu")(minput)
c_time = Dropout(0.25)(c_time)
c_time = Dense(256, activation="relu")(c_time)
out_time = Dense(NTimeSteps, name="targetTime", activation="softmax")(c_time)

# target state branch
x = Dense(512, activation="relu")(minput)
x = Dropout(rate=0.4)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(rate=0.4)(x)
x = Dense(512, activation="elu")(x)
out = Dense(NClasses, activation="softmax", name="targetState")(x)

input_list = [sinput, climinput,  timeinput, distanceinput, siteinput]

# create the model
model = Model(inputs=input_list, outputs=[out, out_time]) 

# compile the model
# the loss in state is more important than the loss in time (loss_weights)
model.compile(loss={'targetState':'sparse_categorical_crossentropy', 
                    'targetTime': 'sparse_categorical_crossentropy'},
              loss_weights={'targetState':1, 'targetTime': 0.5},
              optimizer=tf.keras.optimizers.Adam(lr=0.001),  
              metrics=['accuracy', 'sparse_categorical_accuracy'])

# add tensorboard callback
tbCallBack = TensorBoard(log_dir='tensorboard/v36')

# Scheme for learning rate: reduce LR by 50% if there no further progress for 3 epochs
tbReduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=3, min_lr=0.00001)

"""
Network training
================

Run the network training. Progress can be monitored in Tensorboard.
"""


model.fit(dataset, 
          validation_data=dataset_eval, 
          epochs=40, steps_per_epoch=16200, 
          validation_steps=2025, 
          callbacks=[tbCallBack, tbReduceLr])



"""
Save the model
==============

The last step is to create a "frozen" DNN and store it to disk.
This version is the loaded in SVD. Saving a frozen model is a bit tedious,
and works differently well with different versions of Tensorflow.

Below is code that at least worked that time.

"""
    
    
############# way to save a model - v2  ##################
# (1) save the graph and a checkpoint (important: use Keras-session!)
# (2) Use the "freeze_graph()" function to combine graph + checkpoint (take care of the names of output layers)
####################################################

# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html part IV
from tensorflow.keras import backend as K
sess = K.get_session()

# Use "simple_save" 
### THIS works, 2019-06-06
tf.saved_model.simple_save(
    sess,
    "expSavedModel36", # directory - must not exist
    {'state': sinput, 'restime': timeinput, 'distance': distanceinput,  'climate': climinput, 'site': siteinput},
    {"targetState": out, "targetTime": out_time},
    legacy_init_op=None)

from tensorflow.python.tools import freeze_graph


freeze_graph.freeze_graph(input_graph="", input_saver="", input_binary= False, input_checkpoint= "", 
                          output_node_names = "targetState/Softmax,targetTime/Softmax", restore_op_name="", filename_tensor_name="", 
                          output_graph="SaveFiles/frozen_graph36.pb", clear_devices=True, initializer_nodes="", 
                          input_saved_model_dir='expSavedModel36')

