# -*- coding: utf-8 -*-
"""
#############################################################################################
# Training for the DNN
#
# Rammer, W, Seidl, R: A scalable model of vegetation transitions using deep learning
#############################################################################################

@author: Werner Rammer
"""

# SVD utilities
from iLand_env import Environment
from examples import startWorkers, stopWorkers, fetchExamples, queueState
from utils import saveModel, loadModel

# system libraries
from random import shuffle
import pandas as pd

#OPT_Neighbors = True # v5f
OPT_Neighbors = True   ## include neighbors
OPT_GPPClimate = False # use GPP compressed climate
OPT_Distance = True # use distance from landscape edge

data_path = 'data'


#examples, ridx = loadExamples() # this takes a while
unique_states = pd.read_csv("data/unique.states.pruned.csv")

# files: structure: all.pruned_<runId>_<fileIndex>.csv 
# file-index: starts with 0
def runcode(run, n, offset=0):
    return [str(run) + '_' + str(offset+x+1) for x in range(n)]
    
# data file names encode the run (and the climate scenario)
# Runs: 1,2: baseline, 11,12: arpege, 21,22: ictp, 31,32: remo
# training data: baseline, arpege, remo (BL, C1, C2)
# eval data: ictp  (C3)
train_files = runcode(1, 32) + runcode(2,32) + runcode(11,31) + runcode(12,31) + runcode(31,32) + runcode(32,32)
extra_files = runcode(1, 4, 100) + runcode(2,4,100) + runcode(11,4,100) + runcode(12,4,100) + runcode(31,4,100) + runcode(32,4,100)

val_files = runcode(21, 31) + runcode(22, 31)
# shuffle in place
shuffle(train_files)
shuffle(val_files)


# Training examples are combined on the fly from 
# vegetation transitions, climate and site data
env = Environment(gpp_climate= OPT_GPPClimate, datapath=data_path)


    
### set up of the DNN

# We utilize the KERAS library
from keras.models import  Model
#from keras.layers.core import Dense, Dropout,  Flatten,  RepeatVector, Reshape
from keras.layers.core import Dense, Dropout,  RepeatVector, Reshape

from keras.layers import Input, concatenate, TimeDistributed
from keras.layers.embeddings import Embedding
# from keras.layers.advanced_activations import LeakyReLU
#import matplotlib.pyplot as plt
from keras import backend as K

from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

#K.clear_session()
#K.set_session( tf.Session( config=tf.ConfigProto(log_device_placement=True)) )


    
NClasses = unique_states.shape[0]
NTimeSteps = 10 # we train with a 10 years for residence time
NNeighbors = 62 # number of neighbor values (2 x number of species)

# Inputs
sinput = Input(shape=(1,), dtype='int16', name='state_input')
timeinput = Input(shape=(1,), name='time_input')
climinput = Input(shape=(NTimeSteps, env.NclimateValues(),), name='clim_input') # 120, 24, 40
siteinput = Input(shape=(2,), name='site_input')
neighborinput = Input(shape=(NNeighbors, ), name='neighbor_input')
distanceinput = Input(shape=(1,), name='distance_input')

# embedding layer for the input state
sinput_em = Embedding(output_dim=16, input_dim=NClasses, input_length=1, name="state")(sinput)
sinput_em = Reshape( (16,) )(sinput_em)

# concatenate the climate and site input
siter = RepeatVector(NTimeSteps)(siteinput)
envinput = concatenate([climinput, siter])

# climate data
clim = TimeDistributed(Dense(64, activation="elu"))(envinput)

envx = Dense(64, activation="elu", name="post_clim")(clim)
envx = Reshape( (640,) )(envx) # transform to a flat tensor
envx = Dense(128, activation="elu")(envx)

# combine all data streams to a single tensor
if OPT_Neighbors:
    if OPT_Distance:
        minput = concatenate([sinput_em, timeinput, neighborinput, envx, distanceinput])
    else:    
        minput = concatenate([sinput_em, timeinput, neighborinput, envx])
else:
    minput = concatenate([sinput_em, timeinput,  envx])

# for predicting the residence time
c_time = Dense(256, activation="elu")(minput)
c_time = Dropout(0.25)(c_time)
c_time = Dense(256, activation="elu")(c_time)
# output tensor for residence time
out_time = Dense(NTimeSteps, name="time_out", activation="softmax")(c_time)

# for predicting state
x = Dense(512, activation="elu")(minput)
x = Dropout(0.25)(x)
x = Dense(512, activation="elu")(x)
# output for state probabilities
out = Dense(NClasses, activation="softmax", name="out")(x)

if OPT_Neighbors:
        if OPT_Distance:
            input_list = [sinput, climinput, siteinput, neighborinput, timeinput, distanceinput]
        else:
            input_list = [sinput, climinput, siteinput, neighborinput, timeinput]
else:
    input_list = [sinput, climinput, siteinput, timeinput]

   
# create and build the Keras model
model = Model(inputs=input_list, outputs=[out, out_time]) 
model.compile(loss={'out':'sparse_categorical_crossentropy', 'time_out': 'sparse_categorical_crossentropy'},
              loss_weights={'out':1, 'time_out': 0.5},
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

# enable tensorboard callback - specify a target file!
tbCallBack = TensorBoard(log_dir='e:/tmp/tb_npka/npka_test', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=5, embeddings_layer_names=['state'], embeddings_metadata='/tmp/tb_npka/metadata2.tsv')

# automatically save model checkpoints 
tbCheckPoint = ModelCheckpoint('checkpoint/weights.{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# scheme for reducing the learning rate by a factor of 0.5 after 3 epochs without improvement
tbReduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=3, min_lr=0.00001)

# start the data input pipeline
startWorkers(batch_size=128, environment=env, 
             train_files=train_files, 
             val_files=val_files, 
             threads_train=3, threads_val=2, 
             log_file="log.txt", datapath=data_path)

queueState()

##### Run the training #####
history = model.fit_generator(fetchExamples(False), steps_per_epoch=20000, epochs=60,
                              validation_data=fetchExamples(True), validation_steps=5000,  callbacks=[ tbReduceLr, tbCallBack, tbCheckPoint]) 


## stop the input pipeline again
stopWorkers()


#################################################################################################
#### Postprocessing 
#################################################################################################


saveModel(model, "np_save")

lmodel = loadModel("np_save")
lmodel = model
# load paramters from checkpoint file
lmodel.load_weights("checkpoint/weights.02-0.82.hdf5")

model.load_weights("checkpoint/weights.36-0.82.hdf5")

model.load_weights("checkpoint/weights.16-0.84.hdf5")



############# way to save a model ##################
# (1) (not sure if necessary): fix "test" mode
# (2) save the graph and a checkpoint (important: use Keras-session!)
# (3) Use the "freeze_graph()" function to combine graph + checkpoint (take care of the names of output layers)
####################################################


# (1)
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html part IV
sess = K.get_session()

K.set_learning_phase(0)  # all new operations will be in test mode from now on

# serialize the model and get its weights, for quick re-building
config = lmodel.get_config()
weights = lmodel.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
new_model = Model.from_config(config) # error.... model_from_config() -> error!!
new_model.set_weights(weights)

print(new_model.summary())
print(lmodel.summary())
print(config['layers'])
# (2)
# http://ludicrouslinux.com/tensorflow-and-c
sess = K.get_session()
tf.train.write_graph(sess.graph_def, "SaveFiles", "Graph.pb")
tf.train.Saver().save(sess, "checkpoint/model.ckpt")

# (3)
# Freeze the graph
from freeze_graph import freeze_graph

checkpoint_state_name = "checkpoint"
input_graph_name = "SaveFiles/Graph.pb"
output_graph_name = "SaveFiles/frozen_graph.pb"
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = "checkpoint/model.ckpt"
output_node_names = "out/Softmax,time_out/Softmax"  ### view vars out, out_time..., but without :0
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = False
freeze_graph(input_graph_name, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_name,
                          clear_devices,"")


### try loading frozen model again:
# http://cv-tricks.com/how-to/freeze-tensorflow-models/

frozen_graph="SaveFiles/graph_v26.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )

