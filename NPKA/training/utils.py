# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:22:55 2017

@author: wrammer
"""

from keras.models import model_from_yaml
def saveModel(model, filename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(filename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(filename+".h5")
    print("Saved model to disk")
    
def loadModel(filename):
    # load YAML and create model
    yaml_file = open(filename + '.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(filename + ".h5")
    print("Loaded model from disk")
    return loaded_model
