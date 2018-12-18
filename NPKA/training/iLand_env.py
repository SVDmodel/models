# -*- coding: utf-8 -*-
"""
############################################################################################
 DNN Training utilties

Rammer, W, Seidl, R: A scalable model of vegetation transitions using deep learning
#############################################################################################
"""
from climate import Climate

class Environment:
    clim_dict = None

    def __init__(self, gpp_climate, datapath):
        if gpp_climate:
            
                        # create the instances: hard coded paths....
            bl = Climate(datapath+'/NP_env.csv', 
                         climate_change=False, gpp_climate=True, delimiter=' ')   
            arpege = Climate(datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=True, delimiter=' ')   
            ictp = Climate(datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=True, delimiter=' ')   
            remo = Climate(datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=True, delimiter=' ')  
            # fill the dict: for now hard coded....
            self.clim_dict = { 1: bl, 2: bl, 11: arpege, 12: arpege, 21: ictp, 22: ictp, 31: remo, 32: remo}

        else:
            # create the instances: hard coded paths....
            bl = Climate('../project/climate/mean_clim_bl.txt', datapath+'/NP_env.csv',
                         climate_change=False, gpp_climate=False)   
            arpege = Climate('../project/climate/mean_clim_bl.txt', datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=False)   
            ictp = Climate('../project/climate/mean_clim_ictp.txt', datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=False)   
            remo = Climate('../project/climate/mean_clim_remo.txt', datapath+'/NP_env.csv',
                         climate_change=True, gpp_climate=False)  
            # fill the dict: for now hard coded....
            self.clim_dict = { 1: bl, 2: bl, 11: arpege, 12: arpege, 21: ictp, 22: ictp, 31: remo, 32: remo}
            
    """ Retrieve the climate data for a given run, cell, and year(s). All three variables are included in the example data.
        cellId: resource unit,
        year: year of the simulation
        runId: the (hard coded) unique run Id
    
    """
    def climateData(self, cellId, year, runId):
        return (self.clim_dict[ runId ].data(cellId, year) )
    
    """ return a tuple with (standardized) nitrogen / soil depth
    """
    def soilData(self, cellId):
        return self.clim_dict[ 1 ].soil_table[ cellId ]

    """ return the number of values for a year """
    def NclimateValues(self):
        return self.clim_dict[1].NValues

