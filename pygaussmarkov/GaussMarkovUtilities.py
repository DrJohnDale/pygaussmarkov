# -*- coding: utf-8 -*-
"""
Contains untility functions which can be used in Gaus Markov implementatinos
@author: john
"""
import numpy as np

def generatePMatrix(errors, noMeasurements):
    """
    Takes an error vectror and returns a diagional weight matrix. the weight is 1/error**2
    errors = np list of length equal to noMeasuremetns or 1. If length =1 it will assume that this error is the same for all measurements
    noMeasurements = the number of measurements.
    
    returns a diagional matrix for weights of size noMeasurements X noMeasurements
    """
    if len(errors)>1 and len(errors)!=noMeasurements:
        raise Exception('Error generating weight matrix error array length must be 1 or equal to noMeasurements. len errors = '+str(len(errors))+' noMeasurements = '+str(noMeasurements))
    m = np.eye(noMeasurements)
    if len(errors)==1:
        v = 1.0/(errors[0]**2)
        for i in range(noMeasurements):
            m[i,i] = v
    else:
        for i in range(noMeasurements):
            m[i,i] = 1.0/(errors[i]**2)
    return m

def generatePinvMatrix(errors, noMeasurements):
    """
    Takes an error vectror and returns a diagional weight matrix. the weight is 1/error**2
    errors = np list of length equal to noMeasuremetns or 1. If length =1 it will assume that this error is the same for all measurements
    noMeasurements = the number of measurements.
    
    returns a diagional matrix for weights of size noMeasurements X noMeasurements
    """
    if len(errors)>1 and len(errors)!=noMeasurements:
        raise Exception('Error generating weight matrix error array length must be 1 or equal to noMeasurements. len errors = '+str(len(errors))+' noMeasurements = '+str(noMeasurements))
    m = np.eye(noMeasurements)
    if len(errors)==1:
        v = (errors[0]**2)
        for i in range(noMeasurements):
            m[i,i] = v
    else:
        for i in range(noMeasurements):
            m[i,i] = (errors[i]**2)
    return m

