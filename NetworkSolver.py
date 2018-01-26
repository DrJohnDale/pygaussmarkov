# -*- coding: utf-8 -*-
from pygaussmarkov.ParametricFreeNetworkConstraintSolver import ParametricFreeNetworkConstraintSolver
import numpy as np
import pygaussmarkov.GaussMarkovUtilities as GMUtils
import numpy.linalg as la
from operator import add, sub
import math

class NetworkSolver(ParametricFreeNetworkConstraintSolver):
    def __init__(self,measurements, uncertainties, measurementsMap,noStages,noMarkers, deltaChiSqToStop = 0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        measurements are all the measurements
        uncetainties are the uncertainties on the measurements
        measurementsMap is a list of lists. The inner lists have two elements, the first is the stage number, the second is the network marker number
        """
        self.measurements = measurements
        self.uncert = uncertainties
        self.noStages = noStages
        self.noMarkers = noMarkers
        self.measurementMap = measurementsMap
        super().__init__(6,decomposition=ParametricFreeNetworkConstraintSolver.USE_SVD,deltaChiSqToStop = deltaChiSqToStop,dampingFactor = dampingFactor,useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch, recordHistory = recordHistory)
        
    
    def getListOfParameterNames(self):
        stageParams = ["stage_"+str(s)+'_'+ax for s in range(self.noStages) for ax in ['x','y','z']]
        markerParams = ["marker_"+str(s)+'_'+ax for s in range(self.noMarkers) for ax in ['x','y','z']]
        
        return stageParams + markerParams
    
    def getAandAT(self,variables):
        noMeas = len(self.measurements)
        m = np.matrix(np.zeros([noMeas,self.getNumberOfFreeParameters()]))
        paramNames = self.getListOfParameterNames()
        for i in range(noMeas):
            mc = self.measurementMap[i]
            sx,sy,sz,mx,my,mz = self.getPositionsFromCoordinates(self.measurementMap[i],variables)
            nsx,nsy,nsz,nmx,nmy,nmz = self.getPositionsFromCoordinates(self.measurementMap[i],paramNames)
            
            v = np.sqrt((-sx+mx)**2 +(-sy+my)**2+(-sz+mz)**2)
            
            if not self.fixed[nsx]:
                m[i,mc[0]*3] = (sx - mx)/v
            
            if not self.fixed[nsy]:
                m[i,mc[0]*3+1] = (sy - my)/v
            
            if not self.fixed[nsz]:
                m[i,mc[0]*3+2] = (sz - mz)/v
                
            if not self.fixed[nmx]:
                m[i,self.noStages*3 + mc[1]*3] = -(sx - mx)/v
            
            if not self.fixed[nmy]:
                m[i,self.noStages*3 + mc[1]*3 + 1] = -(sy - my)/v
            
            if not self.fixed[nmz]:
                m[i,self.noStages*3 + mc[1]*3 + 2] = -(sz - mz)/v
        
        mt = m.transpose()
        return m, mt
    
    def getNamesFromCoordinates(self,mc,paraNames):
        sx = paraNames[mc[0]*3]
        sy = paraNames[mc[0]*3 + 1]
        sz = paraNames[mc[0]*3 + 2]
        
        mx = paraNames[self.noStages*3 + mc[1]*3]
        my = paraNames[self.noStages*3 + mc[1]*3 + 1]
        mz = paraNames[self.noStages*3 + mc[1]*3 + 2]
        
        return sx,sy,sz,mx,my,mz
        
    def getPositionsFromCoordinates(self,mc,variables):
        sx = variables[mc[0]*3]
        sy = variables[mc[0]*3 + 1]
        sz = variables[mc[0]*3 + 2]
        
        mx = variables[self.noStages*3 + mc[1]*3]
        my = variables[self.noStages*3 + mc[1]*3 + 1]
        mz = variables[self.noStages*3 + mc[1]*3 + 2]
        
        return sx,sy,sz,mx,my,mz
            
    def getFx(self,variables):
        noMeas = len(self.measurements)  
        v = np.zeros(noMeas)
        for i in range(noMeas):
            sx,sy,sz,mx,my,mz = self.getPositionsFromCoordinates(self.measurementMap[i],variables)
            v[i] =np.sqrt((-sx+mx)**2 +(-sy+my)**2+(-sz+mz)**2)
        return v
        
    def getP(self):
        return GMUtils.generatePMatrix(self.uncert,len(self.measurements))
        
    def getMeasurements(self):
        return self.measurements
    
    def fixParameter(self,name):
        print("fixing/freeing parameters disabled, nothing has changed")
        
    def freeParameter(self,name):
        print("fixing/freeing parameters disabled, nothing has changed")
    
    
def main():
    # define true tracker positions
    trackerPositions = [0,0,0, 0,0,1,  0,0.5,0.5, 0.5,0,0.5]
    noTrackingPositions = math.floor(len(trackerPositions)/3)
    # define true measurements positions
    measurementPositions = [-1,0,-0.5, 0,1,-0.5, -1,0,-0.5,     1,0,0.5, 0,1,0.5, -1,0,1.5,     -1,0,-0.5, 0,1,1.5, -1,0,1.5,]
    #measurementPositions = [-0.5,1.0,-0.5, -0.5,1.0,0.5, 0.0,1.5,0.5, 0.0,1.5,0.5, 0.5,1.0,-0.5, 0.5,1.0,0.5, 1.0,1.5,-0.5, 1.0,1.5,0.5, 1.5,1.0,-0.5, 1.5,1.0,0.5] 
    noMeasurementPositions = math.floor(len(measurementPositions)/3)
    # combine tracker and measurement positions
    variables = trackerPositions + measurementPositions
    # define which trackers to which measurement positions
    measurementsMap = [[t,m] for t in range(noTrackingPositions) for m in range(noMeasurementPositions)]
    # generate distances
    measurements = np.zeros(len(measurementsMap))
    for i,mc in enumerate(measurementsMap):
        sx = variables[mc[0]*3]
        sy = variables[mc[0]*3 + 1]
        sz = variables[mc[0]*3 + 2]
        
        mx = variables[noTrackingPositions*3 + mc[1]*3]
        my = variables[noTrackingPositions*3 + mc[1]*3 + 1]
        mz = variables[noTrackingPositions*3 + mc[1]*3 + 2]
        measurements[i] =np.sqrt((-sx+mx)**2 +(-sy+my)**2+(-sz+mz)**2)
    
    # add noise to distances
    measurementUncert = 0.001
    noise = np.random.normal(0, measurementUncert, len(measurementsMap))
    measurements = np.array(list(map(add, measurements, noise)))
    print(measurements)
    
    # initalise fitter
    fitter = NetworkSolver(measurements,np.array([measurementUncert]),measurementsMap,noTrackingPositions,noMeasurementPositions)
    fitter.printChiSqEachIteration = True
    fitter.printChiSqDiffEachIteration = True
#    
#    A,AT = fitter.getAandAT(np.array(variables))
#    ATPA = AT*p*A
#    u,s,v = la.svd(ATPA)
#    print('u',u)
#    print('s',s)
#    print('v',v)     
    
    # run fitter
    variablesStart = [p+np.random.normal(0, 0.1, 1) for p in variables]
    out, err = fitter.solve(np.array(variablesStart))
    print('res',out,'err',err)
    
    
if __name__ == '__main__':
    main()