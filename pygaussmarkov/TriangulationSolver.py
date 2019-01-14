# -*- coding: utf-8 -*-
"""
module contains the TriangulationSover object wihch can sovle triangulatin problems
@author: john
"""

from pygaussmarkov.ParametricSolver import ParametricSolver
import numpy as np
import pygaussmarkov.GaussMarkovUtilities as GMUtils
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TriangulationSolver(ParametricSolver):
    """ triangulation with parametric solver """
    def __init__(self, xPos, yPos, zPos,measurements, uncertainties,deltaChiSqToStop = 0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        xPos = the x positions
        yPos = the y positions
        measurements = the distances
        uncertainties = the measurement uncertainties
        deltaChiSqToStop = if the change in chiSq is smaller than this the fitter will stop. Default = 0.01
        dampingFactor = constant to chagne convergence speed. Default 1
        useDampedGaussNeutonLineSearch  weather to use the damped gauss neuton method. default = False
        """
        super().__init__(deltaChiSqToStop = deltaChiSqToStop,dampingFactor = dampingFactor,useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch,recordHistory=recordHistory)
        self.xPos = xPos
        self.yPos = yPos
        self.zPos = zPos
        self.measurements = measurements
        self.uncert = uncertainties
    
    def x_parName(self):
        return 'X'
    
    def y_parName(self):
        return 'Y'
    
    def z_parName(self):
        return 'Z'
    
    def getListOfParameterNames(self):
        return [self.x_parName(),self.y_parName(),self.z_parName()]
    
    def getAandAT(self,variables):
        m = np.matrix(np.zeros([len(self.xPos),self.getNumberOfFreeParameters()]))
        for i in range(len(self.xPos)):
            skipVal = 0
            
            v = np.sqrt((-variables[0]+self.xPos[i])**2 +(-variables[1]+self.yPos[i])**2+(-variables[2]+self.zPos[i])**2);
            
            if not self.fixed[self.x_parName()]:
                m[i,0] = (variables[0] - self.xPos[i])/v
            else:
                skipVal+=1
            
            if not self.fixed[self.y_parName()]:
                m[i,1-skipVal] = (variables[1] - self.yPos[i])/v
            else:
                skipVal+=1
            
            if not self.fixed[self.z_parName()]:
                m[i,2-skipVal] = (variables[2] - self.zPos[i])/v
        
        mt = m.transpose()
        return m, mt
        
    def getFx(self,variables):
        return np.sqrt((-variables[0]+self.xPos)**2 +(-variables[1]+self.yPos)**2+(-variables[2]+self.zPos)**2)
        
    def getP(self):
        return GMUtils.generatePMatrix(self.uncert,len(self.measurements))
        
    def getMeasurements(self):
        return self.measurements
                
                
def run_example():
    xPos = np.array([0.0,1.0,2.0,-1.0,-2.0])
    yPos = np.array([0.0,1.0,0.0,1.0,0.0])
    zPos = np.array([0.0,1.0,0.0,0.0,-2.0])
    x0 = 1.0
    y0 = -2.0
    z0 = 0.0
    noiseWidth = 0.001
    noise = np.random.normal(0, noiseWidth, len(xPos))
    distances = np.sqrt((xPos-x0)**2 +(yPos-y0)**2+(zPos-z0)**2) + noise
    print(distances)
    uncert = np.array([noiseWidth])

    fitter = TriangulationSolver(xPos,yPos,zPos,distances,uncert,dampingFactor=0.11,recordHistory = True,useDampedGaussNeutonLineSearch = False)
    
    #fitter.fixParameter(fitter.z_parName())
    
    # perform eigen value decomposion to study design
    p = fitter.getP() 
    ATPA,ATP = fitter.getATPAandATP(np.array([1,2,0.0]),p)
    w,v = la.eig(ATPA)
    print(w)
    print(v)    
    
    # run the fit and display the results
    startVar = np.array([1.5,4.5,1.0]);
    out, err = fitter.solve(startVar)
    print('Number of iterations = ',fitter.noInterations)
    print('Solution = ',out)
    print('Uncertainties = ',err)
    print('Final ChiSq = ', fitter.finalChi_)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xPos, yPos, zPos, c='k', marker='o')
    ax.scatter([startVar[0]],[startVar[1]],[startVar[2]],c='r', marker='o')
    ax.scatter([out[0]],[out[1]],[out[2]],c='g', marker='o')
    xStep, yStep, zStep = list(),list(),list()
    for i in range(1,len(fitter.history)-1):
        xStep.append(fitter.history[i][0])
        yStep.append(fitter.history[i][1])
        zStep.append(fitter.history[i][2])
    ax.scatter(xStep,yStep,zStep,c='b', marker='o')
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

if __name__ == '__main__':
    run_example()