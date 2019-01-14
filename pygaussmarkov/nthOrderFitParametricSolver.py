# -*- coding: utf-8 -*-
"""
@author: john
"""

from pygaussmarkov.ParametricSolver import ParametricSolver
import numpy as np
import matplotlib.pyplot as plt
import pygaussmarkov.GaussMarkovUtilities as GMUtils

class NthOrderFit(ParametricSolver):
    """ Nth Order Fitter """
    
    def __init__(self,order, xData, yData, uncertainties, deltaChiSqToStop = 0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        order = the order of the fit, order 2 is for a*x**2 +b*x +c
        xData = np array with the x data
        yData = np array with the y data
        uncertainties = np array of the measurement uncertainties. If length=1 then the uncertainty will be assumed to be constant
        initialValues = the starting point for the fit
        deltaChiSqToStop = if the change in chiSq is smaller than this the fitter will stop. Default = 0.01
        dampingFactor = constant to chagne convergence speed. Default 1
        useDampedGaussNeutonLineSearch  weather to use the damped gauss neuton method. default = False
        """
        self.order = order        
        super().__init__(deltaChiSqToStop = deltaChiSqToStop,dampingFactor = dampingFactor,useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch,recordHistory=recordHistory)
        self.xData = xData
        self.yData = yData
        self.uncert = uncertainties
    
    def getListOfParameterNames(self):
        return ['P'+str(i) for i in range(self.order+1)]
    
    def getAandAT(self,variables):
        m = np.matrix(np.zeros([len(self.xData),self.getNumberOfFreeParameters()]))
        skipped = 0
        parList = self.getListOfParameterNames()
        for i,x in enumerate(self.xData):
            for j in range(self.order+1):
                if not self.fixed[parList[j]]:
                    m[i,j-skipped] = x**(self.order - j)
                else:
                    skipped+=1
        mt = m.transpose()
        return np.matrix(m), np.matrix(mt)
        
    def getFx(self,variables):
        out = np.array(self.xData)
        for i,x in enumerate(self.xData):
            out[i]=0
            for j in range(self.order+1):
                out[i]=out[i]+variables[j]*x**(self.order - j)
        return out
        
    def getP(self):
        return GMUtils.generatePMatrix(self.uncert,len(self.xData))
        
    def getMeasurements(self):
        return self.yData
                
                
def run_example():
    xData = np.array(range(300))
    mTrue = 2.0
    cTrue = 0.0
    nTrue = 4.0
    noiseWidth = 10
    noise = np.random.normal(0, noiseWidth, len(xData))
    yData = nTrue*xData**2 + mTrue * xData + cTrue
    yData = yData+noise
    uncert = np.array([noiseWidth])

    fitter = NthOrderFit(2,xData,yData,uncert,useDampedGaussNeutonLineSearch=True)
    
    # run the fit and display the results
    out, err = fitter.solve(np.array([1.0,1.0,1.0]))
    print('Number of iterations = ',fitter.noInterations)
    print('Solution = ',out)
    print('Uncertainties = ',err)
    print('Final ChiSq = ', fitter.finalChi_)
    
    # plot data
    plt.plot(xData,yData,'*r')
    plt.plot(xData,fitter.getFx(out),'-k')

if __name__ == '__main__':
    run_example()