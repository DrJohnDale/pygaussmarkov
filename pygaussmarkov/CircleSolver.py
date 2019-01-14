# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:11:17 2017

@author: john
"""

from pygaussmarkov.CombinedSolver import CombinedSolver
import numpy as np
import matplotlib.pyplot as plt

class CircleSolver(CombinedSolver):
    """ fits a circle with a combined solver """
    def __init__(self, xPos, yPos, xPosUncert, yPosUncert ,deltaChiSqToStop = 0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        xPos = the x positions, these are measurements
        yPos = the y positions, these are measurements
        xPosUncert = the errors on the x meaurements. Should be lenght 1 or equal to len(xpos)
        yPosUncert = the errors on the y measurements.  Should be lenght 1 or equal to len(ypos)
        deltaChiSqToStop = if the change in chiSq is smaller than this the fitter will stop. Default = 0.01
        dampingFactor = constant to chagne convergence speed. Default 1
        useDampedGaussNeutonLineSearch  weather to use the damped gauss neuton method. default = False
        """
        super().__init__(deltaChiSqToStop = deltaChiSqToStop,dampingFactor = dampingFactor,useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch,recordHistory=recordHistory)
        self.xPos = xPos
        self.yPos = yPos
        self.xPosUncert = xPosUncert
        self.yPosUncert = yPosUncert
     
    def xc_paramName(self):
        return 'xc'
    
    def yc_paramName(self):
        return 'yc'
    
    def r_paramName(self):
        return 'r'
    
    def getListOfParameterNames(self):
        return [self.xc_paramName(),self.yc_paramName(),self.r_paramName()]
    
    def getAandAT(self,variables):
        m = np.matrix(np.zeros([len(self.xPos),self.getNumberOfFreeParameters()]))
        
        xCen = variables[0]
        yCen = variables[1]
        rad = variables[2]
        
        for i in range(len(self.xPos)):
            skipVal = 0
            
            if not self.fixed[self.xc_paramName()]:
                m[i,0] = -2*self.xPos[i] + 2*xCen
            else:
                skipVal+=1
            
            if not self.fixed[self.yc_paramName()]:
                m[i,1-skipVal] = -2*self.yPos[i] + 2*yCen
            else:
                skipVal+=1
            
            if not self.fixed[self.r_paramName()]:
                m[i,2-skipVal] = -2*rad
        
        mt = m.transpose()
        return m, mt
        
    
    def getBandBT(self,variables):
        m = np.matrix(np.zeros([len(self.xPos),len(self.xPos)*2]))
        
        xCen = variables[0]
        yCen = variables[1]
#        rad = variables[2]
        
        for i in range(len(self.xPos)):
            m[i,0 + i*2] = 2*self.xPos[i] - 2*xCen
            m[i,1 + i*2] = 2*self.yPos[i] - 2*yCen
            
        mt = m.transpose()
        return m, mt
        
    def getFxl(self,variables):
        """
        (x-xc)**2 + (y-yc)**2 - r**2
        """
        out = np.zeros(len(self.xPos))
        xCen = variables[0]
        yCen = variables[1]
        rad = variables[2]
        for i,(x,y) in enumerate(zip(self.xPos,self.yPos)):
            out[i] = (x-xCen)**2 + (y-yCen)**2 - rad**2
        
        return out;
        
    def getPinv(self):
        if len(self.xPosUncert)>1 and len(self.xPosUncert)!=len(self.xPos):
            raise Exception('Error generating weight matrix error. xPosUncert array length must be 1 or equal to length of xPos. len errors = '+str(len(self.xPosUncert))+' len xPos = '+str(self.xPos))
        
        if len(self.yPosUncert)>1 and len(self.yPosUncert)!=len(self.yPos):
            raise Exception('Error generating weight matrix error.yPosUncert array length must be 1 or equal to length of yPos. len errors = '+str(len(self.yPosUncert))+' len yPos = '+str(self.yPos))
        
        m = np.eye(len(self.xPos)*2)
        
        if len(self.xPosUncert)==1:
            v = (self.xPosUncert[0]**2)
            for i in range(len(self.xPos)):
                m[i*2,i*2] = v
        else:
            for i in range(len(self.xPos)):
                m[i*2,i*2] = (self.xPosUncert[i]**2)
                
        if len(self.yPosUncert)==1:
            v = (self.yPosUncert[0]**2)
            for i in range(len(self.yPos)):
                m[i*2+1,i*2+1] = v
        else:
            for i in range(len(self.yPos)):
                m[i*2+1,i*2+1] = (self.yPosUncert[i]**2)       
                
        return m
    
    def determinStatingVariables(self):
        """
        determines initial parameters from the current data
        """
        xMax = np.max(self.xPos)
        xMin = np.min(self.xPos)
        yMin = np.min(self.yPos)
        
        r = (xMax-xMin)/2.0
        xc = xMin+r
        yc = yMin+r
        return np.array([xc,yc,r])
        
def run_example():
    xc = 5
    yc = 5
    r = 5
    noiseX = 0.5
    noiseY = 0.5
    xPos = np.array([x for x in np.arange(xc-r,xc+r+0.05,0.05)])
    yPos = np.zeros(len(xPos))
    for i,x in enumerate(xPos):
        v = np.sqrt(r**2 - (x-xc)**2)
        if i % 2 ==0:
            yPos[i] = v + yc
        else:
            yPos[i] = -v+yc
    
    xPos = xPos + np.random.normal(0, noiseX, len(xPos))
    yPos = yPos + np.random.normal(0, noiseY, len(yPos))
        
    fitter = CircleSolver(xPos,yPos,np.array([noiseX]),np.array([noiseY]),recordHistory=True)
    
    startParam = fitter.determinStatingVariables()
    out,err = fitter.solve(startParam)

    print('Number of iterations = ',fitter.noInterations)
    print('Solution = ',out)
    print('Uncertainties = ',err)
    print('Final ChiSq = ', fitter.finalChi_)

#    print(xPos)
#    print(yPos)
    print(fitter.history)
    cInitial = plt.Circle((startParam[0],startParam[1]),startParam[2],color='r',clip_on=False,fill=False)
    cFit = plt.Circle((out[0],out[1]),out[2],color='g',clip_on=False,fill=False)
    otherCirc = [plt.Circle((fitter.history[i][0],fitter.history[i][1]),fitter.history[i][2],color='b',clip_on=False,fill=False) for i in range(1,len(fitter.history)-1)]
    plt.plot(xPos,yPos,'*k')
    plt.gca().add_artist(cInitial)
    plt.gca().add_artist(cFit)
    for c in otherCirc:
        plt.gca().add_artist(c)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    return        

if __name__ == '__main__':
    run_example()