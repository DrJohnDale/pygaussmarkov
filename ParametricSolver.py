# -*- coding: utf-8 -*-
"""
module contains the ParametricSolver object
@author: john
"""

import abc
import numpy.linalg as la
import numpy as np

class ParametricSolver:
    """
    Class to solve problems of the type f(x) = L. Must be implemented by a user class and the requred abstract functions implemented
    The code is broken down into many small functions which can allow a child class to overload any of the methods to change the 
    performance of different methods.
    """
    
    def __init__(self,deltaChiSqToStop=0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        deltaChiSqToStop = if the change in chiSq is smaller than this the fitter will stop. Default = 0.01
        dampingFactor = constant to chagne convergence speed. Default 1
        useDampedGaussNeutonLineSearch  weather to use the damped gauss neuton method. default = False
        """
        self.deltaChiSqToStop = deltaChiSqToStop  
        self.dampingFactor = dampingFactor
        self.useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch
        self.printChiSqEachIteration = False
        self.printChiSqDiffEachIteration = False
        self.printParametersEachIteration = False
        self.recordHistory = recordHistory
        
        self.fixed = dict()
        for n in self.getListOfParameterNames():
            self.fixed[n]=False
        self.reset()
    
    @abc.abstractmethod
    def getListOfParameterNames(self):
        """
        Returns a list of strings.
        There should be one entry per parameter
        The order should be the same as the order in the the A matrix
        """
        return
    
    @abc.abstractmethod
    def getAandAT(self,variables):
        """
        User implemented method. Computers the determinant matrix A and its transpose AT
        variables = the current variables
        return two numpy matricies A and AT
        """
        return
        
    @abc.abstractmethod
    def getFx(self,variables):
        """
        User implemented method. Gets the values for the given variables.
        variables = the current variables
        return a numpy array
        """
        return
        
    @abc.abstractmethod
    def getP(self):
        """
        User implemented method. Gets the weight matrix (1/(sig**2))
        return a single numpy matrix
        """
        return
        
    @abc.abstractmethod
    def getMeasurements(self):
        """
        User implemented method. Gets the measurements 
        returns a numpy array
        """
        return
    
    def fixParameter(self,name):
        self.fixed[name] = True
        
    def freeParameter(self,name):
        self.fixed[name] = False

    def getNumberOfFreeParameters(self):
        out = 0
        for k,v in self.fixed.items():
            if not v:
                out+=1
        return out

    def getFixedFreeArray(self):
        out = list()
        for n in self.getListOfParameterNames():
            out.append(self.fixed[n])
        return out
    
    def getNumberOfParameters(self):
        return len(self.fixed)
    
    def continueSolver(self,chiSq):
        """
        Checks weather the fitter should continue for another iteration
        """
        if self.firstChi:
            self.firstChi = False
            self.oldChi = chiSq
            self.lastChiDifference_ = self.oldChi-chiSq
            return True
        else:
            self.lastChiDifference_ = self.oldChi-chiSq
            if self.printChiSqDiffEachIteration:
                print(self.noInterations,' ChiSq Difference = ',self.lastChiDifference_)
            if (self.lastChiDifference_)<self.deltaChiSqToStop:
                if self.oldChi<chiSq:
                    self.newVariableOK = False
                else:
                    self.oldChi = chiSq
                return False
            else:
                self.oldChi = chiSq
                return True
        
    def newVariablesOK(self):
        """
        Returns if the new variables calulated are ok.
        return boolean
        """
        return self.newVariableOK
    
    def reset(self):
        """
        Resets the internal variables
        """
        self.newVariableOK = True
        self.firstChi = True
        self.oldChi = 0
        self.noInterations = 0
        self.history = list()
    
    def getChiSq(self,w,p):
        """
        Computes w.p.w
        w = the residual vector, the difference between f(x) and the measurements
        returns value of w.p.w
        """
        wp = w.dot(p)
        wpw = wp.dot(w)
#        print(wpw)
        return wpw/len(w)
        
    def getW(self,measurements, fx):
        """
        Computes f(x) - measurements
        measurements = the measurements
        fx = the calcualted values
        returns fx-measurements
        """
        return fx - measurements
        
    def computeATP(self,AT,P):
        """
        Computes AT*P
        AT the transpose of the determinante matrix
        P the weight matrix
        return AT*P
        """
        return AT * P
        
    def computerATPA(self, ATP, A):
        """
        Computes ATP*P the design matrix
        ATP the transpose of the determinante matrix multiplied by the weight matrix
        A the determinant matrix
        return ATP*A the design matrix
        """
        return ATP * A
        
    def computeXhat(self,ATPAinv,ATP,w):
        """
        Computes ATPAinv * ATP
        ATPAinv = the inverse of the design matrix
        ATP = the transpose of the determinant matrix multiplied by the weight matrix
        w = the residual vecotr (f(x)-measurements)
        retuns the negetive value of the update vector (ATPAinv * ATP).w
        """
        ATPAinvATP = ATPAinv * ATP
        xHat = ATPAinvATP.dot(w)
        return xHat
        
    def getATPAandATP(self,variables,p):
        """
        Computes ATPA and ATP by calling sub functions
        variables =  the current variables
        p = the weight matrix
        returns ATPA and ATP
        """
        A, AT = self.getAandAT(variables)
        ATP = self.computeATP(AT,p)
        ATPA = self.computerATPA(ATP,A)
        return ATPA, ATP
        
    def getATPAinvATPandATPA(self,variables,p):
        """
        Computes ATPAinv, ATP and ATPA by calling sub functions
        variables =  the current variables
        p = the weight matrix
        returns ATPAinv, ATP and ATPA
        """
        ATPA, ATP = self.getATPAandATP(variables,p)
        return la.inv(ATPA), ATP, ATPA
    
    def getXHatandATPA(self,variables,p,w):
        """
        Computes xHat by calling subfunctions.
        variables =  the current variables
        p = the weight matrix
        w = the residual vecotr (f(x)-measurements)
        returns -xHat(subtract from current x to get new x), ATPAInv
        """
        ATPAInv, ATP, ATPA = self.getATPAinvATPandATPA(variables,p)
        xHat = self.computeXhat(ATPAInv,ATP,w)
        return xHat, ATPA
    
    def getFreeVariables(self,variables):
        """
        Returns a list of np array with only the free variables
        variables = np array of all the variables
        
        return = np array of the free variables
        """
        out = np.zeros(self.getNumberOfFreeParameters())
        skipCount = 0
        for i,(v,f) in enumerate(zip(variables,self.getFixedFreeArray())):
            if not f:
                out[i-skipCount] = v
            else:
                skipCount+=1
        return out
            
    
    def runDampedGaussNewtonLineSearch(self,ATPA,variables,xHat,chi0,p,l,fx):
        """
        If useDampedGaussNeutonLineSearch is True will run a line search, else return 1
        returns alpha
        """
        if self.useDampedGaussNeutonLineSearch:
            alpha = 1
            freeVars = self.getFreeVariables(variables)
            v = 0.5*(freeVars.dot(ATPA)).dot(freeVars)
            while not (chi0-(self.getChiSq(self.getW(l,self.getFx(self.getNewVariables(variables,alpha,xHat))),p)))>=alpha*v :
                alpha = alpha*0.5
            return alpha
        else:
            return 1
        
    
    def getNewVariables(self,variables,alpha,xHat):
        """
        Combines the current variables with the update vector xHat and the scale parameter alpha
        The method takes account of the fixed and free variables
        variables = the current variables. This should inclule all the variables either free or fixed
        alpha the scale factor for xHat. this is a scalar value
        xHat the update vector. This should only include the updates for the free parameters
        returns the new variables vector
        """
        out = variables.copy()
        skipCount = 0
        fixedVar = self.getFixedFreeArray()
        for i in range(len(variables)):
            if fixedVar[i]:
                val = variables[i]
                skipCount+=1
            else:
                val = variables[i] - alpha*self.dampingFactor*xHat[0,i-skipCount]
            out[i] = val
        return out
    
    def getUncertaintyList(self,ATPAInv):
        """
        takes the inverse of the design matrix and returns a list of the the uncertainities for the parameters
        ATPAInv is the inverse of the design matix
        returns a list of length number parmetrs, for the uncertainities for each parmeter. If the parameter is fixed
        then the value will be 0.0
        """
        un = np.sqrt(np.diagonal(ATPAInv))
        out = np.zeros(self.getNumberOfParameters())
        skipCount = 0
        fixedVar = self.getFixedFreeArray()
        for i in range(self.getNumberOfParameters()):
            if fixedVar[i]:
                val = 0.0
                skipCount+=1
            else:
                val = un[i-skipCount]
            out[i] = val
        return out
    
    def solve(self,variables):
        """
        Method solves the model
        """
        self.reset()
        variables = variables.copy()
        fx = self.getFx(variables)
        l = self.getMeasurements()
        w = self.getW(l,fx)
        p = self.getP()
        chiSq = self.getChiSq(w,p)
        first = True
        previousVariables = variables.copy()
        if self.recordHistory:
            self.history.append(variables.copy())
        
        while(self.continueSolver(chiSq) or first):
            first = False
            self.noInterations += 1
            
            xHat, ATPA = self.getXHatandATPA(variables,p,w)
            alpha = self.runDampedGaussNewtonLineSearch(ATPA,variables.copy(),xHat,chiSq*len(w),p,l,fx)
            variables = self.getNewVariables(variables,alpha,xHat)
            previousVariables = variables.copy()
            fx = self.getFx(variables)
            w = self.getW(l,fx)
            chiSq = self.getChiSq(w,p)
            if self.printChiSqEachIteration:
                print(self.noInterations,' ChiSq = ',chiSq)
            if self.printParametersEachIteration:
                print(self.noInterations,' Params = ',variables)
            if self.recordHistory:
                self.history.append(variables.copy())
            
        if not self.newVariablesOK():
            print("Using previous variables")
            variables= previousVariables
        
        #compute the final uncertainties
        ATPAInv, ATP, ATPA = self.getATPAinvATPandATPA(variables,p)
        self.finalChi_ = chiSq
        self.covarianceMatrix_ = ATPAInv.copy()
        self.finalVarialbes_ = variables.copy()
        self.uncertainties_ = self.getUncertaintyList(ATPAInv)
        
        return variables, self.uncertainties_
            