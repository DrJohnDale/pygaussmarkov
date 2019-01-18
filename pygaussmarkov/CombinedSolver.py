# -*- coding: utf-8 -*-
"""
@author: john
"""

import abc
import numpy.linalg as la
import numpy as np

class CombinedSolver:
    """
    Class to solve problems of the type f(x,l) = 0. Must be implemented by a user class and the requred abstract functions implemented
    The code is broken down into many small functions which can allow a child class to overload any of the methods to change the 
    performance of different methods.
    
    The variables printChiSqEachIteration,printChiSqDiffEachIteration,printParametersEachIteration and printParametersEachIteration can be used to 
    show logging to help with debugging. They are all defaulted to false
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
    def getBandBT(self,variables):
        """
        User implemented method. Computers the determinant matrix A and its transpose AT
        variables = the current variables
        return two numpy matricies A and AT
        """
        return
        
    @abc.abstractmethod
    def getFxl(self,variables):
        """
        User implemented method. Gets the values for the given variables.
        variables = the current variables
        return a numpy array
        """
        return
        
    @abc.abstractmethod
    def getPinv(self):
        """
        User implemented method. Gets the error matrix ((sig**2)) this is also the inverse of the weight matirx p
        return a single numpy matrix
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
        return [self.fixed[n] for n in self.getListOfParameterNames()]
    
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
    
    def getChiSq(self,w,BTpinvBinv):
        """
        Computes w.p.w
        w = the residual vector, the difference between f(x) and the measurements
        returns value of w.p.w
        """
        wp = w.dot(BTpinvBinv)
        wpw = wp.dot(w)
#        print(wpw[0,0])
        return wpw[0,0]/len(w)
    
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
    
    
    def getBpinvBTinv(self,variables):
        pinv = self.getPinv()
        B, BT = self.getBandBT(variables)
        Bpinv = B*pinv
        BpinvBT = Bpinv*BT
        return la.inv(BpinvBT)
    
    def computeATBPinvBTinv(self,AT,BPinvBTinv):
        """
        Computes AT*BPinvBTinv
        AT the transpose of the determinante matrix
        BPinvBTinv the weight matrix
        return AT*BPinvBTinv
        """
        return AT * BPinvBTinv
        
    def computerATBPinvBTinvA(self, ATBPinvBTinv, A):
        """
        Computes ATBPinvBTinv*A the design matrix
        ATBPinvBTinv the transpose of the determinante matrix multiplied by the weight matrix
        A the determinant matrix
        return ATBPinvBTinv*A the design matrix
        """
        return ATBPinvBTinv * A
    
    def getATBPinvBTinvAandATBPinvBTinv(self,variables,BpinvBTinv):
        """
        Computes ATBPinvBTinvA and ATBPinvBTinv by calling sub functions
        variables =  the current variables
        p = the weight matrix
        returns ATBPinvBTinvA and ATBPinvBTinv
        """
        A, AT = self.getAandAT(variables)
        ATBPinvBTinv = self.computeATBPinvBTinv(AT,BpinvBTinv)
        ATBPinvBTinvA = self.computerATBPinvBTinvA(ATBPinvBTinv,A)
        return ATBPinvBTinvA, ATBPinvBTinv
        
    def getATBPinvBTinvAinvandATBPinvBTinvandATBPinvBTinvA(self,variables,BpinvBTinv):
        """
        Computes ATBPinvBTinvAinv, ATBPinvBTinv and ATBPinvBTinvA by calling sub functions
        variables =  the current variables
        p = the weight matrix
        returns ATBPinvBTinvAinv, ATBPinvBTinv and ATBPinvBTinvA
        """
        ATBPinvBTinvA, ATBPinvBTinv = self.getATBPinvBTinvAandATBPinvBTinv(variables,BpinvBTinv)
        return la.inv(ATBPinvBTinvA), ATBPinvBTinv, ATBPinvBTinvA
    
    def getXHatandATBPinvBTinvA(self,variables,BpinvBTinv,w):
        ATBPinvBTinvAinv, ATBPinvBTinv, ATBPinvBTinvA = self.getATBPinvBTinvAinvandATBPinvBTinvandATBPinvBTinvA(variables,BpinvBTinv)
        xHat = self.computeXhat(ATBPinvBTinvAinv,ATBPinvBTinv,w)
        return xHat, ATBPinvBTinvA
    
    def computeXhat(self,ATBPinvBTinvAinv,ATBPinvBTinv,w):
        """
        Computes ATBPinvBTinvAinv * ATBPinvBTinv
        ATBPinvBTinvAinv = the inverse of the design matrix
        ATBPinvBTinv = the transpose of the determinant matrix multiplied by the weight matrix
        w = the residual vecotr (f(x,l))
        retuns the negetive value of the update vector (ATBPinvBTinvAinv * ATBPinvBTinv).w
        """
        ATBPinvBTinvAinvATBPinvBTinv = ATBPinvBTinvAinv * ATBPinvBTinv
        xHat = ATBPinvBTinvAinvATBPinvBTinv.dot(w)
        return xHat
    
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
    
    def getUncertaintyList(self,ATBPinvBTinvAInv):
        """
        takes the inverse of the design matrix and returns a list of the the uncertainities for the parameters
        ATBPinvBTinvAInv is the inverse of the design matix
        returns a list of length number parmetrs, for the uncertainities for each parmeter. If the parameter is fixed
        then the value will be 0.0
        """
        un = np.sqrt(np.diagonal(ATBPinvBTinvAInv))
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
    
    def runDampedGaussNewtonLineSearch(self,ATBPinvBTinvA,variables,xHat,chi0,BpinvBTinv,fxl):
        """
        If useDampedGaussNeutonLineSearch is True will run a line search, else return 1
        returns alpha
        """
        if self.useDampedGaussNeutonLineSearch:
            alpha = 1
            freeVars = self.getFreeVariables(variables)
            v = 0.5*(freeVars.dot(ATBPinvBTinvA)).dot(freeVars)
            while not (chi0-(self.getChiSq(self.getFxl(self.getNewVariables(variables,alpha,xHat)),BpinvBTinv)))>=alpha*v :
                alpha = alpha*0.5
            return alpha
        else:
            return 1
    
    def solve(self,variables):
        """
        Method solves the model
        """
        self.reset()
        self.lastVars = variables.copy()
        variables = variables.copy()
        fxl = self.getFxl(variables)
        w = fxl
        BpinvBTinv = self.getBpinvBTinv(variables)
        chiSq = self.getChiSq(w,BpinvBTinv)
        first = True
        previousVariables = variables.copy()
        previousChiSq = chiSq
        if self.recordHistory:
            self.history.append(variables.copy())
        while(self.continueSolver(chiSq) or first):
            first = False
            self.noInterations += 1
            
            xHat, ATBPinvBTinvA = self.getXHatandATBPinvBTinvA(variables,BpinvBTinv,w)
            
            alpha = self.runDampedGaussNewtonLineSearch(ATBPinvBTinvA,variables.copy(),xHat,chiSq*len(w),BpinvBTinv,fxl)
            
            previousVariables = variables.copy()
            previousChiSq = chiSq
            variables = self.getNewVariables(variables,alpha,xHat)
            self.lastVars = variables.copy()
            fxl = self.getFxl(variables)
            w = fxl
            
            BpinvBTinv = self.getBpinvBTinv(variables)
            chiSq = self.getChiSq(w,BpinvBTinv)
            
            if self.printChiSqEachIteration:
                print(self.noInterations,' ChiSq = ',chiSq)
            if self.printParametersEachIteration:
                print(self.noInterations,' Params = ',variables)
            if self.recordHistory:
                self.history.append(variables.copy())
        if not self.newVariablesOK():
            print("Using previous variables")
            variables = previousVariables
            chiSq = previousChiSq
        
        #compute the final uncertainties
        ATBPinvBTinvAinv, ATBPinvBTinv, ATBPinvBTinvA = self.getATBPinvBTinvAinvandATBPinvBTinvandATBPinvBTinvA(variables,BpinvBTinv)
        self.finalChi_ = chiSq
        self.covarianceMatrix_ = ATBPinvBTinvAinv.copy()
        self.finalVarialbes_ = variables.copy()
        self.uncertainties_ = self.getUncertaintyList(ATBPinvBTinvAinv)
        
        return variables, self.uncertainties_

