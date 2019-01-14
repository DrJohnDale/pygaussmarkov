# -*- coding: utf-8 -*-
from pygaussmarkov.ParametricSolver import ParametricSolver
import numpy as np
import numpy.linalg as la

class ParametricFreeNetworkConstraintSolver(ParametricSolver):
    USE_EVD = 2
    USE_SVD = 1
    def __init__(self,numberOfConstaints,decomposition=1,deltaChiSqToStop=0.01,dampingFactor = 1,useDampedGaussNeutonLineSearch = False, recordHistory = False):
        """
        implements a parametric solver inner constraint with evd or svd
        """
        super().__init__(deltaChiSqToStop = deltaChiSqToStop,dampingFactor = dampingFactor,useDampedGaussNeutonLineSearch = useDampedGaussNeutonLineSearch,recordHistory=recordHistory)
        self.decomp = decomposition
        self.numberOfConstaints = numberOfConstaints
        
    def getEVD(self,ATPA):
        w,v = la.eig(ATPA)
        idx = w.argsort()[::-1]   
        w = w[idx]
        v = v[:,idx]
        return v
        
    def getSVD(self,ATPA):
        u,s,v = la.svd(ATPA)
        return v
        
    def getConstraintMatrix(self,ATPA):
        if self.decomp == self.USE_EVD:
            decomp = self.getEVD(ATPA)
        elif self.decomp == self.USE_SVD:
            decomp = self.getSVD(ATPA)
            
        cutSize = decomp.shape[0] - self.numberOfConstaints
            
        A2 = -decomp[cutSize:,:]
        
        return A2, A2.transpose()
    
    def getATPAinvATPandATPA(self,variables,p):
        ATPA, ATP = self.getATPAandATP(variables,p)
#        print('initial ATPA',ATPA)
        A2, A2T = self.getConstraintMatrix(ATPA)
#        print('A2',A2)
        ATPA = np.concatenate((ATPA,A2))
#        print('mid ATPA',ATPA)
        ATPA = np.concatenate((ATPA,np.concatenate((A2T,np.zeros([self.numberOfConstaints,self.numberOfConstaints])))),1)
#        print('final ATPA',ATPA)
        return la.inv(ATPA), ATP, ATPA
    
    def getXHatandATPA(self,variables,p,w):
        ATPAInv, ATP, ATPA = self.getATPAinvATPandATPA(variables,p)
        ATPW = ATP.dot(w)
        ATPW = ATPW.transpose()
        z = np.zeros([self.numberOfConstaints,1])
#        print('ATPW Shape',ATPW.shape)
#        print('z Shape',z.shape)
        ATPW = np.concatenate((ATPW,z))
#        print('ATPW Shape',ATPW.shape)
        xHat = ATPAInv * ATPW
#        print("xHat from inner constraint",xHat)
        return xHat.transpose(), ATPA
    
    def computeXhat(self,ATPAinv,ATP,w):
        print("Method not used in inner constraint solver")
        return 0