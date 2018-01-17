The PyGaussMarkov package is package to apply non-linear least squares using the methods developed by Gauss and Markov.

The user only needs to add code to produce the function (F(X)) and its first derivates (dF(X)/dX and/or dF(X,L)/dL) to allow the solvers to fit their data.

Introduction:

These models are used extensively in metroloy (e.g. the alignment of particle accelerators) and photogrammetry. In photogrammetry the term bundle adjustment is often used.
The models are applicable for problems where analyitical equations for the problem at hand can be written, along with the first dirivate with respect to the variables which need to be determined.

The implementations follow the methods described in the lecture notes from D. E. Wells and E. J. Krakiwsky which can be found here http://www2.unb.ca/gge/Pubs/LN18.pdf

Two types of generic solver are implemented: the parametric and the combined.  
The parametric solver can be used in the case when you have a vector of functions F(X), which have variables vector X, equal a vector of measurements L, ie F(X)=L.
The combined solver can be used in the case when you have a vectro of functions F(X,L)=0, i.e. the measurements and the variables cannot be separated onto either side of the equation.

To implement a solver the user must wirite there own class which extends either of the generic solvers and implements the required abstract methods.

Features:
Both the combined and the parametric solvers can have parameters fixed or freed using the fixParameter(paramName) and freeParameter(paramName) functions of the object. 
The two sovers also have a damped gauss newton line search implemented which can be activated or deactivated by setting the objects useDampedGaussNeutonLineSearch to true or false.
The dampingFactor variable can be used to scale the step size between iterations. This is defaulted to 1.
The variables printChiSqEachIteration,printChiSqDiffEachIteration,printParametersEachIteration and printParametersEachIteration can be used to show logging to help with debugging. They are all defaulted to false
The variable recordHistory can be used to recored the results of each iteration (when set to true). The history variable holds the history.

Functions to implement in user object:
To implement a parametric solver the following methods need to be implemented.

getListOfParameterNames(): returns a list of all the parameter names
getAandAT(variables): calculates A and AT matrix. A = dF(x)/d(x) AT is the transpose of A. Note this must take account of free and fixed variables
getFx(variables): calculates an array of values which represent the current calculated values based on the given variables.
getP(): returns the weight matrix. The weight matrix elements are the inverse of the measurement uncertainties squared. If the uncertainties/weights are set correctly the solver should converge to a chiSquared of 1
getMeasurements(): returns a vector of the measurements to compaire the estimates from getFx().
   
To implement a combined solver all the methods for the parametric sover are required along with.
    
getBandBT(variables): calculates B and BT matrix. B = dF(x,l)/d(l) AT is the transpose of A
getPinv(): calculates the inverse of P

An example of a solver using the parametric method is TriangulationSolver.py
An example of a solver using the combined method is CircleSolver.py

Pitfalls:
Make sure that when you run a solver you have enough measurements so that it can solve. If you have less measurements then varaibles the software will fail and throw an exception.

Analysing mode with eigen value decomposition:
If a model cannot calculate the required inverse then the following code snippet can be used to calculate the eigen values and vectors. An eigen values with 0 value indicate where the model is poorly constrained.
    
    p = fitter.getP() 
    ATPA,ATP = fitter.getATPAandATP(variables,p)
    w,v = la.eig(ATPA)
    print(w)
    print(v)   
    
An example of this is in the triangulationSolver.py in the main() funciton at the bottom of the file.

Example:
Here we will go thorough the TriangulationSolver.py as an example of how to implement a solver.
In triangulation we measure the distance between our position and multipule other known positions. The distance measured is represented by the following equation:
d_i = sqrt((x_i-x)^2 + (y_i-y)^2 + (z_i-z)^2)
where the (x_i,y_i,z_i) represent the ith measured position which has a distance from our position (x,y,z) of d_i.
In the solver notation a vector of all measurements is L, the vector (x,y,z) = X and the vector of all the d_i functions is F(X). Note the length of F(X) and L must be equal

The solver will adjsut (x,y,z) untill the difference between the measured distances, and the calulcated distances is minimised.

The derivate equations are:
d d_i/dx = (x_i - x)/d_i
d d_i/dy = (y_i - y)/d_i
d d_i/dz = (z_i - z)/d_i

To build the TriangulationSolver object the constructor has the following required parameters:
xPos array of x positions for the targets
yPos array of y positions for the targets
zPos array of z positions for the targets
measurements the measured distances
uncertainties the uncertainities on the measured distances.

The implemented functions are described below.
getListOfParameterNames is self explanatory from the code.
getFx calculates the estimated distance based on the current variables
getP used the GaussMarkovUtilities and the input uncertainties to compute the weight matrix
getMeasurements returns the measurements passed to the constructor
getAandAT compuated the matrix of derivates, but only for the free parameters, if any of the parameters are fixed the are not included and the matrix is reduced in size.

In the TriangulationSolve.py there is a Main() function which can be run which demonstraits the usage. The process of initalising a solver and running it is as follows:
Initialise the fitter with its parameters, in TriangulationSolve.py main() function the dampingFactor=0.11 is set only to slow it down to demonstrate its usage.
The variabes which should not be adjsted should then be fixed using fixParameter(paramName). E.g. if you only had a 2D problem you could fix z
The required debug variables should then be set
The solver is run passing the initial starting variables. The code snippet is: out, err = fitter.solve(startVar). Where out holds the determined x,y,z values and error holds the unertainties on the determined x,y,z values.
The example has the history turned on and so it then plots the path the solver took to find the solution.

Implemented solvers:
TriangulationSolver.py (parametric)
CircleSolver.py (combined)
nthOrderFitParametricSolver.py (parametric)

Planned features:
Analyitical sover: To display the full equations to allow analysis of weaknesses in the model
Inner Constraint: To allow self constraining models
Kalman Filter