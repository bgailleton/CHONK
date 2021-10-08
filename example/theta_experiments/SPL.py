import numpy as np
import numba as nb
import math
from scipy import optimize

@nb.jit(nopython = True)
def Erosion_spl_explicit_fixed_n(Z,X,Q,theta,K, n):
    E = np.zeros_like(Z)
    for i in range(1,Z.shape[0]):
        E[i] = K[i] * np.power(((Z[i] - Z[i-1])/(X[i] - X[i-1])),(n)) * np.power(Q[i],(theta[i] * n))
    return E

@nb.jit(nopython = True)
def calculate_theta_power_fit(Z,Q,X,n_reg):
    theta = np.zeros_like(Z)
    
    regressiator_Z = Z[:n_reg]
    regressiator_X = X[:n_reg]
    regressiator_Q = Q[:n_reg]
    trange = np.arange(Z.shape[0])[::-1]
    
    for i in range(1,Z.shape[0]):
        if(i >= n_reg and i < Z.shape[0] - n_reg):
            regressiator_Z = Z[i - n_reg:i]
            regressiator_X = X[i - n_reg:i]
            regressiator_Q = Q[i - n_reg:i]
        regressiator_S = np.diff(regressiator_Z)/np.diff(regressiator_X)
        regressiator_Q2 = regressiator_Q[1:]
        ksn,theta[i] = fit_power_law(regressiator_Q2,regressiator_S)
    
    theta[0] = theta[1]
    return theta
        
@nb.jit(nopython = True)       
def fit_power_law(X,Y):
    # Y = aX^b
    # Finding required sum for least square methods
    n = X.shape[0]
    sumX,sumX2,sumY,sumXY = 0,0,0,0
    for i in range(n):
        if(X[i]>0):
            sumX = sumX + math.log(X[i])
        
            sumX2 = sumX2 + math.log(X[i])*math.log(X[i])
        if(Y[i]>0):
            sumY = sumY + math.log(Y[i])
        if(X[i]>0 and Y[i]>0):
            sumXY = sumXY + math.log(X[i])*math.log(Y[i])

    b = (n*sumXY-sumX*sumY)/(n*sumX2-sumX*sumX);
    A = (sumY - b*sumX)/n;
    try:
        a = math.exp(A);
    except:
        a = 0
    return a,b

@nb.jit(nopython = True)
def calculate_theta_mean(Z,Q,X,n_reg):
    theta = np.zeros_like(Z)
    
    regressiator_Z = Z[:n_reg]
    regressiator_X = X[:n_reg]
    regressiator_Q = Q[:n_reg]
    trange = np.arange(Z.shape[0])[::-1]
    
    for i in trange:
        if(i >= n_reg and i < Z.shape[0] - n_reg):
            regressiator_Z = Z[i - n_reg:i]
            regressiator_X = X[i - n_reg:i]
            regressiator_Q = Q[i - n_reg:i]
#         m,b = simple_linreg(regressiator_X,regressiator_Z)
        regressiator_S = np.diff(regressiator_Z)/np.diff(regressiator_X)
#         regressiator_Q2 = (regressiator_Q[1:] + regressiator_Q[:-1])/2
        regressiator_Q2 = regressiator_Q[:-1] 
#         print(regressiator_S,"||",regressiator_Q2)
        theta[i] = np.mean(np.diff(np.log(regressiator_S))/np.diff(np.log(regressiator_Q2)))
#         print(theta[i])
# @nb.jit(nopython = True)
def calculate_theta_linreg(Z,Q,X, n_reg):
    
    theta = np.zeros_like(Z)
    
    regressiator_Z = Z[:n_reg]
    regressiator_X = X[:n_reg]
    regressiator_Q = Q[:n_reg]
    trange = np.arange(Z.shape[0])[::-1]
    
    for i in trange:
        if(i >= n_reg and i < Z.shape[0] - n_reg):
            regressiator_Z = Z[i - n_reg:i]
            regressiator_X = X[i - n_reg:i]
            regressiator_Q = Q[i - n_reg:i]
#         m,b = simple_linreg(regressiator_X,regressiator_Z)
        regressiator_S = np.diff(regressiator_Z)/np.diff(regressiator_X)
#         regressiator_Q2 = (regressiator_Q[1:] + regressiator_Q[:-1])/2
        regressiator_Q2 = regressiator_Q[:-1] 
#         print(regressiator_S,"||",regressiator_Q2)
        theta[i],b = simple_linreg(np.log(regressiator_Q2),np.log(regressiator_S))
        
    return theta
        
    
    

# @nb.jit(nopython = True)
def simple_linreg(X,Y):
    """
        Calculate a simple LS linear regression and return m and b (y = m x + b)
    """
    n = X.shape[0]
    
    sumX  = 0.
    sumX2 = 0.
    sumY  = 0.
    sumXY = 0.
    
    for i in range(n):

        sumX = sumX + X[i];
        sumX2 = sumX2 + X[i]*X[i];
        sumY = sumY + Y[i];
        sumXY = sumXY + X[i]*Y[i];

    m = (n*sumXY-sumX*sumY)/(n*sumX2-sumX*sumX);
    b = (sumY - m*sumX)/n;
    
    return m,b

def powerlaw(x,a,b):
    return a * np.power(x,b)

def optimise_by_breaks(X,Y,breaks, p0 = [-np.inf,np.inf], hb = 10):
    a = np.zeros_like(X) 
    b = np.zeros_like(X)
    start = 0

    for end in breaks:
        tX = X[start:end]
        tY = Y[start:end]
        try:
            params, params_covariance = optimize.curve_fit(powerlaw, tX, tY, maxfev = 5000)#, maxfev = 5000, p0 = p0)
            ta,tb = params
            a[start:end] = ta
            b[start:end] = tb
        except RuntimeError:
            pass
        start = end

    tX = X[start:]
    tY = Y[start:]
#         try:
#         print(tX)
#         print(tY)
    params, params_covariance = optimize.curve_fit(powerlaw, tX, tY, maxfev = 5000)#, maxfev = 5000, p0 = p0)
    ta,tb = params
    a[start:] = ta
    b[start:] = tb
        
    
    return a,b
        