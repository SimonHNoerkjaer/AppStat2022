import numpy as np                                     
from numpy.linalg import matrix_power                  
import matplotlib.pyplot as plt                                                        
from iminuit import Minuit                            
import sys                                            
from scipy import stats
import sympy as sp
from sympy import integrate



# ################   ERROR PROPAGATION   ####################



import sympy as sp
def Errorpropagation(f, par, con = None, rho = None, cov = None):
    
    """
    Description
    -------
    Error propagation for a function f with parameters par.
    The function f can be a function of several variables. 'x**2 + y**2' is a valid function.
    The parameters par is be a list of symbols. 'x y z' for example.
    Con is a list of constraints. 'g c mu_0' for example.
    Rho can be given in a matrix form or as a list of correlations between the parameters: rho_ij = rho[i][j]
    Cov can be given in a matrix form or as a list of variances: cov_ij = cov[i][j]

    Returns 
    -------
    plot object  : sympy object for plotting the function analytically; use Display for nice text

    value object : sympy object for finding the function values;        give the function values and then constans: F(val, con)

    error object : sympy object for finding the errors ;                give the function values, then constans, then errors: F(val, con, error)
    """
    correlation_error = 0

    if con is not None:
        con = sp.symbols(con + ' æ')
    par = sp.symbols(par + ' ø')
    
    if type(f) == str:
        f = sp.parse_expr(f)
    
    error_par = []
    for i in par:
        error_par.append(sp.Symbol('sigma_' + str(i)))
    
    if con is not None:
        all_par = tuple(list(par[:-1]) + list(con[:-1]))
        all_var = tuple(list(par[:-1]) + list(con[:-1]) + error_par[:-1])
    
    else:
        all_par = tuple(list(par[:-1]))
        all_var = tuple(list(par[:-1]) + error_par[:-1])

    par = par[:-1]
    
    if con is not None:
        con = con[:-1]

    if rho is not None:
        for i in range(len(par)):
            for j in range(len(par)):
                correlation_error += rho[i][j] * error_par[i] * error_par[j] * sp.diff(f, par[i]) * sp.diff(f, par[j])
        correlation_error = correlation_error.simplify()
        correlation_error = sp.sqrt(correlation_error)
        fvalue = sp.lambdify(all_par, f)
        error_fvalue = sp.lambdify(all_var, correlation_error)
        return correlation_error, fvalue, error_fvalue

    if cov is not None:
        for i in range(len(par)):
            for j in range(len(par)):
                correlation_error += cov[i][j] * sp.diff(f, par[i]) * sp.diff(f, par[j])
        correlation_error = correlation_error.simplify()
        correlation_error = sp.sqrt(correlation_error)
        fvalue = sp.lambdify(all_par, f)
        error_fvalue = sp.lambdify(all_var, correlation_error)
        return correlation_error, fvalue, error_fvalue

    error_f = sp.sqrt(sum([sp.diff(f, i)**2 * j**2 for i, j in zip(par, error_par)]) + correlation_error)

    error_contributions = []
    for i in range(len(par)):
        error_contributions.append((sp.diff(f, par[i])**2 * error_par[i]**2).simplify())



    error_f = error_f.simplify()
    fvalue = sp.lambdify(all_par, f)
    error_fvalue = sp.lambdify(all_var, error_f)
    return error_f, fvalue, error_fvalue, error_contributions