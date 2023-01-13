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






##################### Histogram  #####################

import numpy as np
def binning(x, xrange, Nbins, remove_empty=True):
    '''Function for binning data and removing empty bins
    
       x: data to be binned
       xrange: range of data in tuble (min, max)
       Nbins: number of bins
       remove_empty_bins: if True, empty bins are removed from
       
       returns: counts, bin_centers, binwidth'''
    binwidth = (xrange[1]-xrange[0])/Nbins
    counts , bin_edges = np.histogram(x, range=xrange, bins=Nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    if remove_empty:
        bin_centers = bin_centers[counts>0]
        counts = counts[counts>0]

    return counts , bin_centers, binwidth





#Plotting histogram:

def easy_hist(x, xrange, Nbins, Figsize=(10, 7)):
    '''Function for plotting a histogram
    
       x: data to be binned
       xrange: range of data in tuble (min, max)
       Nbins: number of bins
       '''
    
    counts, bin_centers, binwidth = binning(x, xrange, Nbins)
    fig, ax = plt.subplots(figsize=Figsize)
    ax.hist(x, bins=Nbins ,range=xrange, histtype='stepfilled', color='lightgreen', edgecolor='grey', linewidth=1.2)
    ax.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='o',mec='k',mfc='g', capsize=1, ecolor='g')
    ax.set(xlabel='x', ylabel=f'Counts / {binwidth:.2f}')
    plt.show()
    
    return fig, ax , counts
    




#####################  Monte Carlo  #####################



# inverse transform method:

import sympy as sp
def inverse_transform(f,N, xmin, xmax=None):
    '''Function for generation random numbers according to the inverse transformation method using sympy.
    
       Input: f: function given as a string 
              N: number of point 
              xmin: lower limit of integration
              xmax: (optional) = infinity as default'''

    if xmax == None:
        xmax = sp.oo
    
    x, y = sp.symbols('x y')
    F_norm = sp.integrate(f, (x, xmin, xmax), conds='none')

    if F_norm == 0:
        print('ERROR: Integral is zero. Choose a different integration range.')
        return
    if F_norm == sp.oo:
        print('ERROR: Integral is diverging. Choose a different integration range.')
        return

    F = 1/F_norm * sp.integrate(f, (x, xmin, x), conds='none')
    F_inv = sp.solve(F-y, x)[0]
    F_inv = sp.lambdify(y, F_inv, 'numpy')
    r = np.random
    r.seed(42)
    y = r.uniform(0,1,N)
    x_values = F_inv(y)

    return x_values ,F_norm, F_inv




#Accept/reject method: 

def Accept_reject(f, xrange, yrange, N_accepted):
    '''Function for generating random numbers according to a given function
       using the accept/reject method.
       
       Input: f, xmin, xmax, ymin, ymax, N_accepted
       
       returns: 
       array of accepted values
       number of tries
       [efficiency, efficiency error]
       [integral, integral error]  
       [normalization, normalization error]'''
       
    r = np.random
    r.seed(42)
    
    N_try = 0
    x_accepted = []
    
    while len(x_accepted) < N_accepted:
        x = r.uniform(*xrange)
        y = r.uniform(*yrange)
        if y < f(x):
            x_accepted.append(x)
        N_try += 1


    eff = N_accepted / N_try 
    eff_err = np.sqrt(eff * (1-eff) / N_try)        # binomial error

    integral =  eff * (xrange[1]-xrange[0]) * (yrange[1]-yrange[0])
    integral_err = eff_err * (xrange[1]-xrange[0]) * (yrange[1]-yrange[0])

    normalization = 1 / integral
    normalization_err = integral_err / integral**2

    return x_accepted, N_try, [eff, eff_err], [integral, integral_err], [normalization, normalization_err]


