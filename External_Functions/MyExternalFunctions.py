import numpy as np                                     
from numpy.linalg import matrix_power                  
import matplotlib.pyplot as plt                                                        
from iminuit import Minuit                            
import sys                                            
from scipy import stats
import sympy as sp
from sympy import integrate






#################################################################   PLOTTING   #################################################################

#Matplotlib settings

import matplotlib.pyplot as plt
import numpy as np
def Matplotlib_settings():
    plt.style.use('classic')
    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['figure.dpi'] = 500
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.color'] = "#cccccc"
    plt.rcParams['lines.color'] = "k"
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['errorbar.capsize'] = 2



# Text on plots  (- Troels Petersen)

import numpy as np

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    names = d.keys()
    max_names = len_of_longest_string(names)
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None



























########################################################################   ERROR PROPAGATION   ##################################################################



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
















############################################################### Histogram  ###############################################################

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
    










######################################################################  Monte Carlo  ###############################################################



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









######################################################################  Chi2  ###############################################################



from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


#chi2 penelty function

def Chi2penelty(parameter, know_parameter, sigma_parameter):
    '''Function for calculating the penelty term for a given parameter, known parameter and sigma.
    
       Input: parameter, known parameter, sigma
       
       returns: penelty term'''

    return (parameter - know_parameter)**2 / sigma_parameter**2



class Chi2Regression:  # override the class with a better one

    def __init__(self, f, x, y, sy=None, weights=None, bound=None, penelty=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]
            
        

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

        # if penelty is None:
        #     self.penelty = 0
        # else:
        #     self.penelty = np.sum(Chi2penelty(*penelty))

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2) + 
        
        return chi2



# chi2 prop string med ddof 
from scipy import stats
def Chi2prop(Minuit_object, N_data):
    '''Function for calculating the chi2 probability for a given Minuit object and number of data points.
    
       Input: Minuit object, number of data points
       
       returns: chi2 p-value , string(chi2 / dof = chi2_reduced)'''

    chi2 = Minuit_object.fval
    dof = N_data - Minuit_object.nfit
    p_value = stats.chi2.sf(chi2, dof)

    # string with chi2 / dof = chi2_reduced
    chi2_string = f'{chi2:.2f} / {dof} = {chi2/dof:.2f}'

    return p_value, chi2_string














############################################################### Fisher Discriminants  ###############################################################



# fisher discriminant function:

import numpy as np

def fisher_discriminant(sample1, sample2, w0=True):
    """
    Calculates Fisher discriminants given two samples with the w0 correction term as default.
    
    Parameters:
        sample1 (numpy array): the first sample
        sample2 (numpy array): the second sample
        w0 (bool): if True, the w0 correction term is calculated
    
    Returns:
        (float): the calculated Fisher discriminant
    """
    mean1 = np.mean(sample1, axis=0)
    mean2 = np.mean(sample2, axis=0)
    print(mean1, mean2)
    cov1 = np.cov(sample1, rowvar=False)
    cov2 = np.cov(sample2, rowvar=False)
    
    cov_combined = cov1 + cov2
    print(cov_combined)
    inv_cov_combined = np.linalg.inv(cov_combined)

    wf = inv_cov_combined @ (mean1 - mean2)

    if w0:
        w0 = - np.sum(wf)

    fisher1 = sample1 @ wf + w0 
    fisher2 = sample2 @ wf + w0

    std_1 = np.std(sample1, ddof=1)
    std_2 = np.std(sample2, ddof=1)
    separation = np.abs((mean1 - mean2)) / np.sqrt(std_1**2 + std_2**2)

    return fisher1, fisher2, separation










################################################################## ROC curve  ###############################################################