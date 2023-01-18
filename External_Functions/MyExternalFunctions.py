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
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
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
def easy_hist(x, Nbins, Figsize=(10, 7), xrange=None, title= None, x_label= 'x'):
    '''Function for plotting a histogram
    
       x: data to be binned
       xrange: range of data in tuble. Default: (min(x), max(x))
       Nbins: number of bins
       '''
    
    
    if xrange is None:
        xrange = (np.min(x), np.max(x))

    counts, bin_centers, binwidth = binning(x, xrange, Nbins)
    fig, ax = plt.subplots(figsize=Figsize)
    ax.hist(x, bins=Nbins ,range=xrange, histtype='stepfilled', color='lightgreen',edgecolor='grey', linewidth=1.2)
    ax.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='o',mec='k',mfc='g', capsize=2, ecolor='g')
    ax.set(xlabel=x_label, ylabel=f'Counts / {binwidth:.2f}', title=title)
    # plt.show()
    return fig, ax , counts , bin_centers, binwidth
    

























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
    F_integral = sp.integrate(f, (x, xmin, xmax), conds='none')

    if F_integral == 0:
        print('ERROR: Integral is zero. Choose a different integration range.')
        return
    if F_integral == sp.oo:
        print('ERROR: Integral is diverging. Choose a different integration range.')
        return

    Norm_Constant = 1/F_integral
    F = Norm_Constant * sp.integrate(f, (x, xmin, x), conds='none')
    F_inv = sp.solve(F-y, x)[0]
    F_inv = sp.lambdify(y, F_inv, 'numpy')
    r = np.random
    r.seed(42)
    y = r.uniform(0,1,N)
    x_values = F_inv(y)

    return x_values ,Norm_Constant, F_inv




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


































############################################################### Fisher Discriminants  ###############################################################



# fisher discriminant function:

import numpy as np

def fisher_2var(sample1, sample2, w0=True):
    """
    Calculates Fisher discriminants given two samples with two variables with the w0 correction term as default.
    
    Parameters:
        sample1 (numpy array): the first sample
        sample2 (numpy array): the second sample
        w0 (bool): if True, the w0 correction term is calculated
    
    Returns:
        (float): the calculated Fisher discriminant
    """
    mean1 = np.mean(sample1, axis=0)
    mean2 = np.mean(sample2, axis=0)
    cov1 = np.cov(sample1, rowvar=False)
    cov2 = np.cov(sample2, rowvar=False)
    
    cov_combined = cov1 + cov2
    inv_cov_combined = np.linalg.inv(cov_combined)

    wf = inv_cov_combined @ (mean1 - mean2)

    if w0:
        w0 = - np.sum(wf)

    fisher1 = sample1 @ wf + w0 
    fisher2 = sample2 @ wf + w0

    std_1 = np.std(sample1, ddof=1)
    std_2 = np.std(sample2, ddof=1)
    sample_sep = np.abs((mean1 - mean2)) / np.sqrt(std_1**2 + std_2**2)

    fish1_mean = np.mean(fisher1)
    fish2_mean = np.mean(fisher2)
    fish1_std = np.std(fisher1, ddof=1)
    fish2_std = np.std(fisher2, ddof=1)
    fish_sep = np.abs(fish1_mean - fish2_mean) / np.sqrt(fish1_std**2 + fish2_std**2)


    return fisher1, fisher2, sample_sep, fish_sep, wf































################################################################## ROC curve  ###############################################################



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def ROC_curve(sample1, sample2, fpr_cond=None, tpr_cond=None, plot=True):
    """
    Calculates the ROC curve and the AUC given two samples. 
    Optionally condtions on either the FPR or TPR can be given to calculate the corresponding threshold.
    
    Parameters:
        sample1 (numpy array): the first sample
        sample2 (numpy array): the second sample
    
    Returns:
        fig, ax (matplotlib figure): the ROC curve
        (array): the calculated FPR
        (array): the calculated TPR
        (float): the calculated AUC
        (optional) (printed string): the calculated threshold for the given condition 
    """


    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(np.concatenate((np.zeros(len(sample1)), np.ones(len(sample2)))), np.concatenate((sample1, sample2)))


    # Plot the ROC curve
    if plot==True:
        fig, ax = plt.subplots(figsize=(8,7))
        ax.plot(fpr, tpr)
        ax.plot([0,1], [0,1], linestyle='--', color='black')
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC curve')



    # Calculate the Area Under the Curve (AUC)
    Area = auc(fpr, tpr)   


    if fpr_cond != None:
        condition_index = np.where(fpr < fpr_cond)[0][-1]
        print(f'The threshold for a false positive rate of {fpr_cond} is {thresholds[condition_index]:.3f}')
        
    
    if tpr_cond != None:
        condition_index = np.where(tpr > tpr_cond)[0][0]
        print(f'The threshold for a true positive rate of {tpr_cond} is {thresholds[condition_index]:.3f}')


    return fpr, tpr, Area







































######################################################################  Chi2  ###############################################################



from iminuit.util import make_func_code
from iminuit import describe #, Minuit

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
       
       returns: penelty term (parameter index, value, sigma)'''

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

        if penelty is not None:
            self.penelty = penelty[1:]
            self.peneltypar = penelty[0]
        else:
            self.penelty = None

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        if self.penelty is not None:
            chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2) +  Chi2penelty(self.peneltypar, self.penelty[0], self.penelty[1])
        else:
            chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2) 
        
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

































######################################################################  Likelihood fits  ###############################################################




def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8


def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)



class UnbinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, weights=None, bound=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]

        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue
        
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)
        
        # compute the function value
        f = compute_f(self.f, self.data, *par)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = (f>0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive] 
        
        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term
        
        return llh
    
    def default_errordef(self):
        return 0.5





class BinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):
        
        if bound is not None:
            data = np.array(data)
            mask = (data >= bound[0]) & (data <= bound[1])
            data = data[mask]
            if (weights is not None) :
                weights = weights[mask]
            if (weighterrors is not None) :
                weighterrors = weighterrors[mask]

        self.weights = set_var_if_None(weights, data)

        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: 
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)
        
        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights**2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors**2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)


        
        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv
        
        
        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)
        

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)
        
        return ret


    def default_errordef(self):
        return 0.5




import warnings


def xlogyx(x, y):
    
    #compute x*log(y/x) to a good precision especially when y~x
    
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    
    if x<y:
        return x*np.log1p( (y-x) / x )
    else:
        return -x*np.log1p( (x-y) / y )


#compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x<y:
        return w*np.log1p( (y-x) / x )
    else:
        return -w*np.log1p( (x-y) / y )


def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    
    N = np.sum(h)
    n = len(edges)

    ret = 0.
    
    for i in range(n-1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i+1]), nint_subdiv, *par)
        
        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm*N) + (th-tm*N)

            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret





def compute_bin_lh_f(f, edges, h, w2, extended, use_sumw2, badvalue, *par):
    
    mask_positive = (h>0)
    
    N = np.sum(h)
    midpoints = (edges[:-1] + edges[1:]) / 2
    b = np.diff(edges)
    
    midpoints_pos = midpoints[mask_positive]
    b_pos = b[mask_positive]
    h_pos = h[mask_positive]
    
    if use_sumw2:
        warnings.warn('use_sumw2 = True: is not yet implemented, assume False ')
        s = np.ones_like(midpoints_pos)
        pass
    else: 
        s = np.ones_like(midpoints_pos)

    
    E_pos = f(midpoints_pos, *par) * b_pos
    if not extended:
        E_pos = E_pos * N
        
    E_pos[E_pos<0] = badvalue
    
    ans = -np.sum( s*( h_pos*np.log( E_pos/h_pos ) + (h_pos-E_pos) ) )

    return ans







