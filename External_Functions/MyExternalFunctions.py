import numpy as np                                     
from numpy.linalg import matrix_power                  
import matplotlib.pyplot as plt                                                        
from iminuit import Minuit                            
import sys                                            
from scipy import stats
import sympy as sp
from sympy import integrate







# ################   ERROR PROPAGATION   ####################

# def simple_error_propagation(f, x, dx):
#     """
#     Calculates the error of a function f(x) using the error propagation formula.
#     f: function
#     x: value of the variable
#     dx: error of the variable
#     """
#     return np.sqrt(np.sum((sp.diff(f, x) * dx)**2))




# import sympy

# class ErrorPropagation:
#     def __init__(self, func, variables, errors):
#         self.func = func
#         self.variables = variables
#         self.errors = errors

#         # Initialize symbols for the variables
#         self.symbols = sympy.symbols([f'x{i}' for i in range(len(variables))])

#         # Define the function using the sympy symbols
#         self.symbolic_func = func(*self.symbols)

#         # Calculate the partial derivatives of the function with respect to each variable
#         self.partial_derivatives = [self.symbolic_func.diff(var) for var in self.symbols]

#     def calculate_error(self):
#         # Substitute the values of the variables into the partial derivatives
#         partial_derivatives = [deriv.subs(zip(self.symbols, self.variables)) for deriv in self.partial_derivatives]

#         # Calculate the error using the formula for error propagation
#         error = 0
#         for i in range(len(self.variables)):
#             error += (partial_derivatives[i] * self.errors[i])**2
#         return sympy.sqrt(error)

#     def get_symbolic_formula(self):
#         # Substitute the symbols back into the partial derivatives
#         partial_derivatives = [deriv.subs(zip(self.symbols, self.symbols)) for deriv in self.partial_derivatives]

#         # Calculate the symbolic formula for the propagated error
#         error = 0
#         for i in range(len(self.variables)):
#             error += (partial_derivatives[i] * f'error_{i}')**2
#         return sympy.sqrt(error)

# # Example usage
# def my_func(x, y, z):
#     return x**2 + y**2 + z**2

# ep = ErrorPropagation(my_func, [1, 2, 3], [0.1, 0.2, 0.3])
# error = ep.calculate_error()
# print(error)  # Outputs: 0.9428090415820634

# symbolic_formula = ep.get_symbolic_formula()
# print(symbolic_formula)  # Outputs: sqrt((2*x0*error_0)**2 + (2*x1*error_1)**2 + (2*x2*error_2)**2)




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
    coralation_error = 0

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
                coralation_error += rho[i][j] * error_par[i] * error_par[j] * sp.diff(f, par[i]) * sp.diff(f, par[j])
        coralation_error = coralation_error.simplify()
        fvalue = sp.lambdify(all_par, f)
        error_fvalue = sp.lambdify(all_var, coralation_error)
        return coralation_error, fvalue, error_fvalue

    if cov is not None:
        for i in range(len(par)):
            for j in range(len(par)):
                coralation_error += cov[i][j] * sp.diff(f, par[i]) * sp.diff(f, par[j])
        coralation_error = coralation_error.simplify()
        fvalue = sp.lambdify(all_par, f)
        error_fvalue = sp.lambdify(all_var, coralation_error)
        return coralation_error, fvalue, error_fvalue

    error_f = sp.sqrt(sum([sp.diff(f, i)**2 * j**2 for i, j in zip(par, error_par)]) + coralation_error)

    error_contributions = []
    for i in range(len(par)):
        error_contributions.append((sp.diff(f, par[i])**2 * error_par[i]**2).simplify())



    error_f = error_f.simplify()
    fvalue = sp.lambdify(all_par, f)
    error_fvalue = sp.lambdify(all_var, error_f)
    return error_f, fvalue, error_fvalue, error_contributions