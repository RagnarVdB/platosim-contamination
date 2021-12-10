import numpy as np
import scipy.optimize as opt
from scipy.misc import derivative
from scipy import stats
from inspect import getfullargspec

def substitute(x, index, array):
    """Vervangt één element in een numpy array"""
    new_array = np.array(array)
    new_array[index] = x
    return new_array

def chi2(x, y, dy, model, parameters):
    """Geeft least-squares waarde terug"""
    errors = ((y - model(x, *parameters))**2) / dy**2
    return np.sum(errors)

def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""
    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad
    return gradient


def _fitParameters(x, y, dy, model, guess, bounds=None, method=None, respect_bounds=False, **kwargs):
    """minimaliseerd de chi2 functie"""
    f = lambda p: chi2(x, y, dy, model, p)
    if respect_bounds:
        jac = gradient_respecting_bounds(bounds, f)
        minobj = opt.minimize(f, guess, bounds=bounds, jac=jac, **kwargs)
    else:
        minobj = opt.minimize(f, guess, bounds=bounds, **kwargs)
    return(minobj["x"])

def _errorFit2(x, y, dy, model, minimum):
    """Geeft onzekerheidsinterval rond fitparameters via tweede afgeleide"""
    uncertainties = []
    for i in range(len(minimum)):
        deriv = derivative(lambda p: chi2(x, y, dy, model, substitute(p, i, minimum)), minimum[i], n=2)
        uncertainties.append(np.sqrt(2/deriv))
    return np.array(uncertainties)

def _errorFit(x, y, dy, model, minimum, symmetric=False):
    """Geeft onzekerheidsinterval rond fitparameters via chi2 + 1"""
    uncertainties = []
    for i in range(len(minimum)):
        chi2_parameter = lambda p: chi2(x, y, dy, model, substitute(p, i, minimum))
        chi2_vector = np.vectorize(chi2_parameter)
        minval = (minimum[i], chi2_parameter(minimum[i]))
        f = lambda p: chi2_parameter(p) - minval[1] - 1
        left = opt.fsolve(f, minval[0] *0.9997)[0]
        right = opt.fsolve(f, minval[0]*1.0003)[0]
        if not symmetric:
            uncertainties.append(np.array([minval[0] - left, right - minval[0]]))
        else:
            avg = (right - left)/2
            uncertainties.append(avg)
    return np.array(uncertainties)

def fit(x, y, dy, model, guess=None, bounds=None, method=None, error_method=1, respect_bounds=False, silent=False, pandas=True, **kwargs):
    """Voert fit uit voor gegeven dataset en model"""
    if type(dy) == int and dy == 0:
        dy = 1
        silent=True
    
    modelParams = getfullargspec(model).args
    if guess is None:
        guess = [1]*(len(modelParams) - 1)

    params = _fitParameters(x, y, dy, model, guess, bounds=bounds, method=method, respect_bounds=respect_bounds, **kwargs)
    if error_method == 1:
        errors = _errorFit(x, y, dy, model, params, symmetric=True)
    else:
        errors = _errorFit2(x, y, dy, model, params)

    if not silent:
        ls, rls, p = goodness_fit(x, y, dy, model, params)
        print("Least squares value: " + str(round(ls, 3)))
        print("Reduced Least squares: " + str(round(rls, 3)))
        print("p-value: " + str(round(p, 5)))
    
    arr = np.array([params, np.abs(errors)]).T
    if not pandas:
        return arr
    else:
        import pandas as pd
        return pd.DataFrame(arr, index=modelParams[1:], columns=["value", "error"])

def goodness_fit(x, y, dy, model, params):
    """Geeft chi2, reduced chi2 en p-waarde terug"""
    ls = chi2(x, y, dy, model, params)
    if len(y.shape) == 1:
        df = len(y) - len(params)
    else:
        df = np.prod(y.shape) - len(params)
    p = 1 - stats.chi2.cdf(ls, df)
    return ls, ls/df, p
