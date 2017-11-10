import scipy.optimize as opt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def fit_model(x ,y, ey, p0, model):
    # fit data to model
    popt, pcov = opt.curve_fit(f=model, xdata=x, ydata=y, p0=p0, sigma=ey, absolute_sigma=True)
    
    # calculate error on parameters
    perr = np.sqrt(np.diag(pcov))
    
    # calculate chi sq value
    r = np.array([yi - model(xi, *popt) for (xi, yi) in zip(x, y)])
    chisq = sum((r / ey)**2)
    
    return popt, perr, chisq

def init_plot():
    plt.figure(figsize=(8.6,7))
    matplotlib.rcParams.update({'font.size': 16})

def plot_data(x, y, ey):
    plt.errorbar(x, y, yerr=ey, fmt='o')

def plot_model(x, model, p, color, label):
    plt.plot(x, [model(xi, *p) for xi in x] , color=color, linewidth=2.5, label=label)
