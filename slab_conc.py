import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numba import jit
import string
from scipy.ndimage import gaussian_filter1d
from mdtraj import element
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from matplotlib.colors import LogNorm
import warnings
import itertools
warnings.filterwarnings('ignore')
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from statsmodels.tsa.stattools import acf

# from analyse import *

import sys

sys.path.append('/Users/sobuelow/software/BLOCKING')
from main import BlockAnalysis

def calcProfile(df_proteins,m,T,L,value,error,tmin=1200,tmax=None,fbase='.'):
    h = np.load(f'{fbase}/{m}/{T:d}/{m}_{T:d}.npy')
    print(h.shape)
    fasta = df_proteins.loc[m].fasta
    N = len(fasta)
    conv = 100/6.022/N/L/L*1e3
    h = h[tmin:tmax]*conv # corresponds to (binned) concentration in mM
    lz = h.shape[1]+1
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d) # hyperbolic function, parameters correspond to csat etc.
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h,axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[1,1,1,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4)) # fit to hyperbolic function
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-.5*res1.x[3],-res2.x[2]+.5*res2.x[3]] # position of interface - half width
    cutoffs2 = [res1.x[2]+6*res1.x[3],-res2.x[2]-6*res2.x[3]] # get far enough from interface for dilute phase calculation

    if np.abs(cutoffs2[1]/cutoffs2[0]) > 2: # ratio between right and left should be close to 1
        print('WRONG',m,cutoffs1,cutoffs2)
        print(res1.x,res2.x)
    if np.abs(cutoffs2[1]/cutoffs2[0]) < 0.5:
        print('WRONG',m,cutoffs1,cutoffs2)
        print(res1.x,res2.x)
        plt.plot(z1, h1)
        plt.plot(z2, h2)
        plt.plot(z1,profile(z1,*res1.x),color='tab:blue')
        plt.plot(z2,profile(z2,*res2.x),color='tab:orange')
        cutoffs2[0] = -cutoffs2[1]
        print(cutoffs2)

    bool1 = np.logical_and(z<cutoffs1[0],z>cutoffs1[1])
    bool2 = np.logical_or(z>cutoffs2[0],z<cutoffs2[1])

    dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, h) # concentration in range [bool2]
    denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, h)

    dil = hm[bool2].mean() # average concentration
    den = hm[bool1].mean()

    block_dil = BlockAnalysis(dilarray)
    block_den = BlockAnalysis(denarray)
    block_dil.SEM()
    block_den.SEM()

    value.loc[m,'{:d}_dil'.format(T)] = block_dil.av 
    value.loc[m,'{:d}_den'.format(T)] = block_den.av 

    error.loc[m,'{:d}_dil'.format(T)] = block_dil.sem 
    error.loc[m,'{:d}_den'.format(T)] = block_den.sem

    return(value, error)
