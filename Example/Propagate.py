import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from combined_fit import spectrum as sp
from combined_fit import constant
from combined_fit import mass
from combined_fit import tensor as ts
from combined_fit import xmax_distr
from combined_fit import draw


### Main ##########################################################
if __name__ == "__main__":

    plt.rcParams.update({'font.size': 14,'legend.fontsize': 12})
    #Spectral parameters
    logRcut = 18.25
    gamma = -1.12
    E_times_k=  np.dot([0.05,  0.36,  0.40,  0.18, 0.01], 3.7150108e+46)  # erg per solar mass

    model="Sibyll2.3d"
    E_th = 18.75 # Compute the deviance from this energy

    #Evolution
    SFRd = ts.Load_evol() # SFRd distribution of sources
    flat = lambda z: SFRd(1) #Flat distribution of sources
    S_z =  SFRd # Can be changed to:  flat

    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)
    Tensor=[]
    Tensor = ts.upload_Tensor()


    mass.Plot_fractions(Tensor, E_times_k, ts.A, ts.Z, w_zR, E_th, model="Sibyll2.3c")
    sp.Plot_spectrum(Tensor, E_times_k, ts.A, ts.Z, w_zR, E_th)
    mass.Plot_Xmax(Tensor,E_times_k,ts.A,ts.Z,w_zR,E_th, model = "Sibyll2.3d")

    xmax,exp_distributions = xmax_distr.set_Xmax_distr()

    arr_reduced, exp_distribution_reduced = [], []
    dx = 20
    for i in range(len(xmax['meanlgE'])):
        min = int(xmax['xMin'][i]/dx)
        max = int(xmax['xMax'][i]/dx)
        arr_reduced.append(np.arange(xmax['xMin'][i], xmax['xMax'][i], dx)+(dx/2))
        exp_distribution_reduced.append(exp_distributions[i][min:max])
    A_tot, frac = mass.get_fractions_distributions(Tensor, E_times_k, ts.A, ts.Z, w_zR, xmax)

    convoluted_gumbel = xmax_distr.Convolve_all(xmax,A_tot, arr_reduced, model)


    draw.Plot_Xmax_distribution(Tensor,E_times_k,ts.A,ts.Z,w_zR,E_th,xmax, model, arr_reduced,exp_distribution_reduced, convoluted_gumbel)

    plt.show()
