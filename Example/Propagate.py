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
    # best fit values SFR Spectral parameters
    #logRcut_n = 18.28
    #gamma_n = -0.88
    #logRcut_p = 18.28
    #gamma_p = 2.85
    #E_times_k = [9.999999999943266e+46,   1.8499081186717069e+46,   1.9291523704992483e+46,   8.326664827209188e+45,   1.0000000000341408e+45]
    #E_times_k=  np.dot([0.98,  0.01,  0.01,  0.0, 0.0], 1.0222631541615742e+46)  # erg per solar mass

    # best fit values SMD Spectral parameters
    logRcut_n = 18.31
    gamma_n = -0.0
    logRcut_p = 18.31
    gamma_p = 2.95
    E_times_k= [ 9.999999999999408e+45,  2.126470897209981e+44,   9.3372424229318e+43 ,  5.164694985238282e+43 ,  1.000000000093353e+43]

    model="Sibyll"
    E_th = 18.75 # Compute the deviance from this energy

    #Evolution
    SFRd = ts.Load_evol() # SFRd distribution of sources
    flat = lambda z: SFRd(1) #Flat distribution of sources
    S_z =  SFRd # Can be changed to:  flat

    f_z = ts.Load_evol_new(file = "smd_local.dat", key="smd") #uncomment these two lines if you want SMD
    S_z = lambda z : 1/sp.dzdt(z)*f_z(z)

    w_R_p = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma_p, logRcut_p)
    w_zR_p = lambda ZA, z, logR: w_R_p(ZA, logR)/sp.dzdt(z)*S_z(z)

    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma_n, logRcut_n)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)

    Tensor=[]
    Tensor = ts.upload_Tensor()


    mass.Plot_fractions(Tensor, E_times_k, ts.A, ts.Z, w_zR,w_zR_p, E_th, model)
    sp.Plot_spectrum(Tensor, E_times_k, ts.A, ts.Z, w_zR,w_zR_p, E_th)
    mass.Plot_Xmax(Tensor,E_times_k,ts.A,ts.Z,w_zR,w_zR_p,E_th, model)

    '''xmax,exp_distributions = xmax_distr.set_Xmax_distr()

    arr_reduced, exp_distribution_reduced = [], []
    dx = 20
    for i in range(len(xmax['meanlgE'])):
        min = int(xmax['xMin'][i]/dx)
        max = int(xmax['xMax'][i]/dx)
        arr_reduced.append(np.arange(xmax['xMin'][i], xmax['xMax'][i], dx)+(dx/2))
        exp_distribution_reduced.append(exp_distributions[i][min:max])
    A_tot, frac = mass.get_fractions_distributions(Tensor, E_times_k, ts.A, ts.Z, w_zR, xmax)

    convoluted_gumbel = xmax_distr.Convolve_all(xmax,A_tot, arr_reduced, model)


    draw.Plot_Xmax_distribution(Tensor,E_times_k,ts.A,ts.Z,w_zR,E_th,xmax, model, arr_reduced,exp_distribution_reduced, convoluted_gumbel)'''

    plt.show()
