import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pathlib

from combined_fit import spectrum as sp
from combined_fit import constant
from combined_fit import mass
from combined_fit import tensor as ts
from combined_fit import xmax_distr
from combined_fit import gumbel

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

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

    xmax,exp_distributions = xmax_distr.set_Xmax_distr() #set experimental distribution
    meanLgE = (xmax['maxlgE']+xmax['minlgE'])/2
    acc = Table.read(os.environ['OLDPWD']+'/Private_data/ICRC2019/acceptance.txt', format='ascii.basic', delimiter=" ", guess=False)
    reso = Table.read(os.environ['OLDPWD']+'/Private_data/ICRC2019/resolution.txt', format='ascii.basic', delimiter=" ", guess=False)

    A_tot, frac = mass.get_fractions_distributions(Tensor, E_times_k, ts.A, ts.Z, w_zR, xmax)
    arr_reduced, exp_distribution_reduced = [], []

    for i in range(len(xmax['meanlgE'])):
        min = int(xmax['xMin'][i]/20)
        max = int(xmax['xMax'][i]/20)
        arr_reduced.append(np.arange(xmax['xMin'][i], xmax['xMax'][i], 20)+10)
        exp_distribution_reduced.append(exp_distributions[i][min:max])
    start = time.time()

    gumbel_tot_slow = []
    meanLgE =  xmax['meanlgE']
    filename = os.path.join(COMBINED_FIT_BASE_DIR, '../Private_data/ICRC2017/acceptance_resolution_2017.txt')
    data = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)
    for i in range( len(meanLgE)):
        p_acc = [data['x1'][i], data['x2'][i], data['l1'][i], data['l2'][i] ]
        a =  xmax_distr.acceptance_func(p_acc,arr_reduced[i])

        p_reso = [data['resparam0'][i], data['resparam2'][i], data['resparam4'][i]]
        x, r = xmax_distr.resolution_func(p_reso)
        sum = np.zeros((len(A_tot),len(arr_reduced[i])))

        for j in range(len(A_tot)):
            if A_tot[j] > 0:
                min = int((np.min(arr_reduced[i])-10)/20)
                max = int((np.max(arr_reduced[i])+10)/20)
                g = gumbel.Gumbel_function(A_tot[j], meanLgE[i], model)
                g.xmax = g.xmax[min:max]
                g.y = g.y[min:max]

                sum[j] = xmax_distr.convolve_diff_slow(g.xmax, g.y, r,x,a)
                sum[j] = sum[j]/np.sum(sum[j])

        gumbel_tot_slow.append(sum)

    end = time.time()
    print("Elapsed time_slow ", end - start)
    start = time.time()

    gumbel_tot_fast = []
    meanLgE =  xmax['meanlgE']
    data = Table.read(os.environ['OLDPWD']+'/Private_data/ICRC2017/acceptance_resolution_2017.txt', format='ascii.basic', delimiter=" ", guess=False)
    for i in range( len(meanLgE)):
        p_acc = [data['x1'][i], data['x2'][i], data['l1'][i], data['l2'][i] ]
        a =  xmax_distr.acceptance_func(p_acc,arr_reduced[i])

        p_reso = [data['resparam0'][i], data['resparam2'][i], data['resparam4'][i]]
        x, r = xmax_distr.resolution_func(p_reso)
        sum = np.zeros((len(A_tot),len(arr_reduced[i])))

        for j in range(len(A_tot)):
            if A_tot[j] > 0:
                min = int((np.min(arr_reduced[i])-10)/20)
                max = int((np.max(arr_reduced[i])+10)/20)
                g = gumbel.Gumbel_function(A_tot[j], meanLgE[i], model)
                g.xmax = g.xmax[min:max]
                g.y = g.y[min:max]

                sum[j] = xmax_distr.convolve_diff_fast(g.xmax, g.y, r,x)
                sum[j] = sum[j]/np.sum(sum[j])
                sum[j] = np.multiply(sum[j], a*20*20)

        gumbel_tot_fast.append(sum)

    end = time.time()
    print("Elapsed time_fast ", end - start)

    #plt.tight_layout()
