import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from combined_fit import spectrum as sp
from combined_fit import constant
from combined_fit import mass
from combined_fit import tensor as ts
from combined_fit import xmax_distr


### Main ##########################################################
if __name__ == "__main__":

    plt.rcParams.update({'font.size': 14,'legend.fontsize': 12})

    #Spectral parameters
    logRcut = 18.25
    gamma = -1.12
    E_times_k=  np.dot([0.05,  0.36,  0.40,  0.18, 0.01], 3.7150108e+46)  # erg per solar mass

    model="EPOS-LHC"
    E_th = 18.75 # Compute the deviance from this energy

    #Evolution
    SFRd = ts.Load_evol() # SFRd distribution of sources
    flat = lambda z: SFRd(1) #Flat distribution of sources
    S_z =  SFRd # Can be changed to:  flat

    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)
    Tensor=[]
    Tensor = ts.upload_Tensor()


    #mass.Plot_fractions(Tensor, E_times_k, ts.A, ts.Z, w_zR, E_th, model="Sibyll2.3c")
    #sp.Plot_spectrum(Tensor, E_times_k, ts.A, ts.Z, w_zR, E_th)
    #mass.Plot_Xmax(Tensor,E_times_k,ts.A,ts.Z,w_zR,E_th, model = "Sibyll2.3d")

    fig = plt.figure()

    xmax,exp_distributions = xmax_distr.set_Xmax_distr()

    arr_reduced, exp_distribution_reduced = [], []
    dx = 20
    for i in range(len(xmax['meanlgE'])):
        min = int(xmax['xMin'][i]/dx)
        max = int(xmax['xMax'][i]/dx)
        arr_reduced.append(np.arange(xmax['xMin'][i], xmax['xMax'][i], dx)+(dx/2))
        exp_distribution_reduced.append(exp_distributions[i][min:max])
    #A_tot, frac = mass.get_fractions_distributions(Tensor, E_times_k, ts.A, ts.Z, w_zR, xmax)
    A_tot =[1,4,14,56]

    t = Table.read('../Private_data/ICRC2017/XmaxFit-eposlhc-ICRC2017.txt', format='ascii.basic', delimiter=" ", guess=False)
    fig = plt.figure()

    convoluted_gumbel = xmax_distr.Convolve_all(xmax,A_tot, arr_reduced, model)
    shift = 6 # avoiding energy bins between 17.2 and 17.8
    f_new = np.zeros((len(xmax['meanlgE']),len(A_tot)))
    for j in range(len(xmax['meanlgE'])):
        if (xmax['meanlgE'][j] > 0):
            f_new[j][0] = t['fH'][shift+j]
            f_new[j][1] = t['fHe'][shift+j]
            f_new[j][2] = t['fN'][shift+j]
            f_new[j][3] = t['fFe'][shift+j]
        print(j, " ", xmax['meanlgE'][j], " ", f_new[j][0], " ", f_new[j][1], " ", f_new[j][2], " ", f_new[j][3], " ", t['Dev'][shift+j])

    #group = Print_Xmax_group_17(A,A_new,f_new,meanLgE,exp_distributions, model, big)
    Dev = 0
    points = 0
    diff = []
    for i in range(len(xmax['meanlgE'])):

        sum = np.zeros((len(A_tot),len(arr_reduced[i])))
        print(xmax['meanlgE'][i], " ", f_new[i])
        f_new[i] = f_new[i]/np.sum (f_new[i])
        for j in range(len(A_tot)):
            sum[j] = np.multiply(convoluted_gumbel[i][j], f_new[i][j])

        final = np.sum(sum, axis = 0)/np.sum(f_new[i])

        data = exp_distribution_reduced[i]
        idk = data > 0
        if (xmax['meanlgE'][i] > E_th): points +=(np.sum(np.multiply(idk,1)))
        Dev = xmax_distr.deviance_Xmax_distr(data,final, xmax['nEntries'][i])
        diff.append(abs((Dev - t['Dev'][shift+i])))

        x = np.arange(0,2000,dx)
        min = int((np.min(exp_distributions[i])-(dx/2))/dx)
        max = int((np.max(exp_distributions[i])+(dx/2))/dx)
        x = x[min:max]
        print(xmax['meanlgE'][i], " ", Dev, " ", points, t['Dev'][shift+i], " ",((Dev-t['Dev'][shift+i])/Dev))

        err = np.zeros(len(data))
        for j in range (len(data)):
            if(data[j] > 0): err[j] = np.sqrt(data[j])
        err = err/np.sum(data)
        data = data/np.sum(data)

    plt.ylabel('Bias, $ |Dev_{C1} -  Dev_{ref} | $', fontsize = 21)
    plt.plot(xmax['meanlgE'], diff,'o', color='black')



    plt.show()
xmax_distr
