import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from iminuit import minimize

from combined_fit import spectrum as sp
from combined_fit import minimizer as minim
from combined_fit import tensor as ts
from combined_fit import mass
from combined_fit import constant
from combined_fit import xmax_distr
from combined_fit import draw

### Main ##########################################################
if __name__ == "__main__":


    start = time.time()

    #Initial guess
    #init_E_times_k = 2.5e45

    #init_E_times_k = [2.5e45,1.8e46,1.8e46,2.5e45,0] #####SFR
    init_E_times_k = [1.e+45, 1.71e+43, 1.99e+43, 1.15289e+43, 0.02e+42] #SMD

    init_Rcut, init_gamma, init_gamma_p =  18.2, -1,3 #SMD
    #init_Rcut, init_gamma, init_gamma_p =  18.35, -0.4,3.24 #SFR
    init_E_times_k = np.transpose(init_E_times_k)
    #Boundaries
    #bnds_E_times_k = [[1.e45, 1.e47]]*len(ts.A) # Boundaries E_times_k SFR
    bnds_E_times_k = [[1.e43, 1.e47]]*len(ts.A) # Boundaries E_times_k SMD

    bnds_Rcut_gamma = ( (18, 19), (-3, 0), (0,4)) # Boundaries Rcut & gamma


    model = "Sibyll" #"Sibyll" or "EPOS-LHC"
    E_th = 18.75 # Compute the deviance from the 4th spectrum points (& equivalent energy for lnA)

    #Distribution of Sources
    SFRd = ts.Load_evol() # SFRd distribution of sources
    flat = lambda z: SFRd(1) #Flat distribution of sources
    S_z =  SFRd # Can be changed to:  flat
    f_z = ts.Load_evol_new(file = "smd_local.dat", key="smd")
    S_z = lambda z : 1/sp.dzdt(z)*f_z(z)



################## Fit ####################################################

    #Loading spectrum, compo, tensor, evolution of sources
    Tensor=[]
    Tensor = ts.upload_Tensor()


    t_EJ = sp.load_Spectrum_Data() # load spectrum
    t_lnA = mass.load_lnA_Data(model) # load lnA
    t_Xmax = mass.load_Xmax_data() # load Xmax

    #---------- initializing the xmax distribution ------------------
    '''xmax,exp_distributions = xmax_distr.set_Xmax_distr()
    arr_reduced, exp_distribution_reduced = [], []
    A_tot = np.arange(1,(ts.A[-1]+1))
    dx = 20

    for i in range(len(xmax['meanlgE'])):
        min = int(xmax['xMin'][i]/dx)
        max = int(xmax['xMax'][i]/dx)
        arr_reduced.append(np.arange(xmax['xMin'][i], xmax['xMax'][i], dx)+(dx/2))
        exp_distribution_reduced.append(exp_distributions[i][min:max])

    convoluted_gumbel = xmax_distr.Convolve_all(xmax,A_tot, arr_reduced, model) '''
    #---------- ------------------------------- ------------------

    StartingFit = np.ndarray.item((np.argwhere(t_EJ['logE'] == E_th)))
    print("Starting the fit from logE = ", t_EJ['logE'][StartingFit])
    #mini = minim.Minimize_Spectrum_And_Compo # For minimizing spectrum and lnA deviance
    mini = minim.Minimize_Spectrum_And_Xmax # For minimizing spectrum and Xmax moments deviance
    #mini = minim.Minimize_Spectrum_And_Distribution # For minimizing spectrum and Xmax distr deviance

    #Do the minimization
    bnds = np.concatenate((bnds_E_times_k, bnds_Rcut_gamma))
    #print(np.shape([init_E_times_k]))
    parameters = np.concatenate(([init_E_times_k[0],init_E_times_k[1],init_E_times_k[2],init_E_times_k[3],init_E_times_k[4]], [init_Rcut, init_gamma, init_gamma_p]))
    args = [[Tensor, ts.A, ts.Z, t_EJ, t_Xmax, E_th,model, S_z]] # fit Xmax moments
    #args = [[Tensor, ts.A, ts.Z, t_EJ, t_lnA, E_th,model, S_z]] # fit lnA
    #args = [[Tensor, ts.A, ts.Z, t_EJ, xmax, E_th,arr_reduced,exp_distribution_reduced,convoluted_gumbel, S_z]] # fit Xmax distr
    res = minimize(mini, parameters, tol=1e-5, options={'maxfev': 10000, 'disp': False, 'stra': 1}, bounds=bnds, args=args) # change disp to True if you want to increase the verbosity of the minimizer

    err = []
    E_times_k = res.x[:len(ts.A)]

    logRcut = res.x[len(ts.A)]
    err_Rcut = np.sqrt(res.hess_inv[len(ts.A)][len(ts.A)])

    gamma = res.x[len(ts.A)+1]
    err_gamma = np.sqrt(res.hess_inv[len(ts.A)+1][len(ts.A)+1])
    gamma_p = res.x[len(ts.A)+2]
    err_gamma_p = np.sqrt(res.hess_inv[len(ts.A)+2][len(ts.A)+2])

    for i in range (len(ts.A)):
        err.append(np.sqrt(res.hess_inv[i][i]))
    print("\n\n\n\nResult of the fit:\n")
    print( "----------------------------------------------\n")
    print(" Spectral parameters \n")
    print("----------------------------------------------\n")
    print("gamma:  ", np.around(gamma, decimals = 2), " +/- ", np.around(err_gamma, decimals = 2))
    print("log(Rcut/V):  ", np.around(logRcut, decimals = 2)," +/- ", np.around(err_Rcut, decimals = 2) )
    print("gamma p:  ", np.around(gamma_p, decimals = 2), " +/- ", np.around(err_gamma_p, decimals = 2))

    for i in range (len(ts.A)):
        print(constant.Masses[i]," (%) :  ",np.around(E_times_k[i]/np.sum(E_times_k), decimals = 2)," +/- " , np.around(err[i]/np.sum(E_times_k), decimals = 2), " * ", np.sum(E_times_k), "erg per solar mass" )
    #res.minuit.minos()
    #res.minuit.draw_mnprofile('x6')
    print(res.minuit)
    print("\n\n\n\n Correlation Matrix: \n")
    corr_matrix = np.zeros((len(ts.A)+3,len(ts.A)+3))

    for i in range (len(ts.A)+3):
        for j in range (len(ts.A)+3):
            corr_matrix[i][j] = np.around(res.hess_inv[i][j]/np.sqrt(res.hess_inv[i][i]*res.hess_inv[j][j]), decimals = 3)
    print(corr_matrix)

    w_R_p = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma_p, logRcut)
    w_zR_p = lambda ZA, z, logR: w_R_p(ZA, logR)/sp.dzdt(z)*S_z(z)

    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)

    mass.Plot_fractions(Tensor, E_times_k, ts.A, ts.Z, w_zR,w_zR_p, E_th, model)
    sp.Plot_spectrum(Tensor, E_times_k, ts.A, ts.Z, w_zR,w_zR_p, E_th)
    mass.Plot_Xmax(Tensor,E_times_k,ts.A,ts.Z,w_zR, w_zR_p, E_th, model)
    end = time.time()
    print("Elapsed time_tot ", end - start)
    #draw.Plot_Xmax_distribution(Tensor,E_times_k,ts.A,ts.Z,w_zR,E_th,xmax, model, arr_reduced,exp_distribution_reduced, convoluted_gumbel,)


    plt.show()
