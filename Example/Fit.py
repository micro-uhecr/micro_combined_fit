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
from combined_fit import draw

### Main ##########################################################
if __name__ == "__main__":

    ################################# Inputs ##################################
    ###########################################################################

    #Injected masses
    A	= [	1,	 4,	14,	28,	56]
    Z	= [	1,	 2,  7,	14, 26]

    hadr_model = "Sibyll" #"Sibyll" or "EPOS-LHC"
    logRmin = 17.8 #Integrate the injected spectrum from logR_min to get total energy x k
    logE_th = 18.75 # Compute the all-particle spectral deviance from logE_th 	
    isSFR = False # Or False for SMD

    #Initial guess
    sigma_shiftXmax = 1
    if isSFR:
        init_logRcut, init_gamma_nucl, init_gamma_p =  18.24, -0.46, 3.54
        init_E_times_k = [1.79E+46, 8.18E+45, 1.96E+46, 8.55E+45, 1.52E45]
        bnds_E_times_k = [[1E40, 1E50]]*len(ts.A)
        unit_E_times_k = "erg per solar mass"
    else: 
        init_logRcut, init_gamma_nucl, init_gamma_p =  18.33, 0.30, 3.44
        init_E_times_k = [2.63E+36, 6.20E+35, 9.23E+35, 3.73E+35, 1.36E35]
        bnds_E_times_k = [[1E30, 1E40]]*len(ts.A)  
        unit_E_times_k = "erg per solar mass per year"
        
    parameters = np.concatenate((init_E_times_k, [sigma_shiftXmax, init_logRcut, init_gamma_nucl, init_gamma_p]))
    
    bnds_shiftXmax_logRcut_gammanucl_gammap = ( (-5,5), (18., 19.), (-4, 4), (-4, 4)) 
    bnds = np.concatenate((bnds_E_times_k, bnds_shiftXmax_logRcut_gammanucl_gammap))

    #Distribution of Sources
    if isSFR: key = "sfrd"#M Mpc-3 yr-1 
    else: key = "smd"#M Mpc-3
    S_z = ts.Load_evol(file = key+"_local.dat", key=key)

    ################################### Fit ###################################
    ###########################################################################
  
    #Loading spectrum and composition
    t_EJ = sp.load_Spectrum_Data() # load spectrum
    t_EJp = sp.load_ProtonSpectrum_Data(hadr_model) # load proton spectrum
    t_Xmax = mass.load_Xmax_data() # load Xmax
    mini = minim.Minimize_Spectrum_And_Xmax # For minimizing spectrum and Xmax moments deviance
        
    #Do the minimization
    Tensor = ts.upload_Tensor(logRmin = logRmin)
    args = [[Tensor, ts.A, ts.Z, t_EJ, t_EJp, t_Xmax, logE_th, hadr_model, S_z]] # fit Xmax moments
    res = minimize(mini, parameters, tol=1e-5, options={'maxfev': 10000, 'disp': False, 'stra': 1}, bounds=bnds, args=args) # change disp to True if you want to increase the verbosity of the minimizer
    #res.minuit.minos()
    #res.minuit.draw_mnprofile('x6')   
    E_times_k, sigma_shift_sys, logRcut, gamma_nucl, gamma_p = minim.Results(res, len(ts.A), constant.Masses, unit_E_times_k, logRmin, verbose = True)

    #Plot the results
    w_zR_nucl = sp.weight_tensor(S_z, gamma_nucl, logRcut)
    w_zR_p = sp.weight_tensor(S_z, gamma_p, logRcut)
    
    plt.rcParams.update({'font.size': 14,'legend.fontsize': 12})    
    sp.Plot_spectrum(	Tensor, E_times_k, ts.A, ts.Z, w_zR_nucl, w_zR_p, logE_th, hadr_model, isE3dJdE= False, ext_save=key)
    mass.Plot_Xmax(		Tensor, E_times_k, sigma_shift_sys, ts.A, ts.Z, w_zR_nucl, w_zR_p, logE_th, hadr_model, ext_save=key)

    plt.show()
