import numpy as np
from numba import jit

from combined_fit import xmax_tools as xmax_tls
from combined_fit import spectrum as sp
from combined_fit import mass


@jit
def Xmax_Minimization(parms, args, w_zR, w_zR_p):
    """ Compute the Xmax minimization

    Parameters
    ----------
    parms: `list`
        List of parameters to be fitted
    args: `list`
        List of arguments needed to the function
    w_zR: `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of Xmax
    """
    
    t = args[0]
    A = args[1]
    Z = args[2]
    xmax = args[5]
    E_th = args[6]
    model = args[7]
    
    #Computation##################################
    sigma_shift_sys = parms[len(A)]
    logE, XmaxLM, RMSLM = mass.expected_Xmax_sigmaXmax(t, parms[:len(A)], A, Z, w_zR, w_zR_p, model, sigma_shift_sys)     
    return mass.compute_Xmax_Deviance(logE, XmaxLM, RMSLM, xmax, E_th, sigma_shift_sys)


@jit
def Spectrum_Minimization_p(parms, args, w_zR, w_zR_p, isRealUnits = True):
    """ Performing the minimization of the energy spectrum

    Parameters
    ----------
    parms: `list`
        List of parameters to be fitted
    args: `list`
        List of arguments needed to the function
    w_zR: `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of Spectrum
    """
   
    t = args[0]
    A = args[1]
    Z = args[2]
    T_J = args[3]
    T_Jp = args[4]    
    E_th = args[6]
    
    #Computation##################################
    logE, total_spectrum, spectrum_per_inj, det_spectra = sp.Compute_expected_spectrum(t, parms[:len(A)], A, Z, w_zR, w_zR_p)
    norm, dev = sp.Deviance_spectrum_proton_p(logE, total_spectrum, T_J, det_spectra, T_Jp, E_th)

    return dev
   
    
def Minimize_Spectrum_And_Xmax(parms, args, verbose = True):
    """ Computing the deviance of the energy spectrum and Xmax

    Parameters
    ----------
    parms: `list`
        List of parameters to be fitted
    args: `list`
        List of arguments needed to the function
    verbose: `bool`
        Increse the verbosity of the function

    Returns
    -------
    Deviance: 'float'
        return the sum of the deviance of spectrum and Xmax
    """
    logRcut = parms[-3]
    gamma = parms[-2]
    gamma_p = parms[-1]
    S_z = args[-1]

    w_zR = sp.weight_tensor(S_z, gamma, logRcut)
    w_zR_p = sp.weight_tensor(S_z, gamma_p, logRcut)

    Dev_Compo = Xmax_Minimization(parms, args, w_zR, w_zR_p)
    Dev_Spectrum = Spectrum_Minimization_p(parms, args, w_zR, w_zR_p)

    if verbose:
        print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2), " | Composition deviance: ", np.around(Dev_Compo, decimals=2), "  | gamma: ", np.around(gamma, decimals=2), "  | logRcut: ", np.around(logRcut, decimals=2), "| gamma p ", np.around(gamma_p, decimals=2), " ", parms[0], " ", parms[1]," ", parms[2]," ", parms[3]," ", parms[4])
        
    return Dev_Compo + Dev_Spectrum
    
    
def Minimize_Spectrum(parms, args, verbose = True):
    """ Computing the deviance of the energy spectrum

    Parameters
    ----------
    parms: `list`
        List of parameters to be fitted
    args: `list`
        List of arguments needed to the function
    verbose: `bool`
        Increse the verbosity of the function

    Returns
    -------
    Deviance: 'float'
        return the sum of the deviance of spectrum and Xmax
    """
    logRcut = parms[-3]
    gamma = parms[-2]
    gamma_p = parms[-1]
    S_z = args[-1]

    w_zR = sp.weight_tensor(S_z, gamma, logRcut)
    w_zR_p = sp.weight_tensor(S_z, gamma_p, logRcut)

    Dev_Spectrum = Spectrum_Minimization_p(parms, args, w_zR, w_zR_p)

    if verbose:
        print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2), "  | gamma: ", np.around(gamma, decimals=2), "  | logRcut: ", np.around(logRcut, decimals=2), "| gamma p ", np.around(gamma_p, decimals=2), " ", parms[0], " ", parms[1]," ", parms[2]," ", parms[3]," ", parms[4])
        
    return Dev_Spectrum    
    
    
def Results(res, nA, masses, unit_E_times_k, logRmin, verbose = True):
    """ Computing the deviance of the energy spectrum and Xmax

    Parameters
    ----------
    res: `object`
        output of iminuit minimize
    nA: `int`
        number of nuclear component
    masses: `list of str`
        string with names of nuclei
    unit_E_times_k: `str`
        unit of E_times_k
    verbose: `bool`
        Increse the verbosity of the function

    Returns
    -------
    E_times_k: `list`
        List of energy flux of nuclear components
    sigma_shift_sys: `float`
        shift of the Xmax model by nsigma_sys        
    logRcut: 'float'
        cut off rigidity
    gamma_nucl: 'float'
        index of nuclei
    gamma_p: 'float'
        index of protons
    """
           
    #Extract best-fit parameters
    E_times_k = res.x[:nA]
    err = [np.sqrt(res.hess_inv[i][i]) for i in range (nA)]

    sigma_shift_sys = res.x[nA]
    err_shift_sys = np.sqrt(res.hess_inv[nA][nA])

    logRcut = res.x[nA+1]
    err_logRcut = np.sqrt(res.hess_inv[nA+1][nA+1])

    gamma_nucl = res.x[nA+2]
    err_gamma_nucl = np.sqrt(res.hess_inv[nA+2][nA+2])

    gamma_p = res.x[nA+3]
    err_gamma_p = np.sqrt(res.hess_inv[nA+3][nA+3])

    #Extract correlation matrix
    corr_matrix = np.zeros((nA+4,nA+4))
    for i in range (nA+4):
        for j in range (nA+4):
            corr_matrix[i][j] = np.around(res.hess_inv[i][j]/np.sqrt(res.hess_inv[i][i]*res.hess_inv[j][j]), decimals = 3)

    #Compute total energy
    Etot_times_k = np.sum(E_times_k)
    err_Etot_times_k = np.sqrt(np.sum(res.hess_inv))
    powEtot = np.power(10, np.floor(np.log10(Etot_times_k)))
    
    #Print the results
    if verbose:
        print(res.minuit)       
        print("\n\n\n\n Correlation Matrix: \n")
        print(corr_matrix)

        print("\n\n\n\nResult of the fit:\n")
        print("Xmax shift [sigma]:  ", np.around(sigma_shift_sys, decimals = 2)," +/- ", np.around(err_shift_sys, decimals = 2) )
        print( "----------------------------------------------\n")
        print(" Spectral parameters \n")
        print("----------------------------------------------\n")
        print("log(Rcut/V):  ", np.around(logRcut, decimals = 2)," +/- ", np.around(err_logRcut, decimals = 2) )
        print("gamma p:  ", np.around(gamma_p, decimals = 2), " +/- ", np.around(err_gamma_p, decimals = 2))
        print("gamma nucl:  ", np.around(gamma_nucl, decimals = 2), " +/- ", np.around(err_gamma_nucl, decimals = 2))      
        print("k x Etot  above log(R/V) =",logRmin,": (", np.around(Etot_times_k/powEtot, decimals = 2), "+/-", np.around(err_Etot_times_k/powEtot, decimals = 2), ") x ",np.format_float_scientific(powEtot, precision=0), unit_E_times_k)
        for i in range (nA):
            print(masses[i]," (%):  (",np.around(100*E_times_k[i]/Etot_times_k, decimals = 1),"+/-" , np.around(100*err[i]/Etot_times_k, decimals = 1), ") x ", np.format_float_scientific(Etot_times_k, precision=2), unit_E_times_k)
            
    return E_times_k, sigma_shift_sys, logRcut, gamma_nucl, gamma_p       
