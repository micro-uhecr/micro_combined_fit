import os
import pathlib
import numpy as np

from scipy import interpolate
from astropy.table import Table

from combined_fit import xmax_tools as xmax_tls
from combined_fit import draw

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()


def Plot_Xmax(t, frac, sigma_shift_sys, A, Z, w_zR, w_zR_p, E_fit, model, ext_save=""):
    """Compute the expected xmax mean and sigma, upload the experimental results, compute the deviance and plot both of them

    Parameters
    -----------
    t : `tensor`
        upload tensor for the extra-galactic propagation
    frac : `list`
        fractions at the top of the atmosphere
    sigma_shift_sys: `float`
        shift of the model by nsigma_sys        
    A,Z: `list`
        Mass and charge of the injected particles
    w_zR : `list`
        log Rigidity of the injected particles
    E_fit: `float`
        energy from which the deviance is computed
    model: `string`
        Hadronic Interaction model
    ext_save: `string`
        extension for the saved file
    Returns
    -------
    None
    """
    logE, Xmax, RMS = expected_Xmax_sigmaXmax(t, frac, A, Z, w_zR, w_zR_p, model, sigma_shift_sys)
    Experimental_Xmax = load_Xmax_data()
    dev = compute_Xmax_Deviance(logE, Xmax, RMS, Experimental_Xmax, E_fit, sigma_shift_sys) # computation of deviance
    draw.Draw_Xmax(logE, Xmax, RMS, Experimental_Xmax, E_fit, model, sigma_shift_sys*xmax_tls._sysXmax, dev,  saveTitlePlot = "uhecr_atmospheric_depth_"+ext_save)


def expected_Xmax_sigmaXmax(t, frac, A, Z, w_zR, w_zR_p, model, sigma_shift_sys=0):
    """Compute the expected xmax mean and sigma

    Parameters
    ----------
    t : `tensor`
        upload tensor for the extra-galactic propagation
    frac : `list`
        fractions at the top of the atmosphere
    A,Z: `list`
        Mass and charge of the injected particles
    w_zR : `list`
        log Rigidity of the injected particles
    model: `string`
        Hadronic Interaction model
    sigma_shift_sys: `float`
        shift of the model by nsigma_sys

    Returns
    -------
    logE: `list`
        energy bins from the read tensor
    Xmax: `list`
        mean Xmax(for different lgE)
    RMS: `list`
        RMS of Xmax (for different lgE)
    """
    logE = t[0].logE
    A, frac_tot = get_fractions_p(t, frac, A, Z, w_zR, w_zR_p)
    lnA = np.log(A)

    #Fraction shaping    
    place = np.argwhere(np.sum(frac_tot, axis=0) != 0)
    index = np.ndarray.item(place[0])
    logE = logE[index:]
    frac_tot = frac_tot[:,index:]
    frac_tot = np.transpose(frac_tot)

    #Mean Xmax model
    mean_lnA, Xmax, = [], []
    for i, e in enumerate(logE):
        mean_lnA.append(np.dot(frac_tot[i], lnA)/ np.sum(frac_tot[i]))
        Xmax.append(xmax_tls.getXmax(e, mean_lnA[i], model, sigma_shift_sys))

    #Sigma Xmax model    
    RMS = []
    for i, e in enumerate(logE):
        meanVXmax = np.sum(np.dot(xmax_tls.getVar_sh(e, lnA, model), frac_tot[i]))/np.sum(frac_tot[i], axis=0)
        V_A = np.dot(frac_tot[i], lnA**2)/ np.sum(frac_tot[i], axis=0) - mean_lnA[i]**2
        RMS.append(xmax_tls.getXRMS (e, meanVXmax, V_A, model))

    return logE, Xmax, RMS


def compute_Xmax_Deviance(logE, Xmax, RMS, experimental_xmax, E_fit, sigma_shift_sys=0, verbose=False):
    """Compute the deviance for Xmax

    Parameters
    ----------
    logE : `list`
        energy bins from the read tensor
    Xmax : `list`
        mean Xmax (for different lgE)
    RMS: `list`
        Variance of Xmax (for different lgE)
    experimental_xmax : `Table`
        Xmax as read in data folder
    E_fit : `float`
        energy from which the deviance is computed
    sigma_shift_sys: `float`
        shift of the Xmax model by nsigma_sys        
    Returns
    -------
    None
    """
    XmaxMean_E = interpolate.interp1d(logE, Xmax)
    RMS_E = interpolate.interp1d(logE, RMS)

    BinNumberXmax = np.ndarray.item(np.argwhere(np.around(experimental_xmax['meanLgE'], decimals=2)== E_fit))  
    res_Xmax = 		( experimental_xmax["fXmax"][BinNumberXmax:] - XmaxMean_E(experimental_xmax["meanLgE"][BinNumberXmax:]) )/ experimental_xmax['statXmax'][BinNumberXmax:]
    res_sigmaXmax = ( experimental_xmax['fRMS'][BinNumberXmax:] - RMS_E(experimental_xmax["meanLgE"][BinNumberXmax:]) ) / experimental_xmax['statRMS'][BinNumberXmax:]
    
    Dev = sigma_shift_sys**2 + np.sum(res_Xmax**2 + res_sigmaXmax**2)

    if verbose: print("Composition deviance, from logE=", experimental_xmax['meanLgE'][BinNumberXmax],": ", Dev , "(",len(experimental_xmax['fXmax']) + len(experimental_xmax['fRMS']) - 2*BinNumberXmax, ")")
    
    return Dev


def get_fractions_p(t, frac, A, Z, w_zR, w_zR_p):
    """Provide the mass fraction at the top of the atmosphere for a given choice of the parameters at the source

    Parameters
    ----------
    t : `tensor`
        upload tensor for the extra-galactic propagation
    frac : `list`
        fractions at the top of the atmosphere
    A,Z: `list`
        Mass and charge of the injected particles
    w_zR : `list`
        log Rigidity of the injected particles
    Returns
    -------
    A: `list`
        Mass at the top of the atmosphere
    frac_def: `list`
        Mass fractions at the top of the atmosphere
            #if (i == 0):
                #je = t[i].J_E(t[i].tensor_stacked, w_zR_p, Z[i])
            #else:
    """
    sel_A, fractions = [], []
    for i,a in enumerate(A):
        if i==0:
            je = t[i].J_E(t[i].tensor_stacked_A, w_zR_p, Z[i])
        else:
            je = t[i].J_E(t[i].tensor_stacked_A, w_zR, Z[i])
        fractions.append(frac[i]*je/(10**t[i].logE))
        sel_A.append(t[i].A)

    A = np.concatenate(sel_A)
    frac_tot = np.concatenate(fractions, axis=0)
    
    return A, frac_tot


def reduced_fractions(A_old, frac_old,size):
    """Reduce the mass fraction to a 56 size for all the energies

    Parameters
    ----------
    A_old : `list`
        concatenated mass at the top of the atmosphere
    frac_old : `list`
        concatenated mass fractions at the top of the atmosphere
    size: `int`
        number of concatenated fractions
    Returns
    -------
    A: `list`
        Mass at the top of the atmosphere (56)
    frac: `list`
        Mass fractions at the top of the atmosphere  (56)
    """
    
    #TBD: could likely be fastened
    A = np.zeros(56)
    frac = np.zeros((size,56))
    for j in range (size):
        for i,a in enumerate(A):
            frac[j][i] = np.sum(frac_old[j][np.where(A_old ==i)])

    return A, frac


def load_Xmax_data():
    """Upload the experimental data (Xmax) for a given hadronic Interaction model

    Parameters
    -------

    Returns
    -------
    Table: `read`
        experimental data
    """
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Composition/Xmax_moments_icrc17_v2.txt')
    return Table.read(filename, format='ascii.ecsv', delimiter=" ", guess=False)
