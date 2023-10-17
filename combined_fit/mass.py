import os
import pathlib
import numpy as np

from scipy import interpolate
from astropy.table import Table

from combined_fit import xmax_tools as xmax_tls
from combined_fit import draw

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()


def Plot_Xmax(t, frac, sigma_shift_sys, A, Z, w_zR, w_zR_p, E_fit, model, ext_save=""):
    '''Compute the expected xmax mean and sigma, upload the experimental results, compute the deviance and plot both of them

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
        '''
    logE, Xmax, RMS = expected_Xmax_sigmaXmax(t, frac, A, Z, w_zR, w_zR_p, model, sigma_shift_sys)
    Experimental_Xmax = load_Xmax_data()#load_Xmax_data()
    dev = compute_Xmax_Deviance(logE, Xmax, RMS, Experimental_Xmax, E_fit, sigma_shift_sys) # computation of deviance
    draw.Draw_Xmax(logE, Xmax, RMS, Experimental_Xmax, E_fit, model, sigma_shift_sys*xmax_tls._sysXmax, dev,  saveTitlePlot = "uhecr_atmospheric_depth_"+ext_save)


def expected_Xmax_sigmaXmax(t, frac, A, Z, w_zR, w_zR_p, model, sigma_shift_sys=0):
    '''Compute the expected xmax mean and sigma

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
'''
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
    '''Compute the deviance for Xmax

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
    '''
    XmaxMean_E = interpolate.interp1d(logE, Xmax)
    RMS_E = interpolate.interp1d(logE, RMS)

    BinNumberXmax = np.ndarray.item(np.argwhere(np.around(experimental_xmax['meanLgE'], decimals=2)== E_fit))
    res_Xmax = 		( experimental_xmax["fXmax"][BinNumberXmax:] - XmaxMean_E(experimental_xmax["meanLgE"][BinNumberXmax:]) )/ experimental_xmax['statXmax'][BinNumberXmax:]
    res_sigmaXmax = ( experimental_xmax['fRMS'][BinNumberXmax:] - RMS_E(experimental_xmax["meanLgE"][BinNumberXmax:]) ) / experimental_xmax['statRMS'][BinNumberXmax:]

    Dev = sigma_shift_sys**2 + np.sum(res_Xmax**2 + res_sigmaXmax**2)

    if verbose: print("Composition deviance, from logE=", experimental_xmax['meanLgE'][BinNumberXmax],": ", Dev , "(",len(experimental_xmax['fXmax']) + len(experimental_xmax['fRMS']) - 2*BinNumberXmax, ")")

    return Dev

def get_fractions(t,frac,A,Z,w_zR):
    '''Provide the mass fraction at the top of the atmosphere for a given choice of the parameters at the source

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
    xmax: `Table`
        experimental xmax moments

    Returns
    -------
    A: `list`
        Mass at the top of the atmosphere
    frac_def: `list`
        Mass fractions at the top of the atmosphere
    '''
    sel_A, fractions = [], []
    for i,a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked_A, w_zR, Z[i])
        fractions.append(frac[i]*je/(10**t[i].logE))
        sel_A.append(t[i].A)

    A = np.concatenate(sel_A)
    frac_tot = np.concatenate(fractions, axis=0)
    return A, frac_tot

def get_fractions_p(t, frac, A, Z, w_zR, w_zR_p):
    '''Provide the mass fraction at the top of the atmosphere for a given choice of the parameters at the source

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
    xmax: `Table`
        experimental xmax moments

    Returns
    -------
    A: `list`
        Mass at the top of the atmosphere
    frac_def: `list`
        Mass fractions at the top of the atmosphere
    '''
    sel_A, fractions = [], []
    for i,a in enumerate(A):
        if (i == 0):
            je = t[i].J_E(t[i].tensor_stacked_A, w_zR_p, Z[i])
        else:
            je = t[i].J_E(t[i].tensor_stacked_A, w_zR, Z[i])
        fractions.append(frac[i]*je/(10**t[i].logE))
        sel_A.append(t[i].A)

    A = np.concatenate(sel_A)
    frac_tot = np.concatenate(fractions, axis=0)

    return A, frac_tot


def reduced_fractions(A_old, frac_old,size):
    '''Reduce the mass fraction to a 56 size for all the energies

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
    '''

    #TBD: could likely be fastened
    A = np.zeros(56)
    frac = np.zeros((size,56))
    for j in range (size):
        for i,a in enumerate(A):
            index = int(A_old[i])
            A[index-1] = index
            frac[j][i] = np.sum(frac_old[j][np.where(A_old ==i)])

    return A, frac

def get_fractions_distributions(t,frac,A,Z,w_zR,w_zR_p, xmax):
    '''Provide the mass fraction at the top of the atmosphere for a given choice of the parameters at the source

       specific function to provide mass fractions at each energy distribution

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
    xmax: `Table`
        experimental xmax moments

    Returns
    -------
    A: `list`
        Mass at the top of the atmosphere
    frac_def: `list`
        Mass fractions at the top of the atmosphere
        '''
    A, frac_tot = get_fractions_p(t,frac,A,Z,w_zR, w_zR_p)

    frac_tot = np.transpose(frac_tot)

    start = np.ndarray.item((np.argwhere(t[0].logE == np.around(xmax['meanlgE'][0], decimals = 2))))

    frac_def = np.zeros((len(xmax['maxlgE']),len(A)))

    for j,a in enumerate(xmax['maxlgE']):
        index = int(round((xmax['maxlgE'][j]- xmax['minlgE'][j])/0.1))
        for k in range (index):
            frac_def[j] += frac_tot[j+k+start]

    A_new , frac_new = reduced_fractions(A, frac_def,np.size(xmax['meanlgE']))
    return A_new, frac_new

def load_Xmax_data():
    '''Upload the experimental data (Xmax) for a given Hadronic Interaction model

    Parameters
    -------

    Returns
    -------
    Table: `read`
        experimental data
    '''
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Composition/ICRC2023/Xmax_moments_icrc23.txt')
    return Table.read(filename, format='ascii.ecsv', delimiter=" ", guess=False)
    
def load_Fractions_Data(hadr_model):
    ''' Upload the experimental spectrum

    Parameters
    ----------
    hadr_model : `string`
        hadronic interaction model

    Returns
    -------
    T_J : `table`
       experimental spectrum as read in 'Data'
    '''
    #Load fractions
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Composition/ICRC2023/composition_fractions_icrc2023.txt')
    t_frac = Table.read(filename, format='ascii.basic', delimiter="\t", guess=False)

    #Remove HEAT-based data below 17.8
    t_frac = t_frac[t_frac['meanLgE']>17.8]

    #Build proton spectrum from given HIM
    if hadr_model not in xmax_tls.HIM_list :
        print("Hadronic interaction model not valid! ", hadr_model)
        sys.exit(0)
    if hadr_model == "EPOS-LHC": ext = '_EPOS'
    else: ext = '_SIB'
    
    #Load fractions
    keys_in = ['meanLgE', 'p'+ext, 'p_err_low'+ext, 'p_err_up'+ext, 'p_sys_low'+ext, 'p_sys_up'+ext]
    keys_in += [	'He'+ext, 'He_err_low'+ext, 'He_err_up'+ext, 'He_sys_low'+ext, 'He_sys_up'+ext]
    keys_in += [	'N'+ext, 'N_err_low'+ext, 'N_err_up'+ext, 'N_sys_low'+ext, 'N_sys_up'+ext]
    keys_in += [	'Fe'+ext, 'Fe_err_low'+ext, 'Fe_err_up'+ext, 'Fe_sys_low'+ext, 'Fe_sys_up'+ext]

    keys_out = ['meanLgE', 'p', 'p_err_low', 'p_err_up', 'p_sys_low', 'p_sys_up']
    keys_out += [	'He', 'He_err_low', 'He_err_up', 'He_sys_low', 'He_sys_up']
    keys_out += [	'N', 'N_err_low', 'N_err_up', 'N_sys_low', 'N_sys_up']
    keys_out += [	'Fe', 'Fe_err_low', 'Fe_err_up', 'Fe_sys_low', 'Fe_sys_up']
    for i, k in enumerate(keys_in): t_frac.rename_column(k, keys_out[i])

    return t_frac
