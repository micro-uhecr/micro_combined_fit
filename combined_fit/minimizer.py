import time
import numpy as np
from scipy import interpolate
from numba import jit

from combined_fit import xmax_tools as xmax_tls
from combined_fit import spectrum as sp
from combined_fit import mass


@jit
def Fractions_Minimization(parms, args, w_zR):
    ''' Performing the minimization of the mass fraction
    at the top of the atmosphere (lnA and sigma(lnA)).

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    w_zR : `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of lnA
    '''
    t = args[0]
    A = args[1]
    Z = args[2]
    t_lnA = args[4]
    E_th = args[5]
    #Computation##################################
    sel_A, fractions = [], []

    for i,a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked_A, w_zR, Z[i])
        fractions.append(parms[i]*je/(10**t[i].logE))
        sel_A.append(t[i].A)


    logE = t[0].logE
    StartingFrom = np.asscalar(np.argwhere(t_lnA['logE'] == E_th))

    A = np.concatenate(sel_A)

    frac_tot = np.concatenate(fractions, axis=0)
    place = np.argwhere(np.sum(frac_tot, axis=0) != 0)
    index = np.asscalar(place[0])

    frac_tot = frac_tot[:,index:]
    logE = logE[index:]

    start = time.time()
    mean_A = np.dot(np.log(A),frac_tot)/ np.sum(frac_tot, axis=0)
    avA_E = interpolate.interp1d(logE, mean_A)

    V_A = np.dot(np.log(A)**2,frac_tot) / np.sum(frac_tot, axis=0) - mean_A**2
    sig2A_E = interpolate.interp1d(logE, V_A)


    Dev = np.sum(( t_lnA['mean_lnA'][StartingFrom:] - avA_E(t_lnA['logE'][StartingFrom:]) )**2/ t_lnA['mean_Stat'][StartingFrom:]**2)
    Dev += np.sum(( t_lnA['Var_lnA'][StartingFrom:] - sig2A_E(t_lnA['logE'][StartingFrom:]) )**2 / ((t_lnA['Var_StatUp'][StartingFrom:]+t_lnA['Var_StatLow'][StartingFrom:])/2)**2)
    end = time.time()


    return Dev


@jit
def Xmax_Minimization(parms, args, w_zR):
    ''' Compute the Xmax minimization

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    w_zR : `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of Xmax
    '''
    t = args[0]
    A = args[1]
    Z = args[2]
    xmax = args[4]
    E_th = args[5]
    model = args[6]
    #Computation##################################

    sel_A, fractions = [], []

    for i,a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked_A, w_zR, Z[i])
        fractions.append(parms[i]*je/(10**t[i].logE))
        sel_A.append(t[i].A)

    BinNumberXmax = np.asscalar(np.argwhere(np.around(xmax["meanLgE"], decimals=2)== E_th))

    logE = t[0].logE
    A = np.concatenate(sel_A)

    frac_tot = np.concatenate(fractions, axis=0)

    place = np.argwhere(np.sum(frac_tot, axis=0) != 0)
    index = np.asscalar(place[0])

    frac_tot = frac_tot[:,index:]
    logE = logE[index:]

    frac_tot = np.transpose(frac_tot)
    mean_A, XmaxLM, V_A, RMSLM = [], [], [], []
    for i,e in enumerate(logE):
        mean_A.append(np.dot(frac_tot[i], np.log(A))/ np.sum(frac_tot[i]))
        XmaxLM.append(xmax_tls.getXmax(logE[i], mean_A[i], model ))

    meanVXmax = []
    for i,e in enumerate(logE):
        meanVXmax.append(np.sum(np.dot(xmax_tls.getVar_sh(logE[i], np.log(A), model), frac_tot[i]))/np.sum(frac_tot[i], axis=0))
    for i,e in enumerate(logE):
        V_A.append(np.dot(frac_tot[i], np.log(A)**2)/ np.sum(frac_tot[i], axis=0) - mean_A[i]**2)

    for i,e in enumerate(logE):
        RMSLM.append(xmax_tls.getXRMS (logE[i], meanVXmax[i], V_A[i], model))

    XmaxMean_E = interpolate.interp1d(logE, XmaxLM)
    RMS_E = interpolate.interp1d(logE, RMSLM)


    Dev = np.sum(( xmax["fXmax"][BinNumberXmax:] - XmaxMean_E(xmax["meanLgE"][BinNumberXmax:]) )**2/ xmax['statXmax'][BinNumberXmax:]**2)
    Dev += np.sum(( xmax['fRMS'][BinNumberXmax:] - RMS_E(xmax["meanLgE"][BinNumberXmax:]) )**2 / xmax['statRMS'][BinNumberXmax:]**2)

    return Dev


@jit
def Distr_Minimization(parms, args, w_zR):
    ''' Compute the Xmax distribution minimization

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    w_zR : `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of Spectrum
    '''
    t = args[0]
    A = args[1]
    Z = args[2]
    xmax = args[4]
    E_th = args[5]
    exp_distributions_x= args[6]
    exp_distributions_y= args[7]
    convoluted_gumbel = args[8]

    #Computation##################################
    nentries = xmax['nEntries']
    meanLgE = xmax['meanlgE']

    A_new, f_new = mass.get_fractions_distributions(t, parms, A, Z, w_zR,xmax)
    Dev = mass.compute_Distr_Deviance(A_new, f_new, meanLgE, exp_distributions_x, exp_distributions_y, convoluted_gumbel, E_th, nentries)

    return Dev


@jit
def Spectrum_Minimization(parms, args, w_zR):
    ''' Performing the minimization of the energy spectrum

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    w_zR : `float`
        Weights in

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of Spectrum
    '''
    t = args[0]
    A = args[1]
    Z = args[2]
    T_J = args[3]
    E_th = args[5]
    #Computation##################################
    models, sel_A = [], []
    for i, a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked, w_zR, Z[i])
        models.append([t[i].logE,parms[i]*je/(10**t[i].logE * (t[i].logE[1]-t[i].logE[0]) * np.log(10))])
        sel_A.append(a)
    #print("Elapsed time_1", end - start)
    spectrum_per_inj = []
    logE = []
    for i, m in enumerate(models):
        logE, je = m[0], m[1]
        spectrum_per_inj.append(je)
    total_spectrum = np.sum(np.array(spectrum_per_inj), axis=0)
    interpolate_model = interpolate.interp1d(logE, total_spectrum)

    BinNumber = np.asscalar(np.argwhere(T_J['logE'] == E_th))
    Dev = 0
    StartingFrom = BinNumber # Third point
    Errors = (T_J['J_up'][StartingFrom:]-T_J['J_low'][StartingFrom:])/2
    Dev = np.sum((T_J['J'][StartingFrom:]- interpolate_model(T_J['logE'][StartingFrom:]))**2/Errors**2) #

    return Dev


def Minimize_Spectrum(parms, args):
    ''' Computing the deviance of the energy spectrum

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function

    Returns
    -------
    Deviance: 'float'
        return the deviance due to the minimization of the energy spectrum
    '''
    logRcut = parms[-2]
    gamma = parms[-1]
    S_z = args[-1]

    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)

    Dev_Spectrum = Spectrum_Minimization(parms, args, w_zR)
    print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2))
    return Dev_Spectrum


def Minimize_Spectrum_And_Compo(parms, args, verbose = True):
    ''' Computing the deviance of the energy spectrum and lnA

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    verbose : `bool`
        Increse the verbosity of the function

    Returns
    -------
    Deviance: 'float'
        return the sum of the deviance of spectrum and composition
    '''
    logRcut = parms[-2]
    gamma = parms[-1]
    S_z = args[-1]
    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)
    start = time.time()
    Dev_fractions = Fractions_Minimization(parms, args, w_zR)
    end = time.time()
    #print("Elapsed time_C ", end - start)
    start = time.time()
    Dev_Spectrum = Spectrum_Minimization(parms, args, w_zR)
    end = time.time()
    #print("Elapsed time_S ", end - start)
    if verbose:
        print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2), " | Composition deviance: ", np.around(Dev_fractions, decimals=2))
    #exit()
    return Dev_fractions + Dev_Spectrum


def Minimize_Spectrum_And_Xmax(parms, args, verbose = True):
    ''' Computing the deviance of the energy spectrum and Xmax

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    verbose : `bool`
        Increse the verbosity of the function

    Returns
    -------
    Deviance: 'float'
        return the sum of the deviance of spectrum and Xmax
    '''
    logRcut = parms[-2]
    gamma = parms[-1]
    S_z = args[-1]
    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)
    #start = time.time()
    Dev_fractions = Xmax_Minimization(parms, args, w_zR)
    #end = time.time()
    #print("Elapsed time_C ", end - start)

    #start = time.time()
    Dev_Spectrum = Spectrum_Minimization(parms, args, w_zR)
    #end = time.time()
    #print("Elapsed time_S ", end - start)
    #exit()

    if verbose:
        print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2), " | Composition deviance: ", np.around(Dev_fractions, decimals=2))
    return Dev_fractions + Dev_Spectrum


def Minimize_Spectrum_And_Distribution(parms, args, verbose = True):
    ''' Computing the deviance of the energy spectrum and Xmax distributions

    Parameters
    ----------
    parms : `list`
        List of parameters to be fitted
    args : `list`
        List of arguments needed to the function
    verbose : `bool`
        Increse the verbosity of the function

    Returns
    -------
    Deviance: 'float'
        return the sum of the deviance of spectrum and Xmax distributions
    '''
    logRcut = parms[-2]
    gamma = parms[-1]
    S_z = args[-1]
    w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)/sp.dzdt(z)*S_z(z)

    Dev_distributions = Distr_Minimization(parms, args, w_zR)

    Dev_Spectrum = Spectrum_Minimization(parms, args, w_zR)

    if verbose:
        print("Spectrum deviance: ", np.around(Dev_Spectrum, decimals=2), " | Composition deviance: ", np.around(Dev_distributions, decimals=2))
    return Dev_distributions + Dev_Spectrum
