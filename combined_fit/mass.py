import os
import sys
import numpy as np
from scipy import interpolate
from astropy.table import Table
import pathlib

from combined_fit import constant
from combined_fit import xmax_tools as xmax_tls
from combined_fit import xmax_distr
from combined_fit import draw

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

### whole functions ###
def Plot_fractions(t,frac, A, Z, w_zR,w_zR_p, E_fit, model="EPOS-LHC"):
    '''Compute the expected fractions, upload the experimental fractions, perform the deviance and plot the results

    Parameters
    -----------
    t : `tensor`
        upload tensor for the extra-galactic propagation
    frac : `list`
        fractions at the top of the atmosphere
    A,Z: `list`
        Mass and charge of the injected particles
    w_zR : `list`
        log Rigidity of the injected particles
    E_fit: `float`
        energy from which the deviance is computed
    model: `string`
        Hadronic Interaction model

    Returns
    -------
    None
    '''
    logE, mean_A, sig2A_E = expected_lnA(t,frac,A,Z,w_zR, w_zR_p) # expected lnA and sigma lnA
    Auger_lnA = load_lnA_Data(model) # experimental lnA for a given HIM
    compute_lnA_Deviance(logE, mean_A, sig2A_E, Auger_lnA, E_fit) # computation of deviance
    draw.draw_mass_fractions(logE, mean_A, sig2A_E, Auger_lnA, E_fit, model) # Plot

def Plot_Xmax(t,frac,A,Z,w_zR,w_zR_p, E_fit, model):
    '''Compute the expected xmax mean and sigma, upload the experimental results, compute the deviance and plot both of them

    Parameters
    -----------
    t : `tensor`
        upload tensor for the extra-galactic propagation
    frac : `list`
        fractions at the top of the atmosphere
    A,Z: `list`
        Mass and charge of the injected particles
    w_zR : `list`
        log Rigidity of the injected particles
    E_fit: `float`
        energy from which the deviance is computed
    model: `string`
        Hadronic Interaction model

    Returns
    -------
    None
        '''
    logE,Xmax, RMS = expected_Xmax_sigmaXmax(t,frac,A,Z,w_zR,w_zR_p, model)
    Experimental_Xmax = load_Xmax_data()
    compute_Xmax_Deviance(logE, Xmax, RMS, Experimental_Xmax, E_fit) # computation of deviance
    draw.draw_Xmax(logE,Xmax, RMS,Experimental_Xmax, E_fit, model)

def expected_lnA(t,frac,A,Z,w_zR,w_zR_p):
    '''Compute the expected fractions

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
    logE: `list`
        energy bins from the read tensor
    mean_A: `list`
        mean_A (for different lgE)
    V_A: `list`
        Variance of A (for different lgE)
    '''
    logE = t[0].logE

    A, frac_tot = get_fractions_p(t,frac,A,Z,w_zR,w_zR_p)
    place = np.argwhere(np.sum(frac_tot, axis=0) != 0)
    index = np.ndarray.item(place[0])

    frac_tot = frac_tot[:,index:]
    logE = logE[index:]

    mean_A = np.dot(np.log(A),frac_tot) / np.sum(frac_tot, axis=0)

    V_A = np.dot(np.log(A)**2,frac_tot) / np.sum(frac_tot, axis=0) - mean_A**2

    return logE, mean_A, V_A


def expected_Xmax_sigmaXmax(t,frac,A,Z,w_zR,w_zR_p, model):
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

    A, frac_tot = get_fractions_p(t,frac,A,Z,w_zR, w_zR_p)
    place = np.argwhere(np.sum(frac_tot, axis=0) != 0)
    index = np.ndarray.item(place[0])

    frac_tot = frac_tot[:,index:]
    logE = logE[index:]

    frac_tot = np.transpose(frac_tot)

    mean_A, Xmax, V_A, RMS = [], [], [], []
    for i,e in enumerate(logE):
        frac_tot[i] = frac_tot[i]/np.sum(frac_tot[i])
        mean_A.append(np.dot(frac_tot[i], np.log(A))/ np.sum(frac_tot[i]))
        Xmax.append(xmax_tls.getXmax(logE[i], mean_A[i], model))

    meanVXmax = []
    for i,e in enumerate(logE):
        meanVXmax.append(np.sum(np.dot(xmax_tls.getVar_sh(logE[i], np.log(A), model), frac_tot[i]))/np.sum(frac_tot[i], axis=0))
    for i,e in  enumerate(logE):
        V_A.append(np.dot(frac_tot[i], np.log(A)**2)/ np.sum(frac_tot[i], axis=0) - mean_A[i]**2)

    for i,e in enumerate(logE):
        RMS.append(xmax_tls.getXRMS (logE[i], meanVXmax[i], V_A[i], model))

    return logE,Xmax, RMS

#### computation of deviance ###

def compute_lnA_Deviance(logE, mean_A, V_A, t_lnA, E_fit):
    '''Perform the deviance between the expected and the experimental fractions

    Parameters
    ----------
    logE : `list`
        energy bins from the read tensor
    mean_A : `list`
        mean_A (for different lgE)
    V_A: `list`
        Variance of A (for different lgE)
    t_lnA : `Table`
        lnA as read in data folder
    E_fit : `float`
        energy from which the deviance is computed

    Returns
    -------
    None
    '''
    StartingFrom = np.ndarray.item(np.argwhere(t_lnA['logE'] == E_fit))

    avA_E = interpolate.interp1d(logE, mean_A)
    sig2A_E = interpolate.interp1d(logE, V_A)


    Dev = np.sum(( t_lnA['mean_lnA'][StartingFrom:] - avA_E(t_lnA['logE'][StartingFrom:]) )**2/ t_lnA['mean_Stat'][StartingFrom:]**2)
    Dev += np.sum(( t_lnA['Var_lnA'][StartingFrom:] - sig2A_E(t_lnA['logE'][StartingFrom:]) )**2 / ((t_lnA['Var_StatUp'][StartingFrom:]+t_lnA['Var_StatLow'][StartingFrom:])/2)**2)
    print("Composition deviance, from logE=", t_lnA['logE'][StartingFrom],": ", Dev, " (", len(t_lnA['mean_lnA']) + len(t_lnA['Var_lnA']) - StartingFrom, ")")

def compute_Xmax_Deviance(logE, Xmax, RMS, experimental_xmax, E_fit):
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
    Returns
    -------
    None
    '''
    XmaxMean_E = interpolate.interp1d(logE, Xmax)
    RMS_E = interpolate.interp1d(logE, RMS)

    BinNumberXmax = np.ndarray.item(np.argwhere(np.around(experimental_xmax['meanLgE'], decimals=2)== E_fit))

    Dev = np.sum(( experimental_xmax['fXmax'][BinNumberXmax:] - XmaxMean_E(experimental_xmax['meanLgE'][BinNumberXmax:]) )**2/ experimental_xmax['statXmax'][BinNumberXmax:]**2)
    Dev += np.sum(( experimental_xmax['fRMS'][BinNumberXmax:] - RMS_E(experimental_xmax['meanLgE'][BinNumberXmax:]) )**2 / experimental_xmax['statRMS'][BinNumberXmax:]**2)

    print("Composition deviance, from logE=", experimental_xmax['meanLgE'][BinNumberXmax],": ", Dev , "(",len(experimental_xmax['fXmax']) + len(experimental_xmax['fRMS']) - 2*BinNumberXmax, ")")

def compute_Distr_Deviance(A, frac,meanLgE, exp_distributions_x, exp_distributions_y, convoluted_gumbel, E_th, nEntries):
    '''Compute the deviance for Xmax distributions

    Parameters
    ----------
    A : `list`
        mass at the top of the atmosphere
    frac : `list`
        mass fraction at the top of the atmosphere
    meanLgE: `list`
        energy bins from the read xmax distributions
    exp_distributions_x : `list`
        experimental Xmax distribution (x-axis)
    exp_distributions_y : `list`
        experimental Xmax distribution (y-axis)
    convoluted_gumbel : `ndarray`
        convoluted gumbel functions for each mass and energy
    E_th : `float`
        energy from which the deviance is computed
    nEntries : `list`
        number of entries of each Xmax distribution

    Returns
    -------
    None
    '''
    Dev = 0
    BinNumberXmax = np.ndarray.item(np.argwhere(np.around(meanLgE, decimals=2)== E_th))

    for i in range(BinNumberXmax, len(meanLgE)):
        sum = np.zeros((len(A),len(exp_distributions_x[i])))
        frac[i] = frac[i]/np.sum (frac[i])

        for j in range(len(A)):
            sum[j] = np.multiply(convoluted_gumbel[i][j], frac[i][j])

        final = np.sum(sum, axis = 0)/np.sum(frac[i])
        data = exp_distributions_y[i]

        Dev += xmax_distr.deviance_Xmax_distr(data,final, nEntries[i])

    return Dev
 ### useful functions ###
def get_fractions_distributions(t,frac,A,Z,w_zR, xmax):
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
    A, frac_tot = get_fractions(t,frac,A,Z,w_zR)

    frac_tot = np.transpose(frac_tot)

    start = np.ndarray.item(np.argwhere(t[0].logE == np.around(xmax['meanlgE'][0], decimals = 2)))

    frac_def = np.zeros((len(xmax['maxlgE']),len(A)))

    for j,a in enumerate(xmax['maxlgE']):
        index = int(round((xmax['maxlgE'][j]- xmax['minlgE'][j])/0.1))
        for k in range (index):
            frac_def[j] += frac_tot[j+k+start]



    A_new , frac_new = reduced_fractions(A, frac_def,np.size(xmax['meanlgE']))

    return A_new, frac_new

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

def get_fractions_p(t,frac,A,Z,w_zR, w_zR_p):
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
                #if (i == 0):
                    #je = t[i].J_E(t[i].tensor_stacked, w_zR_p, Z[i])
                #else:
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

    A = np.zeros(56)
    frac = np.zeros((size,56))
    for j in range (size):
        for i,a in enumerate(A):
            #index = int(A_old[i])
            #A[index-1] = index
            frac[j][i] = np.sum(frac_old[j][np.where(A_old ==i)])

    return A, frac

########### Loading functions ##############

def load_lnA_Data(model="EPOS-LHC"):
    '''Upload the experimental data (lnA) for a given Hadronic Interaction model

    Parameters
    ----------
    model: `string`
        hadronic interaction model

    Returns
    -------
    Table: `read`
        experimental data
    '''
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/lnA/')
    if model=="EPOS-LHC":
        return Table.read(filename+'lnA_EPOS-LHC', format='ascii.basic', delimiter=" ", guess=False)
    elif model=="Sibyll":
        return Table.read(filename+'lnA_Sibyll2.3c', format='ascii.basic', delimiter=" ", guess=False)
    else:
        print("Wrong model in load_lnA_Data")
        sys.exit(0)

def load_Xmax_data():
    '''Upload the experimental data (Xmax) for a given Hadronic Interaction model

    Parameters
    -------

    Returns
    -------
    Table: `read`
        experimental data
    '''
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Moments/Xmax_moments_icrc17_v2.txt')
    return Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)



def get_Energies_2017():
    '''Get the minLgE and the maxLgE for the distribution computed in ICRC2017

    Parameters
    ----------
    None

    Returns
    -------
    minLgE : `list`
        lower edge of energy bins
    maxLgE: `list`
        upper edge of energy bins
    '''
    totBins = 23
    lgEFirst = 17.2
    dlgE = 0.1
    minLgE, maxLgE = [], []

    for iEnergy in range(totBins+1):
        minLgE.append(lgEFirst + iEnergy * dlgE)
        if iEnergy < totBins:
            maxLgE.append(lgEFirst + (iEnergy+1) * dlgE)
        else:
            maxLgE.append (21.)
    return minLgE, maxLgE
