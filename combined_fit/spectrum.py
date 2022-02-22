import numpy as np
import os
from scipy import interpolate, integrate
from astropy.table import Table
import pathlib

from combined_fit import constant
from combined_fit import draw

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()
dzdt = lambda z: constant._H0*(1+z)*np.sqrt(constant._OmegaLambda + constant._OmegaM*(1.+z)**3)


def Plot_spectrum(t, frac, A, Z, w_zR, E_fit):
    ''' Calculate the expected spectrum, compute the deviance with the experimental one and
    Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    t : `tensor`
        tensor of extra-galactic propagation
    frac : `list`
        fractions at the source
    A,Z : `list`
        mass and charge of injected particles
    w_zR : `list`
        weights on z and R (redshift and rigidity)
    E_fit : `float`
        Energy bin from which the deviance is computed

    Returns
    -------
    None
    '''
    logE,expected_spectrum, spectrum_per_inj = Compute_expected_spectrum(t,frac,A,Z,w_zR) # compute the expected spectrum
    experimental_spectrum = load_Spectrum_Data() # load the experimental spectrum

    norm = Deviance_spectrum(logE,expected_spectrum, experimental_spectrum, E_fit) # find the normalization
    draw.Draw_spectrum(A,logE, expected_spectrum, spectrum_per_inj, norm,E_fit) # plot the spectra


def Compute_expected_spectrum(t, frac, A, Z, w_zR):
    ''' Compute the expected spectrum

    Parameters
    ----------
    t : `tensor`
        tensor of extra-galactic propagation
    frac : `list`
        fractions at the source
    A,Z : `list`
        mass and charge of injected particles
    w_zR : `list`
        weights on z and R (redshift and rigidity)

    Returns
    -------
    logE : `list`
        list of  energy bins as stored in the tensor
    total_spectrum : `list`
        total spectrum at the top of the atmosphere
    spectrum_per_inj : `list`
        Injected spectra (for each injected particle) at the top of the atmosphere
    '''
    models, sel_A = [], []
    total_spectrum = []
    spectrum_per_inj = []
    logE = []

    for i, a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked, w_zR, Z[i])
        models.append([t[i].logE , frac[i]*je/(10**t[i].logE * (t[i].logE[1]-t[i].logE[0]) * np.log(10))])
        sel_A.append(a)

    for i, m in enumerate(models):
        logE, je = m[0], m[1]
        total_spectrum.append(je)
        spectrum_per_inj.append(je)
    total_spectrum = np.sum(np.array(total_spectrum), axis=0)
    return logE,total_spectrum, spectrum_per_inj


def Deviance_spectrum(logE,expected_spectrum, experimental_spectrum, E_fit):
    ''' Compute the deviance between the expected and the experimental spectrum

    Parameters
    ----------
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum : `list`
        total expected spectrum at the top of the atmosphere
    experimental_spectrum: `Table`
        expected spepctrum (energy and flux)
    E_fit : `float`
        Energy bin from which the deviance is computed

    Returns
    -------
    norm : `float`
        normalization of the expected spectrum
    '''
    BinNumber = np.asscalar(np.argwhere(experimental_spectrum['logE'] == E_fit))
    #---------shift -----------#
    eneshift = 0
    dEnScale = 0.14
    shift = 1 + dEnScale * eneshift
    experimental_spectrum['logE'] = experimental_spectrum['logE'] +np.log10(shift)
    experimental_spectrum['J'] = experimental_spectrum['J'] * shift * shift
    experimental_spectrum['J_up'] = experimental_spectrum['J_up'] * shift * shift
    experimental_spectrum['J_low'] = experimental_spectrum['J_low'] * shift * shift
    #---------shift -----------#

    interpol_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])
    interpolate_model = interpolate.interp1d(logE, expected_spectrum)
    MaxE = np.max(experimental_spectrum['logE'])
    print("Normalization starting at logE=", experimental_spectrum['logE'][BinNumber])
    norm = integrate.quad(interpol_data, experimental_spectrum['logE'][BinNumber], MaxE)[0]/integrate.quad(interpolate_model, experimental_spectrum['logE'][BinNumber], MaxE)[0] # 1

    num = 0
    Dev = 0
    for i in range(BinNumber, len(experimental_spectrum['J'])):
        Errors = (experimental_spectrum['J_up'][i]-experimental_spectrum['J_low'][i])/2
        Dev += (experimental_spectrum['J'][i]- interpolate_model(experimental_spectrum['logE'][i])*norm )**2/Errors**2
        num = num +1
    print("Spectrum deviance, from logE=", experimental_spectrum['logE'][BinNumber], ": ", Dev, " (", num, ")")
    return norm


def load_Spectrum_Data():
    ''' Upload the experimental spectrum

    Parameters
    ----------
    None

    Returns
    -------
    T_J : `table`
       experimental spectrum as read in 'Data'
'''
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Spectra/SpectrumSD1500_PRD.dat')
    T_J = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)
    return T_J


def Spectrum_Energy(Z, logR, gamma, logRcut):
    ''' Build the expected spectrum starting from the parameters at the source

    initially flat between lRmin and lRmax

    Parameters
    ----------
    Z : `list`
        charge of injected particles
    logR : `float`
        log Rigidity for protons
    gamma: `float`
        spectral index of injected particles
    logRcut : `float`
        log Rigidity of the injected particles

    Returns
    -------
    weights : `list`
        weights in z and R
    '''
    E = Z*np.power(10, logR)
    Ecut = Z*np.power(10, logRcut)
    ind = np.where(E>Ecut)
    weights = np.zeros_like(logR)
    weights = np.power(E,-gamma+1)*(logR[1]-logR[0])*np.log(10)
    weights[ind] *= np.exp(1-E[ind]/Ecut)
    weights = weights/np.sum(weights*E, axis=0)
    weights *= 1/(4*np.pi)*constant._c
    weights *= constant._erg_to_EeV *1e18

    return weights
