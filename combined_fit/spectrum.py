import os
import pathlib
import copy
import numpy as np
from scipy import interpolate, integrate
from astropy.table import Table

from combined_fit import constant
from combined_fit import draw
from combined_fit import mass
from combined_fit import xmax_tools as xmax_tls

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

def Plot_spectrum(t, frac, A, Z, w_zR, w_zR_p, E_fit, hadr_model, isE3dJdE= True, isRenormalized = False, ext_save=""):
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
    hadr_model : `string`
        hadronic interaction model
    ext_save: `string`
        extension for the saved file
               
    Returns
    -------
    None
    '''
    logE,expected_spectrum, spectrum_per_inj, spectrum_det = Compute_expected_spectrum(t, frac, A, Z, w_zR, w_zR_p) # compute the expected spectrum
    experimental_spectrum = load_Spectrum_Data() # load the experimental spectrum
    experimental_proton = load_ProtonSpectrum_Data(hadr_model) # load the proton spectrum

    norm, dev = Deviance_spectrum_proton_p(logE, expected_spectrum, experimental_spectrum, spectrum_det, experimental_proton, E_fit, isRenormalized = isRenormalized) # if isRenormalized = True, the overall normalisation minimizing the spectral deviance for fixed fractions can be determined (norm = 1 by default, so that input values keep real units)
    if isRenormalized: print("Normalization factor:",norm)
    
    draw.Draw_spectrum(		A, logE, expected_spectrum, spectrum_per_inj,	norm, E_fit, hadr_model, isInjected  = True, isE3dJdE= isE3dJdE, isSysDisplayed=False,  saveTitlePlot = "uhecr_spectrum_inj_"+ext_save) # plot the spectra as a function of injected mass
    draw.Draw_spectrum(		A, logE, expected_spectrum, spectrum_det, 		norm, E_fit, hadr_model, Dev = dev, isInjected  = False, isE3dJdE= isE3dJdE, saveTitlePlot = "uhecr_spectrum_det_"+ext_save) # plot the spectra as a function of detected mass    

def Compute_expected_spectrum(t, frac, A, Z, w_zR, w_zR_p):
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
    #Load spectrum and fraction per injected mass    
    spectrum_per_inj = []
    for i, a in enumerate(A):
        if (i ==0):
            je = t[i].J_E(t[i].tensor_stacked, w_zR_p, Z[i])
        else:
            je = t[i].J_E(t[i].tensor_stacked, w_zR, Z[i])
        logE = t[i].logE
        spectrum_per_inj.append(frac[i]*je/(10**t[i].logE * (t[i].logE[1]-t[i].logE[0]) * np.log(10)))

    #Compute total detected spectrum        
    total_spectrum = np.sum(np.array(spectrum_per_inj), axis=0)

    #Compute the 56 spectra
    A_new, frac_new = mass.get_fractions_p(t, frac, A, Z, w_zR, w_zR_p)
    frac_new = frac_new/np.sum(frac_new, axis=0)
    A_red, frac_red = mass.reduced_fractions(A_new, np.transpose(frac_new), np.size(logE))
    det_spectra = np.transpose(frac_red)*total_spectrum

    return logE, total_spectrum, spectrum_per_inj, det_spectra
    
def Compute_integral_spectrum(t, frac, A, Z, w_zR, w_zR_p):
    ''' Compute the integral spectrum above the minimum energy of the tensors

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
    logEth : `float`
        threshold energy in log(eV)
    total_flux : `list`
        cosmic-ray flux in 1/(km2 sr yr)
    '''
    #Load spectrum and fraction per injected mass    
    spectrum_per_inj = []
    dlogE, ln10 = t[0].logE[1]-t[0].logE[0], np.log(10)
    for i, a in enumerate(A):
        if (i ==0):
            je = t[i].	J_E(t[i].tensor_stacked, w_zR_p, Z[i])
        else:
            je = t[i].J_E(t[i].tensor_stacked, w_zR, Z[i])
        spectrum_per_inj.append(frac[i]*je/(dlogE * ln10))

    return t[0].logE[0]-0.5*dlogE, np.sum(np.array(spectrum_per_inj))*ln10*dlogE

def Compute_single_integrals(t, frac, A, Z, w_R, w_R_p):
    ''' Compute the integral spectrum for a single galaxy above the minimum energy of the tensors

    Parameters
    ----------
    t : `tensor`
        tensor of extra-galactic propagation
    frac : `list`
        fractions at the source
    A,Z : `list`
        mass and charge of injected particles
    w_R : `list`
        weights vs rigidity for nuclei
    w_R_p : `list`
        weights vs rigidity for protons

    Returns
    -------
    logEth : `float`
        threshold energy in log(eV)
    z : `float`
        threshold energy in log(eV)
    total_flux : `list`
        cosmic-ray flux in arbitrary units x 1/(km2 sr yr)
    cum_weighted_R : `float`
        cumulated weighted rigidity to compute mean observed rigidity 
    cum_weighs : `float`
        cumulated weights to compute mean observed rigidity                 
    '''
    #Load spectrum and fraction per injected mass    
    spectrum_per_inj = []
    spectrum_per_Zdet, Rspectrum_per_Zdet = [], []
    dlogE, ln10 = t[0].logE[1]-t[0].logE[0], np.log(10)
    for i, a in enumerate(A):
        if (i ==0):
            je = t[i].j_zE(t[i].tensor_stacked, w_R_p, Z[i])
            je_Zobs = t[i].j_zE(t[i].tensor_stacked_Z, w_R_p, Z[i])
        else:
            je = t[i].j_zE(t[i].tensor_stacked, w_R, Z[i])
            je_Zobs = t[i].j_zE(t[i].tensor_stacked_Z, w_R_p, Z[i]) 
        #print(je_Zobs.shape, len(t[i].Z))
        spectrum_per_inj.append(frac[i]*je/(dlogE * ln10))
        for j, Zdet in enumerate(t[i].Z):
            w = frac[i]*je_Zobs[j]/(dlogE * ln10)
            rig = 10**t[i].logE/Zdet
            rw = w*rig#*w/
            spectrum_per_Zdet.append(w)
            Rspectrum_per_Zdet.append(rw)

    cum_R = np.sum(Rspectrum_per_Zdet)
    cum_w = np.sum(spectrum_per_Zdet)
    logEth= t[0].logE[0]-0.5*dlogE
    total_flux = np.sum(np.array(spectrum_per_inj), axis = (0,2) )*ln10*dlogE
    
    return logEth, t[0].z, total_flux, cum_R, cum_w

def Deviance_spectrum_proton_p(logE, expected_spectrum, experimental_spectrum, det_spectra, experimental_proton, E_fit, isRenormalized = False, verbose = False):
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

    #---------shift -----------#
    eneshift = 0
    dEnScale = 0.14
    shift = 1 + dEnScale * eneshift
    experimental_spectrum['logE'] = experimental_spectrum['logE'] +np.log10(shift)
    experimental_spectrum['J'] = experimental_spectrum['J'] * shift * shift
    experimental_spectrum['J_up'] = experimental_spectrum['J_up'] * shift * shift
    experimental_spectrum['J_low'] = experimental_spectrum['J_low'] * shift * shift
    #---------shift -----------#

    #Normalization for all-particle and proton spectra
    StartingFrom = np.ndarray.item((np.argwhere(experimental_spectrum['logE'] == E_fit)))
    MaxE = np.max(experimental_spectrum['logE'])
    interpol_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])    
    interpolate_model = interpolate.interp1d(logE, expected_spectrum, fill_value="extrapolate")
    if not isRenormalized: norm = 1
    else: norm = integrate.quad(interpol_data, experimental_spectrum['logE'][StartingFrom], MaxE)[0]/integrate.quad(interpolate_model, experimental_spectrum['logE'][StartingFrom], MaxE)[0] #

    #Residuals for all-particle spectrum
    res = experimental_spectrum['J'][StartingFrom:]- norm*interpolate_model(experimental_spectrum['logE'][StartingFrom:])
    Sigma = (res>0)*experimental_spectrum['J_low'][StartingFrom:]+(res<=0)*experimental_spectrum['J_up'][StartingFrom:]
    dev_all = np.sum((res/Sigma)**2)

    #Residuals for proton spectrum
    StartingFrom_p = 0 #
    MaxBinNumber_p =  np.argmax(experimental_proton['logE'] > E_fit-0.01)    
    interpolate_proton_model = interpolate.interp1d(logE, det_spectra[1], fill_value="extrapolate")    
    res_p = experimental_proton['J'][StartingFrom_p:MaxBinNumber_p]- norm*interpolate_proton_model(experimental_proton['logE'][StartingFrom_p:MaxBinNumber_p])
    Sigma_p = (res_p>0)*experimental_proton['J_low'][StartingFrom_p:MaxBinNumber_p]+(res_p<=0)*experimental_proton['J_up'][StartingFrom_p:MaxBinNumber_p]
    dev_p = np.sum((res_p/Sigma_p)**2)
    
#    print(Table([experimental_proton['logE'][StartingFrom_p:MaxBinNumber_p], experimental_proton['J'][StartingFrom_p:MaxBinNumber_p], interpolate_model(experimental_proton['logE'][StartingFrom_p:MaxBinNumber_p]), Sigma_p, res_p/Sigma_p], names=['logE','data','model','sigma','res/sigma']))
       
    if verbose:
        print("Spectrum deviance w/o protons, from logE=", experimental_spectrum['logE'][StartingFrom], ": ", dev_all, " (", len(res[StartingFrom:]), ") ", norm)
        print("Spectrum deviance w/ protons, from logE=", experimental_proton['lgE'][StartingFrom_p], ": ", dev_p, " (", len(res_p[StartingFrom_p:]), ")")

    return norm, dev_all+dev_p

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
    weights = np.power(E,-gamma+1)*(logR[1]-logR[0])*np.log(10)

    Ecut = Z*np.power(10, float(logRcut))
    ind = np.where(E>Ecut)
    weights[ind] *= np.exp(1-E[ind]/Ecut)

    return weights/np.sum(weights*E, axis=0)

def weight_tensor(S_z, gamma, logRcut):
    ''' Return the tensor weight account for evolution and spectral shape

    Parameters
    ----------
    S_z : `1D function`
        evolution of the source density
    gamma: `float`
        spectral index of injected particles
    logRcut : `float`
        log Rigidity of the injected particles

    Returns
    -------
    w_zR : `3D function`
       function to weight the tensor
'''
    unit_fact = constant._erg_to_eV*constant._c_ov_4pi/(constant._Mpc_2_km)**3#"Etot unit"x[km/(s.sr)]x"dt[s]"x "Tracer_unit per [km3]"
    
    w_R = lambda ZA, logR: Spectrum_Energy(ZA, logR, gamma, logRcut)
    w_zR = lambda ZA, z, logR: w_R(ZA, logR)*constant._dtdz_s(z)*S_z(z)*unit_fact

    return w_zR
        
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
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Spectrum/spectrum_combined.txt')

    return Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)

def load_ProtonSpectrum_Data(hadr_model):
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
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/Composition/composition_fractions_icrc17.txt')
    t_frac = Table.read(filename, format='ascii.basic', delimiter="\t", guess=False)

    #Remove HEAT-based data below 17.8
    t_frac = t_frac[t_frac['meanLgE']>17.8]

    #Build proton spectrum from given HIM
    if hadr_model not in xmax_tls.HIM_list :
        print("Hadronic interaction model not valid! ", hadr_model)
        sys.exit(0)
    if hadr_model == "EPOS-LHC": ext = '_EPOS'
    elif hadr_model == "QGSJET-II04": ext = '_QGS'
    else: ext = '_SIB'

    keys_in = ['meanLgE', 'p'+ext, 'p_err_low'+ext, 'p_err_up'+ext, 'p_sys_low'+ext, 'p_sys_up'+ext]
    keys_out = ['logE', 'J', 'J_low', 'J_up', 'J_sys_low', 'J_sys_up']
    f = copy.copy(t_frac['p'+ext])
    t_proton = t_frac[keys_in]
    for i, k in enumerate(keys_in): t_proton.rename_column(k, keys_out[i])
        
    #Load all-particle spectrum
    experimental_spectrum = load_Spectrum_Data()
    interp_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])  
    interp_err = {  'J_low': interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J_low']),
                    'J_up': interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J_up']),
                    'J_sys_low': interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J_sys_low']),   
                    'J_sys_up': interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J_sys_up'])}  
    
    #Multiply fraction by spectrum and compute uncertainties
    spectrum = interp_data(t_proton['logE'])
    for k in keys_out[1:]: t_proton[k]*=spectrum
    for k in keys_out[2:]: t_proton[k] = np.sqrt(t_proton[k]**2 + f**2*interp_err[k](t_proton['logE'])**2)
    
    return t_proton

