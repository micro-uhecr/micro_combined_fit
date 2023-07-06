import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from combined_fit import spectrum as sp
from combined_fit import constant
from combined_fit import mass as M
from combined_fit import xmax_tools as xmax_tls
from combined_fit import xmax_distr
#from combined_fit import gumbel
lEmin,lEmax = 16.25,20.5

def draw_mass_fractions(logE, mean_A, V_A, t_lnA, E_fit, model):
    '''plot the expected and the experimental mass fractions

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
    model: `string`
        hadronic interaction model

    Returns
    -------
    None
    '''

    avA_E = interpolate.interp1d(logE, mean_A)
    sig2A_E = interpolate.interp1d(logE, V_A)
    BinNumber = np.ndarray.item(np.argwhere(logE == E_fit))
    #Show <ln A> et V(ln A)
    fig = plt.subplots(figsize=(10, 4), nrows=1, ncols = 2)

    plt.tight_layout(pad=3, w_pad=2, h_pad=2)
    plt.subplots_adjust(bottom = 0.16,top = 0.9)
    lEmin,lEmax,dlogE = 18.7,19.8, 0.01

    plt.subplot(121)
    plt.xlabel(r'Energy, log$_{10} E$ [eV]')
    plt.ylabel(r'<ln A>')
    plt.xlim(lEmin,lEmax)
    plt.ylim(1.,3.5)
    plt.errorbar(t_lnA['logE']+0.5*dlogE, t_lnA['mean_lnA'], yerr=t_lnA['mean_Stat'], fmt='o',  mfc='none',label = model)
    plt.plot(logE, avA_E(logE), label="low-E extrapolation", color='tab:gray', linestyle='--')
    plt.plot(logE[BinNumber:], avA_E(logE)[BinNumber:], label="best-fit model", color='tab:gray')

    plt.legend()

    plt.subplot(122)
    plt.xlabel(r'Energy, log$_{10} E$ [eV]')
    plt.ylabel(r'$\sigma^{2}$ ln A')
    plt.xlim(lEmin,lEmax)
    plt.ylim(-1.0,2.0)

    plt.errorbar(t_lnA['logE']+0.5*dlogE, t_lnA['Var_lnA'], yerr=[t_lnA['Var_StatLow'],t_lnA['Var_StatUp'] ], fmt='o',  mfc='none',label = model)
    plt.plot(logE, sig2A_E(logE), label="low-E extrapolation", color='tab:gray', linestyle='--')
    plt.plot(logE[BinNumber:], sig2A_E(logE)[BinNumber:], label="best-fit model", color='tab:gray')


### plot xmax distribution ###
def Plot_Xmax_distribution(Tensor,frac,A,Z,w_zR,E_th,xmax, model, exp_distributions_x,exp_distributions_y, convoluted_gumbel):
    '''Plot the Xmax distributions

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
    E_th : `float`
        energy from which the deviance is computed
    xmax: `Table`
        experimental xmax data (energy)
    model: `string`
        hadronic interaction model
    exp_distributions_x : `list`
        experimental distributions (x axis)
    exp_distributions_y : `float`
        experimental distributions (y axis)
    convoluted_gumbel : `ndarray`
        convoluted gumbel for each energy and mass

    Returns
    -------
    None
    '''


    A_new, f_new = M.get_fractions_distributions(Tensor, frac, A, Z, w_zR,xmax)

    fig2 = plt.figure(figsize=(16,6))

    group = Draw_Xmax_group(A,A_new,f_new,xmax['meanlgE'],exp_distributions_x, convoluted_gumbel)
    Dev = 0
    points = 0
    dx = 20
    for i in range(len(xmax['meanlgE'])):

        sum = np.zeros((len(A_new),len(exp_distributions_x[i])))

        f_new[i] = f_new[i]/np.sum (f_new[i])
        for j in range(len(A_new)):
            sum[j] = np.multiply(convoluted_gumbel[i][j], f_new[i][j])

        final = np.sum(sum, axis = 0)/np.sum(f_new[i])

        data = exp_distributions_y[i]
        idk = data > 0

        if xmax['maxlgE'][i] > E_th: points +=(np.sum(np.multiply(idk,1)))
        Dev = xmax_distr.deviance_Xmax_distr(data,final, xmax['nEntries'][i])
        #print(xmax['meanlgE'][i], " ", Dev, " ", points)

        x = np.arange(0,2000,dx)
        min = int((np.min(exp_distributions_x[i])-(dx/2))/dx)
        max = int((np.max(exp_distributions_x[i])+(dx/2))/dx)
        x = x[min:max]
        err = np.zeros(len(data))
        for j in range (len(data)):
            if data[j] > 0:
                err[j] = np.sqrt(data[j])
        err = err/np.sum(data)
        data = data/np.sum(data)
        if i >= 2:

            ax2 = fig2.add_subplot(4, 4, i-1)
            plt.errorbar(x, data, fmt='o', color = 'white', label = 'log(E/eV) = '+str(np.around(xmax['meanlgE'][i], decimals= 2)))
            plt.errorbar(x, data, fmt='o', color = 'black',   mfc='w', yerr = err)
            plt.plot(x, final, linewidth=2, linestyle='--', color ='brown')
            plt.xlim(600,1000)
            if i == 6:
                plt.ylabel(r' frequency')
            if i == 15:
                plt.xlabel(r'<$X_{\mathrm{max}}>  [\mathrm { g \ cm^{-2}}]$')

            plt.legend(loc="upper right")

            [plt.plot(x, group[i][h], linewidth=2, linestyle='-', color =constant.colors[h]) for h in range(len(A))]

def Draw_spectrum(A,logE, expected_spectrum, spectrum_per_inj, norm,E_fit):
    ''' Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    A : `list`
        mass of injected particles
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum: `list`
        total expected spectrum at the top of the atmosphere
    spectrum_per_inj: `list`
        total expected spectrum at the top of the atmosphere
    norm : `float`
        normalization of the expected spectrum
    E_fit : `float`
        Energy bin from which the deviance is computed

    Returns
    -------
    None
        '''
    power_repr = 3
    experimental_spectrum = sp.load_Spectrum_Data()
    exp_proton = sp.load_Spectrum_Proton()
    fig = plt.subplots(figsize=(8, 4), nrows=1, ncols = 1)

    plt.tight_layout(pad=3, w_pad=2, h_pad=2)
    plt.subplots_adjust(bottom = 0.16,top = 0.9)

    plt.subplot(111)
    plt.xlabel(r'Energy, log$_{10} E$ [eV]')
    plt.ylabel(r'$E^3 J(E)$ [eV$^{2}\,$km$^{-2}\,$yr$^{-1}\,$sr$^{-1}\,$]')
    plt.xlim(17.3,lEmax)
    plt.ylim(1e36,1e38)
    plt.yscale('log')

    Espec = np.power(10,experimental_spectrum['logE'])#eV
    Data_spectrum = experimental_spectrum['J']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_up = experimental_spectrum['J_up']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_low = experimental_spectrum['J_low']#/Espec*(1e3)**2*365.25*24*3600
    interpol_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])

    Espec_prot = np.power(10,exp_proton['lgE'])#eV
    proton_spectrum = exp_proton['Frac']*interpol_data(exp_proton['lgE'])
    proton_error = exp_proton['Err']*interpol_data(exp_proton['lgE'])
    plt_proton_spectrum = Espec_prot**power_repr*proton_spectrum
    plt_Data_Err_proton = Espec_prot**power_repr*proton_error
    BinNumber_p =  np.ndarray.item(np.argwhere(exp_proton['lgE'] == E_fit))


    plt_Data_spectrum = Espec**power_repr*Data_spectrum
    plt_Data_Err_up = Espec**power_repr*Data_Err_up#-plt_Data_spectrum
    #plt_Data_Err_low = plt_Data_spectrum-Espec**power_repr*Data_Err_low
    plt_Data_Err_low = Espec**power_repr*Data_Err_low
    plt_Data_Err = [plt_Data_Err_low, plt_Data_Err_up]
    BinNumber =  np.ndarray.item(np.argwhere(logE == E_fit))
    #print(plt_Data_Err_up, " ", plt_Data_Err_low)

    plt.errorbar(experimental_spectrum['logE'], plt_Data_spectrum, yerr=plt_Data_Err, fmt='o',  mfc='none', color='k', label = "Auger")
    plt.errorbar(exp_proton['lgE'][:BinNumber_p], plt_proton_spectrum[:BinNumber_p], yerr=plt_Data_Err_proton[:BinNumber_p], fmt='o',  mfc='none', color='tab:red', label = "protons")

    for i,e in enumerate(spectrum_per_inj):
        e = np.power(10.,logE)
        plt.plot(logE, e**power_repr*norm*spectrum_per_inj[i], label = 'A = '+str(A[i]), color=constant.colors[i])
    e = np.power(10.,logE)
    plt.plot(logE, e**power_repr*norm*expected_spectrum, label="low-E exptrapolation", color='tab:brown', linestyle='--')
    plt.plot(logE[BinNumber:], e[BinNumber:]**power_repr*norm*expected_spectrum[BinNumber:], label="best-fit model", color='tab:brown')

    #plt.ylim(1E23,3E24)
    plt.yscale('log')
    plt.legend(fontsize=12)

def Draw_spectrum_inj(A,logE, expected_spectrum, spectrum_per_inj, norm,E_fit):
    ''' Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    A : `list`
        mass of injected particles
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum: `list`
        total expected spectrum at the top of the atmosphere
    spectrum_per_inj: `list`
        total expected spectrum at the top of the atmosphere
    norm : `float`
        normalization of the expected spectrum
    E_fit : `float`
        Energy bin from which the deviance is computed

    Returns
    -------
    None
        '''
    power_repr = 0
    experimental_spectrum = sp.load_Spectrum_Data()
    exp_proton = sp.load_Spectrum_Proton()
    fig = plt.subplots(figsize=(8, 4), nrows=1, ncols = 1)

    plt.tight_layout(pad=3, w_pad=2, h_pad=2)
    plt.subplots_adjust(bottom = 0.16,top = 0.9)

    plt.subplot(111)
    plt.xlabel(r'Energy, log$_{10} E$ [eV]')
    plt.ylabel(r'$E^3 J(E)$ [eV$^{2}\,$km$^{-2}\,$yr$^{-1}\,$sr$^{-1}\,$]')
    plt.xlim(17.3,lEmax)
    plt.yscale('log')

    Espec = np.power(10,experimental_spectrum['logE'])#eV
    Data_spectrum = experimental_spectrum['J']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_up = experimental_spectrum['J_up']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_low = experimental_spectrum['J_low']#/Espec*(1e3)**2*365.25*24*3600
    interpol_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])

    Espec_prot = np.power(10,exp_proton['lgE'])#eV
    proton_spectrum = exp_proton['Frac']*interpol_data(exp_proton['lgE'])
    proton_error = exp_proton['Err']*interpol_data(exp_proton['lgE'])
    plt_proton_spectrum = Espec_prot**power_repr*proton_spectrum
    plt_Data_Err_proton = Espec_prot**power_repr*proton_error
    BinNumber_p =  np.ndarray.item(np.argwhere(exp_proton['lgE'] == E_fit))


    plt_Data_spectrum = Espec**power_repr*Data_spectrum
    plt_Data_Err_up = Espec**power_repr*Data_Err_up#-plt_Data_spectrum
    #plt_Data_Err_low = plt_Data_spectrum-Espec**power_repr*Data_Err_low
    plt_Data_Err_low = Espec**power_repr*Data_Err_low
    plt_Data_Err = [plt_Data_Err_low, plt_Data_Err_up]
    BinNumber =  np.ndarray.item(np.argwhere(logE == E_fit))
    #print(plt_Data_Err_up, " ", plt_Data_Err_low)


    for i,e in enumerate(spectrum_per_inj):
        e = np.power(10.,logE)
        plt.plot(logE, e**power_repr*norm*spectrum_per_inj[i], label = 'A = '+str(A[i]), color=constant.colors[i])
    e = np.power(10.,logE)
    plt.plot(logE, e**power_repr*norm*expected_spectrum, label="low-E exptrapolation", color='tab:brown', linestyle='--')
    plt.plot(logE[BinNumber:], e[BinNumber:]**power_repr*norm*expected_spectrum[BinNumber:], label="best-fit model", color='tab:brown')

    #plt.ylim(1E23,3E24)
    plt.yscale('log')
    plt.legend(fontsize=12)

def Draw_spectrum_det(A,logE, expected_spectrum, spectrum_det, norm,E_fit):
    ''' Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    A : `list`
        mass of injected particles
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum: `list`
        total expected spectrum at the top of the atmosphere
    spectrum_per_inj: `list`
        total expected spectrum at the top of the atmosphere
    norm : `float`
        normalization of the expected spectrum
    E_fit : `float`
        Energy bin from which the deviance is computed

    Returns
    -------
    None
        '''
    power_repr = 3
    experimental_spectrum = sp.load_Spectrum_Data()
    exp_proton = sp.load_Spectrum_Proton()
    fig = plt.subplots(figsize=(8, 4), nrows=1, ncols = 1)

    plt.tight_layout(pad=3, w_pad=2, h_pad=2)
    plt.subplots_adjust(bottom = 0.16,top = 0.9)

    plt.subplot(111)
    plt.xlabel(r'Energy, log$_{10} E$ [eV]')
    plt.ylabel(r'$E^3 J(E)$ [eV$^{2}\,$km$^{-2}\,$yr$^{-1}\,$sr$^{-1}\,$]')
    plt.xlim(17.3,lEmax)
    plt.ylim(1e36,1e38)
    plt.yscale('log')

    Espec = np.power(10,experimental_spectrum['logE'])#eV
    Data_spectrum = experimental_spectrum['J']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_up = experimental_spectrum['J_up']#/Espec*(1e3)**2*365.25*24*3600
    Data_Err_low = experimental_spectrum['J_low']#/Espec*(1e3)**2*365.25*24*3600
    interpol_data = interpolate.interp1d(experimental_spectrum['logE'], experimental_spectrum['J'])

    Espec_prot = np.power(10,exp_proton['lgE'])#eV
    proton_spectrum = exp_proton['Frac']*interpol_data(exp_proton['lgE'])
    proton_error = exp_proton['Err']*interpol_data(exp_proton['lgE'])
    #print(proton_spectrum, " ", proton_error)
    plt_proton_spectrum = Espec_prot**power_repr*proton_spectrum
    plt_Data_Err_proton = Espec_prot**power_repr*proton_error
    BinNumber_p =  np.ndarray.item(np.argwhere(exp_proton['lgE'] == E_fit))


    plt_Data_spectrum = Espec**power_repr*Data_spectrum
    plt_Data_Err_up = Espec**power_repr*Data_Err_up#-plt_Data_spectrum
    #plt_Data_Err_low = plt_Data_spectrum-Espec**power_repr*Data_Err_low
    plt_Data_Err_low = Espec**power_repr*Data_Err_low
    plt_Data_Err = [plt_Data_Err_low, plt_Data_Err_up]
    BinNumber =  np.ndarray.item(np.argwhere(logE == E_fit))
    #print(plt_Data_Err_up, " ", plt_Data_Err_low)
    det_spectra_fin = []
    det_spectra_fin.append(spectrum_det[1])
    det_spectra_fin.append(np.sum(spectrum_det[2:4], axis =0))
    det_spectra_fin.append(np.sum(spectrum_det[5:22], axis =0))
    det_spectra_fin.append(np.sum(spectrum_det[23:38], axis =0))
    det_spectra_fin.append(np.sum(spectrum_det[39:56], axis =0))



    plt.errorbar(experimental_spectrum['logE'], plt_Data_spectrum, yerr=plt_Data_Err, fmt='o',  mfc='none', color='k', label = "Auger")
    plt.errorbar(exp_proton['lgE'][:BinNumber_p], plt_proton_spectrum[:BinNumber_p], yerr=plt_Data_Err_proton[:BinNumber_p], fmt='o',  mfc='none', color='tab:red', label = "protons")
    for i,e in enumerate(det_spectra_fin):
        e = np.power(10.,logE)
        plt.plot(logE, e**power_repr*norm*det_spectra_fin[i],  color=constant.colors[i], label = str(constant.lGroupLow[i])+"$\geq$ A $\leq$"+ str(constant.lGroupUp[i]))
    e = np.power(10.,logE)
    plt.plot(logE, e**power_repr*norm*expected_spectrum, label="low-E exptrapolation", color='tab:brown', linestyle='--')
    plt.plot(logE[BinNumber:], e[BinNumber:]**power_repr*norm*expected_spectrum[BinNumber:], label="best-fit model", color='tab:brown')

    #plt.ylim(1E23,3E24)
    plt.yscale('log')
    plt.legend(fontsize=12)

def draw_Xmax(logE,Xmax, RMS,experimental_xmax, E_fit, model):
    '''Draw the experimental and the expected Xmax mean and sigma

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
    model: `string`
        hadronic interaction model

    Returns
    -------
    None
    '''
    mass = [1, 4, 14, 56]
    massXmax = np.zeros((len(mass),len(logE)))
    massRMS  = np.zeros((len(mass),len(logE)))
    #massXmax, massRMS = [], []
    for i,lnA in enumerate(mass):
        lnA = np.log(mass[i])
        vXmax = xmax_tls.getXmax (logE, lnA, model)
        vVar = xmax_tls.getVar_sh (logE, lnA, model)
        massXmax[i] = vXmax
        massRMS[i] = np.sqrt(vVar)

    fig = plt.subplots(figsize=(10, 4), nrows=1, ncols = 2)
    plt.tight_layout(pad=3, w_pad=2, h_pad=2)
    plt.subplots_adjust(bottom = 0.16,top = 0.9)
    plt.subplot(121)
    plt.xlabel(r'Energy, log$_{10} \ (E/ \mathrm{eV}) $')
    plt.ylabel(r'<$X_{\mathrm{max}}>  [\mathrm { g \ cm^{-2}}]$')
    plt.xlim(18,20)
    plt.ylim(600,900)
    plt.text(20, 870, r'p', fontsize=15, color = 'red')
    plt.text(20, 830, r'He', fontsize=15, color = 'grey')
    plt.text(20, 800, r'N', fontsize=15, color = 'green')
    plt.text(20, 770, r'Fe', fontsize=15, color = 'blue')
    plt.text(18, 875, model, fontsize=17, color = 'black',fontweight='bold')

    for j,e in enumerate(mass):
        plt.plot(logE, massXmax[j],color = xmax_tls.colors_Xmax[j], linestyle='--' )

    plt.subplot(122)
    plt.xlabel(r'Energy, log$_{10} \ (E/ \mathrm{eV}) $')
    plt.ylabel(r'$ \sigma (X_{\mathrm{max}}) \  [\mathrm { g \ cm^{-2}}] $')
    plt.xlim(18,20)
    plt.ylim(0,70)
    for j,e in enumerate(mass):
        plt.plot(logE, massRMS[j],color = xmax_tls.colors_Xmax[j], linestyle='--')



    experimental_xmax["sysRMS_low"] = experimental_xmax["sysRMS_low"]* -1
    experimental_xmax["sysXmax_low"] = experimental_xmax["sysXmax_low"]* -1
    minLgE, maxLgE = M.get_Energies_2017()
    diffE1 = np.subtract(experimental_xmax["meanLgE"],minLgE)
    diffE2 = np.subtract(maxLgE, experimental_xmax["meanLgE"])

    BinNumber = np.ndarray.item(np.argwhere(logE == E_fit))
    plt.subplot(121)
    plt.errorbar(experimental_xmax["meanLgE"], experimental_xmax["fXmax"], fmt='o',yerr =[experimental_xmax["statXmax"],experimental_xmax["statXmax"]], xerr =[diffE1,diffE2], mfc='none', color = 'black')

    plt.plot(logE, Xmax, label="Total", color='tab:brown', linestyle='--')
    plt.plot(logE[BinNumber:], Xmax[BinNumber:], label="Total", color='tab:brown')
    plt.subplot(122)
    plt.errorbar(experimental_xmax["meanLgE"], experimental_xmax["fRMS"], fmt='o',yerr =[experimental_xmax["statRMS"],experimental_xmax["statRMS"]],xerr =[diffE1,diffE2], mfc='none', color = 'black')
    plt.plot(logE, RMS, label="Total", color='tab:brown', linestyle='--')
    plt.plot(logE[BinNumber:], RMS[BinNumber:], label="Total", color='tab:brown')


def Draw_Xmax_group(A,A_tot,frac,meanLgE,arr, convoluted_gumbel):
    '''Print the gumbel function grouping the mass fractions

    Parameters
    ----------
    A : `list`
        mass at the injection
    A_tot : `list`
        mass in the atmosphere
    frac : `list`
        fractions at the top of the atmosphere
    meanLgE : `list`
        fractions at the top of the atmosphere
    arr : `list`
        fractions at the top of the atmosphere
    convoluted_gumbel : `list`
        fractions at the top of the atmosphere
    model : `string`
        hadronic interaction model

    Returns
    -------
    total: 'list'
        return the gumbel functions convoluted per groups
    '''

    total_group = []
    for i in range( len(meanLgE)):
        group = np.zeros((len(A),len(arr[i])))
        frac[i] = frac[i]/np.sum (frac[i])
        fG =[0,0,0,0,0]
        for j in range(len(A_tot)):
            if (A_tot[j] == 1):
                fG[0] += frac[i][j]
            if (A_tot[j] >= 2 and A_tot[j] <= 4 ):
                fG[1] += frac[i][j]
            if (A_tot[j] >= 5 and A_tot[j] <= 22 ):
                fG[2] += frac[i][j]
            if (A_tot[j] >= 23 and A_tot[j] <= 38 ):
                fG[3] += frac[i][j]
            if (A_tot[j] >= 39 and A_tot[j] <= 56):
                fG[4] += frac[i][j]

        for k,a in enumerate(A):

            group[k] = np.multiply(convoluted_gumbel[i][a-1], fG[k])
        total_group.append(group)
    return total_group
