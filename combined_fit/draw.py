import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from combined_fit import spectrum as sp
from combined_fit import constant
from combined_fit import mass as M
from combined_fit import xmax_tools as xmax_tls


def MySaveFig(fig, pltname, pngsave=True):
    print ('Saving plot as ' + pltname + '.pdf')
    fig.savefig(pltname + '.pdf', dpi=300)
    if pngsave:
        fig.savefig(pltname + '.png', dpi=300)


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def Draw_spectrum(A, logE, expected_spectrum, spectrum_per_mass, norm, E_fit, hadr_model, Dev = None, lEmin = 17.8, lEmax = 20.2, isInjected  = True, isE3dJdE= True, isSysDisplayed=False, saveTitlePlot=None):
    ''' Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    A : `list`
        mass of injected particles
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum: `list`
        total expected spectrum at the top of the atmosphere
    spectrum_per_mass: `list`
        total expected spectrum at the top of the atmosphere
    norm : `float`
        normalization of the expected spectrum
    E_fit : `float`
        Energy bin from which the deviance is computed
    hadr_model : `string`
        hadronic interaction model
    Dev : `float`
        Deviance if None, not printed

    Returns
    -------
    None
        '''
    #Power to which energy is raised in plot e.g. 2 -> E2dJ/dE
    power_repr = 2
    if isE3dJdE: power_repr = 3
    sys_logE_dex = 0.14/np.log(10)#assumes 14%
    dsys_logE = power_repr*sys_logE_dex
    if power_repr<2.5: angle_rotate_deg = -70#to display sys

    #Infill + SD spectral points
    exp_spectrum = sp.load_Spectrum_Data()#eV, eV-1 km-2 yr-1 sr-1
    norm_repr = np.power(10,power_repr*exp_spectrum['logE'])#
    plt_Data_spectrum = norm_repr*exp_spectrum['J']#
    plt_Data_Err_up = norm_repr*exp_spectrum['J_up']#
    plt_Data_Err_low = norm_repr*exp_spectrum['J_low']#
    MinBinNumber = np.argmax(exp_spectrum['logE'] >= E_fit)

    #Proton spectral points
    exp_proton = sp.load_ProtonSpectrum_Data(hadr_model)#eV, eV-1 km-2 yr-1 sr-1
    norm_repr_p = np.power(10,power_repr*exp_proton['logE'])#
    plt_Proton_spectrum = norm_repr_p*exp_proton['J']#
    plt_Proton_Err_up = norm_repr_p*exp_proton['J_up']#
    plt_Proton_Err_low = norm_repr_p*exp_proton['J_low']#

    #Restrict protons to points below the E_fit
    MaxBinNumber_p =  np.argmax(exp_proton['logE'] > E_fit-0.01)
    plt_Proton_E = exp_proton['logE'][:MaxBinNumber_p]
    plt_Proton_spectrum = plt_Proton_spectrum[:MaxBinNumber_p]
    plt_Proton_Err_up = plt_Proton_Err_up[:MaxBinNumber_p]
    plt_Proton_Err_low = plt_Proton_Err_low[:MaxBinNumber_p]

    #Plot setup
    fig = plt.subplots(figsize=(6, 4), nrows=1, ncols = 1)
    plt.subplots_adjust(bottom = 0.15, top = 0.92, left=0.15, right=0.96)
    if Dev!=None:
        N_J = len(plt_Proton_E) + len(exp_spectrum['logE'][MinBinNumber:])
        plt.title(r"$D_{\rm J} = $"+ str( np.around(Dev, decimals = 1)) + r" ($N_{\rm J} = $" + str(N_J)+")", ha="center", va = 'center', fontsize=12)
    plt.tick_params(top=True, right=True)
    plt.xlabel(r'Energy, log$_{10}\, E$ [eV]')
    plt.ylabel(r'$E^2 J(E)$ [eV$\,$km$^{-2}\,$yr$^{-1}\,$sr$^{-1}\,$]')
    plt.xlim(lEmin,lEmax)
    plt.ylim(3E16,3E20)
    if isE3dJdE:
        plt.ylabel(r'$E^3 J(E)$ [eV$^{2}\,$km$^{-2}\,$yr$^{-1}\,$sr$^{-1}\,$]')
        plt.ylim(3E36,1E38)
    plt.yscale('log')

    #Plot data points
    if not isInjected:
        plt.errorbar(plt_Proton_E, plt_Proton_spectrum, yerr=[plt_Proton_Err_low, plt_Proton_Err_up], fmt='s',  mfc='none', color='tab:red', label = r"$p$ ("+hadr_model+", Mayotte+ '23)")
    plt.errorbar(exp_spectrum['logE'], plt_Data_spectrum, yerr=[plt_Data_Err_low, plt_Data_Err_up], fmt='o',  mfc='none', color='gray')
    plt.errorbar(exp_spectrum['logE'][MinBinNumber:], plt_Data_spectrum[MinBinNumber:], yerr=[plt_Data_Err_low[MinBinNumber:], plt_Data_Err_up[MinBinNumber:]], fmt='o',  mfc='none', color='k', label = r"$p + {}^A_Z{X}$ (Auger coll. '21) ")

    #Plot bounds w/ stat + sys
    if power_repr<2.5 and isSysDisplayed:#do not display systematics in E3dJdE -> overlap of bars
        J_upper_stat_sys = exp_spectrum["J"] + np.sqrt(exp_spectrum["J_up"]**2 + exp_spectrum["J_sys_up"]**2)
        J_lower_stat_sys = exp_spectrum["J"] - np.sqrt(exp_spectrum["J_low"]**2 + exp_spectrum["J_sys_low"]**2)
        Jp_upper_stat_sys = exp_proton["J"] + np.sqrt(exp_proton["J_up"]**2 + exp_proton["J_sys_up"]**2)
        Jp_lower_stat_sys = exp_proton["J"] - np.sqrt(exp_proton["J_low"]**2 + exp_proton["J_sys_low"]**2)

        MinPltNumber = np.argmax(exp_spectrum["logE"] > lEmin)

        t = mpl.markers.MarkerStyle(marker=r'$\ulcorner\urcorner$')
        t._transform = t.get_transform().rotate_deg(angle_rotate_deg)
        plt.scatter(exp_spectrum["logE"][MinBinNumber:]+dsys_logE, norm_repr[MinBinNumber:]*J_upper_stat_sys[MinBinNumber:], marker = t, color = 'k')
        plt.scatter(exp_proton["logE"][:MaxBinNumber_p]+dsys_logE, norm_repr_p[:MaxBinNumber_p]*Jp_upper_stat_sys[:MaxBinNumber_p], marker = t, color = 'tab:red')

        t = mpl.markers.MarkerStyle(marker=r'$\llcorner\lrcorner$')
        t._transform = t.get_transform().rotate_deg(angle_rotate_deg)
        plt.scatter(exp_spectrum["logE"][MinBinNumber:]-dsys_logE, norm_repr[MinBinNumber:]*J_lower_stat_sys[MinBinNumber:], marker = t, color = 'k')
        plt.scatter(exp_proton["logE"][:MaxBinNumber_p]-dsys_logE, norm_repr_p[:MaxBinNumber_p]*Jp_lower_stat_sys[:MaxBinNumber_p], marker = t, color = 'tab:red')

    #Plot model
    plot_lines, label_lines = [], []
    MinBinNumberModel =  np.argmax(logE >= E_fit)
    norm_repr = np.power(10.,power_repr*logE)*norm
    plt.plot(logE, norm_repr*expected_spectrum, color='tab:brown', linestyle='--')
    l, = plt.plot(logE[MinBinNumberModel:], norm_repr[MinBinNumberModel:]*expected_spectrum[MinBinNumberModel:], color='tab:brown')
    plot_lines.append(l)
    label_lines.append("Best-fit")

    if isInjected:
        color_mass = constant.colors_alt
    else:
        color_mass = constant.colors
        det_spectra_fin, A_det = [], []
        det_spectra_fin.append(spectrum_per_mass[1])
        A_det.append([0,1])
        det_spectra_fin.append(np.sum(spectrum_per_mass[2:5], axis =0))
        A_det.append([2,4])
        det_spectra_fin.append(np.sum(spectrum_per_mass[5:23], axis =0))
        A_det.append([5,22])
        det_spectra_fin.append(np.sum(spectrum_per_mass[23:39], axis =0))
        A_det.append([23,38])
        det_spectra_fin.append(np.sum(spectrum_per_mass[39:56], axis =0))
        A_det.append([39,56])
        spectrum_per_mass = det_spectra_fin

    for i, e in enumerate(spectrum_per_mass):
        l, = plt.plot(logE, norm_repr*spectrum_per_mass[i], color=color_mass[i])
        plot_lines.append(l)
        if isInjected:
            label_lines.append(r'$A_{\rm inj} = $'+str(A[i]))
        else:
            if i == 0:
                label_lines.append(r'$A_{\rm det} = $'+str(A_det[i][1]))
            else:
                label_lines.append(str(A_det[i][0])+r'$ \leq A_{\rm det} \leq $'+str(A_det[i][1]))

    legend1 = plt.legend(plot_lines, label_lines, fontsize=10, loc="upper right")
    plt.legend(fontsize=12, loc="lower left")
    plt.gca().add_artist(legend1)

    if saveTitlePlot is not None: MySaveFig(fig[0], saveTitlePlot)


def Draw_Xmax(logE, Xmax, RMS, experimental_xmax, E_fit, model, delta_shift_sys = 0, Dev=None, lEmin = 17.8, lEmax = 20.2, saveTitlePlot=None):
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
    delta_shift_sys: `float`
        shift of the model by delta g/cm2

    Returns
    -------
    None
    '''

    #Mean and RMS Xmax from models
    mass = [1, 4, 14, 56]
    massXmax = np.zeros((len(mass),len(logE)))
    massRMS  = np.zeros((len(mass),len(logE)))
    for i,lnA in enumerate(mass):
        lnA = np.log(mass[i])
        massXmax[i] = xmax_tls.getXmax(logE, lnA, model)
        massRMS[i] = np.sqrt( xmax_tls.getVar_sh(logE, lnA, model) )

    #Data
    experimental_xmax["sysRMS_low"] = experimental_xmax["sysRMS_low"]* -1
    experimental_xmax["sysXmax_low"] = experimental_xmax["sysXmax_low"]* -1
    MinBinData =  np.ndarray.item(np.argwhere(np.round(experimental_xmax["meanLgE"],2) == E_fit))

    #Plot setup
    fig, axs = plt.subplots(figsize=(6, 4), nrows=2, ncols = 1, sharex=True)
    plt.subplots_adjust(bottom = 0.15, top = 0.92, left=0.15, right=0.96, hspace=0.0)
    if Dev!=None:
        N_X = len(experimental_xmax["fXmax"][MinBinData:]) + len(experimental_xmax["fRMS"][MinBinData:])
        axs[0].set_title(r"$D_{\rm X} = $"+ str( np.around(Dev, decimals = 1)) + r" ($N_{\rm X} = $" + str(N_X)+")", ha="center", va = 'center', fontsize=12)

    ###### Mean Xmax plot ##########################################################
    axs[0].tick_params(labelbottom=False, bottom=False, top=True, right=True)
    axs[0].set_ylabel(r'$\langle X_{\mathrm{max}} \rangle \ \ [\mathrm { g \ cm^{-2}}]$')
    axs[0].set_xlim(lEmin,lEmax)
    axs[0].set_ylim(700,800)

    #Xmax shift of the model
    if delta_shift_sys!=0:
        axs[0].annotate(r"$\delta X_{\rm model}$", fontsize=12,
                        xy= (logE[0], massXmax[0][0]+delta_shift_sys+2), xytext=(logE[0], massXmax[0][0]-18),#3 and 18: MAGIC numbers so that the arrow size accounts for the small margins
                        arrowprops=dict(arrowstyle="->"), ha='center', verticalalignment='bottom')

    #Plot data points w/ stat. uncertainty
    axs[0].errorbar(experimental_xmax["meanLgE"], experimental_xmax["fXmax"], fmt='o',yerr = experimental_xmax["statXmax"], mfc='none', color = 'tab:gray')
    axs[0].errorbar(experimental_xmax["meanLgE"][MinBinData:], experimental_xmax["fXmax"][MinBinData:],yerr = experimental_xmax["statXmax"][MinBinData:], fmt='o', mfc='none', color = 'k', label = "Fitoussi+ '23")

    #Plot bounds w/ stat + sys
    Xmax_upper_stat_sys = experimental_xmax["fXmax"] + np.sqrt(experimental_xmax["statXmax"]**2 + experimental_xmax["sysXmax_up"]**2)
    Xmax_lower_stat_sys = experimental_xmax["fXmax"] - np.sqrt(experimental_xmax["statXmax"]**2 + experimental_xmax["sysXmax_low"]**2)
    axs[0].scatter(experimental_xmax["meanLgE"], Xmax_upper_stat_sys, marker = r'$\ulcorner\urcorner$', color = 'tab:gray')
    axs[0].scatter(experimental_xmax["meanLgE"], Xmax_lower_stat_sys, marker = r'$\llcorner\lrcorner$', color = 'tab:gray')
    axs[0].scatter(experimental_xmax["meanLgE"][MinBinData:], Xmax_upper_stat_sys[MinBinData:], marker = r'$\ulcorner\urcorner$', color = 'k')
    axs[0].scatter(experimental_xmax["meanLgE"][MinBinData:], Xmax_lower_stat_sys[MinBinData:], marker = r'$\llcorner\lrcorner$', color = 'k')

    #Plot model
    MinBinNumber = np.ndarray.item(np.argwhere(logE == E_fit))
    axs[0].plot(logE, Xmax, color='tab:brown', linestyle='--')
    axs[0].plot(logE[MinBinNumber:], Xmax[MinBinNumber:], color='tab:brown')
    for j,e in enumerate(mass):
        axs[0].plot(logE, massXmax[j],color = xmax_tls.colors_Xmax[j], linestyle=':' )

    axs[0].legend(loc = "lower right")

    ###### Sigma Xmax plot #########################################################
    axs[1].tick_params(bottom=True, top=True, right=False)
    axs[1].set_xlabel(r'Energy, log$_{10}\, E$ [eV]')
    axs[1].set_ylabel(r'$ \sigma (X_{\mathrm{max}})$')
    axs[1].set_ylim(15,75)

    #Plot data points
    axs[1].errorbar(experimental_xmax["meanLgE"], experimental_xmax["fRMS"],yerr =experimental_xmax["statRMS"], fmt='o', mfc='none', color = 'tab:gray')
    axs[1].errorbar(experimental_xmax["meanLgE"][MinBinData:], experimental_xmax["fRMS"][MinBinData:],yerr =experimental_xmax["statRMS"][MinBinData:], fmt='o', mfc='none', color = 'k')

    #Plot bounds w/ stat + sys
    RMS_upper_stat_sys = experimental_xmax["fRMS"] + np.sqrt(experimental_xmax["statRMS"]**2 + experimental_xmax["sysRMS_up"]**2)
    RMS_lower_stat_sys = experimental_xmax["fRMS"] - np.sqrt(experimental_xmax["statRMS"]**2 + experimental_xmax["sysRMS_low"]**2)
    axs[1].scatter(experimental_xmax["meanLgE"], RMS_upper_stat_sys, marker = r'$\ulcorner\urcorner$', color = 'tab:gray')
    axs[1].scatter(experimental_xmax["meanLgE"], RMS_lower_stat_sys, marker = r'$\llcorner\lrcorner$', color = 'tab:gray')
    axs[1].scatter(experimental_xmax["meanLgE"][MinBinData:], RMS_upper_stat_sys[MinBinData:], marker = r'$\ulcorner\urcorner$', color = 'k')
    axs[1].scatter(experimental_xmax["meanLgE"][MinBinData:], RMS_lower_stat_sys[MinBinData:], marker = r'$\llcorner\lrcorner$', color = 'k')

    #Plot model
    for j,e in enumerate(mass):
        l, = axs[1].plot(logE, massRMS[j],color = xmax_tls.colors_Xmax[j], linestyle=':')
        if j==1: lref=l
    axs[1].plot(logE, RMS, color='tab:brown', linestyle='--')
    axs[1].plot(logE[MinBinNumber:], RMS[MinBinNumber:], color='tab:brown')

    axs[1].legend([lref], [model], loc="upper right")

    axs[1].text(lEmax, 56, 'p', fontsize=12, color = 'tab:red')
    axs[1].text(lEmax, 40, 'He', fontsize=12, color = 'tab:grey')
    axs[1].text(lEmax, 28, 'N', fontsize=12, color = 'tab:green')
    axs[1].text(lEmax, 19, 'Fe', fontsize=12, color = 'tab:blue')

    fig.align_ylabels(axs[:])

    if saveTitlePlot is not None: MySaveFig(fig, saveTitlePlot)

def Draw_fractions(A, logE, expected_spectrum, spectrum_per_mass, E_fit, t_frac, him_text, lEmin = 17.8, lEmax = 20.2, isSysDisplayed=False, saveTitlePlot=None):
    ''' Plot the expected and the experimental spectrum above the threshold energy

    Parameters
    ----------
    A : `list`
        mass of injected particles
    logE : `list`
        list of  energy bins as stored in the tensor
    expected_spectrum: `list`
        total expected spectrum at the top of the atmosphere
    spectrum_per_mass: `list`
        total expected spectrum at the top of the atmosphere
    E_fit : `float`
        Energy bin from which the deviance is computed
    t_frac : `astropy table`
        observed fractions for a given hadronic interaction model
    him_text : `string`
        hadronic interaction model

    Returns
    -------
    None
        '''

    #Load model
    color_mass = [constant.colors[i] for i in [0,1,2,4]]
    masses = ["p", "He", "N", "Fe"]
    aliases = [r"$A_{\rm det} = 1$", r"$2 \leq A_{\rm det} \leq 4$", r"$5 \leq A_{\rm det} \leq 27$", r"$28 \leq A_{\rm det} \leq 56$"]
    det_spectra_fin, A_det = [], []

    det_spectra_fin.append(spectrum_per_mass[1])
    A_det.append([0,1])

    det_spectra_fin.append(np.sum(spectrum_per_mass[2:5], axis =0))
    A_det.append([2,4])

    det_spectra_fin.append(np.sum(spectrum_per_mass[5:28], axis =0))
    A_det.append([5,27])

    det_spectra_fin.append(np.sum(spectrum_per_mass[28:57], axis =0))
    A_det.append([28,56])

    spectrum_tot = det_spectra_fin[0] + det_spectra_fin[1] + det_spectra_fin[2] + det_spectra_fin[3]
    det_spectra_fin[0] /= spectrum_tot
    det_spectra_fin[1] /= spectrum_tot
    det_spectra_fin[2] /= spectrum_tot
    det_spectra_fin[3] /= spectrum_tot

    #Plot
    fig, axs = plt.subplots(figsize=(6, 4), nrows=4, ncols = 1, sharex=True)
    plt.subplots_adjust(bottom = 0.15, top = 0.92, left=0.15, right=0.96, hspace=0.05)
    axs[0].set_title(r"Fractions on Earth", ha="center", va = 'center', fontsize=12)
    axs[3].set_xlabel(r'Energy, $\log_{10} E$ [eV]')

    #plot model
    MinBinNumber = np.argmax(logE >= E_fit)
    for i in range(len(A_det)):
        frac = det_spectra_fin[i]
        axs[i].plot(logE, frac, color=color_mass[i], linestyle='--')
        l, = axs[i].plot(logE[MinBinNumber:], frac[MinBinNumber:], color=color_mass[i])
        if i==1: lref=l

    #plot data
    yticks = [0, 0.25, 0.5, 0.75, 1]
    xplot = np.linspace(lEmin, lEmax)
    for i in range(len(A_det)):
        axs[i].tick_params(top=True, right=True)
        axs[i].set_ylabel(masses[i])
        axs[i].set_yticks([0, 0.5, 1])
        for y in yticks: axs[i].plot(xplot, y*np.ones_like(xplot), lw=1, c='k', ls=':', alpha=0.5)
        axs[i].set_xlim(lEmin, lEmax)
        axs[i].set_ylim(-0.1,1.1)
        p = axs[i].errorbar(t_frac['meanLgE'], t_frac[masses[i]], yerr = [t_frac[masses[i]+"_err_low"], t_frac[masses[i]+"_err_up"]], fmt='o',  mfc='none', color=color_mass[i])
        if i==0:
            axs[0].legend([p, lref], ["Mayotte+ '23", him_text], loc="upper right")

        #Plot bounds w/ stat + sys
        upper_stat_sys = t_frac[masses[i]] + np.sqrt(t_frac[masses[i]+"_err_up"]**2 + t_frac[masses[i]+"_sys_up"]**2)
        lower_stat_sys = t_frac[masses[i]] - np.sqrt(t_frac[masses[i]+"_err_low"]**2 + t_frac[masses[i]+"_sys_low"]**2)
        axs[i].scatter(t_frac["meanLgE"], upper_stat_sys, marker = r'$\ulcorner\urcorner$', color = color_mass[i])
        axs[i].scatter(t_frac["meanLgE"], lower_stat_sys, marker = r'$\llcorner\lrcorner$', color = color_mass[i])

    if saveTitlePlot is not None: MySaveFig(fig, saveTitlePlot)
