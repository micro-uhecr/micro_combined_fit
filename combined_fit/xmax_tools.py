import sys
import numpy as np

colors_Xmax = ['tab:red', 'tab:grey', 'tab:green', 'tab:blue']
HIM_list = ["EPOS-LHC", "QGSJET-II04", "Sibyll"]

_sysXmax = 8#g/cm2


def getXRMS (lgE, meanVar_sh, VarLnA, model):
    """ This function provides the Xmax RMS

    Parameters
    ----------
    lgE: `float`
        log of energy
    meanVar_sh: `float`
        mean variance of lnA
    VarLnA: `float`
        variance of lnA
    model: `string`
        hadronic interaction model

    Returns
    -------
    np.sqrt(Var): 'float'
        Xmax RMS
    """
    par  = getParX (model)
    D = par[1]
    csi = par[2]
    delta = par[3]

    fE = csi - D/np.log(10) + delta * (lgE - 19.)
    Var = meanVar_sh + fE * fE * VarLnA

    return np.sqrt(Var)


def getXmax (lgE, lnA, model, sigma_shift_sys=0):
    """ This function provides the mean Xmax

    Parameters
    ----------
    lgE: `float`
        log of energy
    lnA: `float`
        lnA
    model: `string`
        hadronic interaction model
    sigma_shift_sys: `float`
        shift of the model by nsigma_sys

    Returns
    -------
    vXmax: 'float'
        Xmax mean
    """
    paramXmax = getParX (model)
    X0 = paramXmax[0]
    D = paramXmax[1]
    csi = paramXmax[2]
    delta = paramXmax[3]

    fE = csi - D/np.log(10) + delta * (lgE - 19.)
    Xmaxp = X0 + D * (lgE - 19.)

    vXmax = Xmaxp + fE * lnA

    return vXmax + sigma_shift_sys*_sysXmax


def getParX (model):
    """ Take the parameters for the chosen HIM (see GAP2018-021)
        necessary for mean Xmax

    Parameters
    ----------
    model: `string`
        hadronic interaction model

    Returns
    -------
    np.sqrt(Var): 'list'
        Parameters for the chosen HIM
    """
    # - EPOS-LHC -----------------------------------
    #    X01:            806.0366 +/- 0.2642
    #    D1:             56.2948 +/- 0.2491
    #    csi:              0.3463 +/- 0.1191
    #    delta:              1.0442 +/- 0.0995
    # ----------------------------------------------
    ParamEL = [ 806.04, 56.29, 0.346, 1.044]

    # - QGSJETII-04 --------------------------------
    #    X01:            790.1490 +/- 0.2642
    #    D1:             54.1910 +/- 0.2491
    #    csi:             -0.4170 +/- 0.1191
    #    delta:              0.6927 +/- 0.0995
    # ----------------------------------------------
    ParamQ2L = [790.15, 54.19, -0.417, 0.693]

    # - SIBYLL 2.3d --------------------------------
    #---------------------------------------------
    #   X01:            815.8664 +/- 0.2642
    #   D1:             57.8733 +/- 0.2491
    #   csi:             -0.3035 +/- 0.1191
    #   delta:              0.7963 +/- 0.0995
    #----------------------------------------------

    ParamS23d = [815.8664, 57.8733,-0.3035, 0.7963]

    if model not in HIM_list :
        print("Hadronic interaction model not valid! ", model)
        sys.exit(0)

    if model == "EPOS-LHC":
        return ParamEL
    elif model == "QGSJET-II04":
        return ParamQ2L
    else:
        return ParamS23d


def getParS(model):
    """ Take the parameters for the chosen HIM (see GAP2018-021)
        necessary for sigma Xmax

    Parameters
    ----------
    model: `string`
        hadronic interaction model

    Returns
    -------
    p: 'list'
        Parameters for the chosen HIM
    """
    pS = [3.72671955e+03, -4.83807008e+02, 1.32476965e+02, -4.05484134e-01, -2.53645381e-04, 4.74942981e-02] # parameter sibyll2.3d
    pE = [3.28443199e+03, -2.59993619e+02, 1.32120013e+02, -4.62258867e-01, -8.27458740e-04, 5.88101979e-02] # parameter eposlhc
    pQ = [3.73791570e+03, -3.74535249e+02, -2.12774852e+01, -3.96929960e-01, 8.17397773e-04, 4.57900290e-02] # parameter qgsjetII04

    if model not in HIM_list:
        print("Hadronic interaction model not valid! ", model)
        sys.exit(0)

    if model == "EPOS-LHC":
        return pE
    elif model == "QGSJET-II04":
        return pQ
    else:
        return pS


def  getVar_sh (lgE, lnA, model):
    """ Get the variance for a certain HIM

    Parameters
    ----------
    lgE: `float`
        log of energy
    lnA: `float`
        lnA
    model: `string`
        hadronic interaction model

    Returns
    -------
    variance: 'float'
        Xmax variance
        """
    if model not in HIM_list:
        print("Hadronic interaction model not valid! ", model)
        sys.exit(0)

    par = getParS (model)

    Vp = par[0] + par[1] * (lgE - 19.) + par[2] * (lgE - 19.) * (lgE - 19.)
    a = par[3] + par[4] * (lgE - 19.)
    b = par[5]

    variance = Vp * (1 + a * lnA + b * lnA * lnA)

    return variance
