import os
import sys
import numpy as np
from numba import jit
from scipy import interpolate
from astropy.table import Table
import pathlib

from combined_fit import gumbel
from combined_fit import xmax_tools as xmax_tls
from combined_fit import constant

COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

def acceptance_func(p, arr):
    ''' Acceptance function

    Parameters
    ----------
    p : `list`
        List of parameters needed
    x1,x2 : `int`
        limits of the acceptance function

    Returns
    -------
    a: 'float'
        return the acceptance function (y axis)
    '''
    x = np.arange(int(np.min(arr)-(constant.dx/2)),int(np.max(arr)+(constant.dx/2)),constant.dx, dtype = float)
    a = np.ones(2000, dtype = float)
    a = np.piecewise(x, [x < p[0], x > p[1]], [lambda x: np.exp((x-p[0])/p[2]),lambda x: np.exp(-(x-p[1])/p[3]),1])

    return  a


def resolution_func(p):
    ''' Resolution function using the PRD2014 parametrization

    Parameters
    ----------
    p : `list`
        List of parameters to be fitted

    Returns
    -------
    X,r: 'float'
        return the acceptance function (x-y axis)
    '''
    x = np.arange(-700,700, 2, dtype = float)
    r = np.arange(0,1,constant.dx, dtype=float)
    r = p[2]*xmax_tls.gaussian(x,0,p[0])+(1-p[2])*xmax_tls.gaussian(x,0,p[1])

    return x, r


def read_Xmax_syst():
    ''' Resolution function using the PRD2014 parametrization

    Parameters
    ----------
    None

    Returns
    -------
    stream : 'string'
        returns the content of a xmax systematics file
    '''
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Public_data/xmaxSystematics.txt')
    return Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)


def set_Xmax_distr(Xshift):
    ''' upload the Xmax Distribution

    Parameters
    ----------
    None

    Returns
    -------
    t: 'table'
        return the xmax-distribution tensor
    arr: 'list'
        return the x-axis of the experimental xmax distributions
    '''
    headers = ["Index", "minlgE", "maxlgE ","meanlgE", "nEntries ", "xMin ", "xMax"]

    for i in range(100):
        bin = "Bin_%d" % i
        headers.append(bin)
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Private_data/ICRC2017/XmaxBinsFDSt.txt')
    t = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)

    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Private_data/ICRC2017/Xmax_moments_icrc17_v2.txt')
    moments = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)

    subcolumn = headers[7:]
    '''------------- Shift ------------- '''
    eneshift = 0
    dEnScale = 0.14
    shift = 1 + dEnScale * eneshift
    t['minlgE'] += np.log10(shift)
    t['maxlgE'] += np.log10(shift)
    #syst = read_Xmax_syst()

    moments['sysXmax_Up'] = moments['sysXmax_Up'].astype('int64')
    moments['sysXmax_Low'] = moments['sysXmax_Low'].astype('int64')

    if (Xshift>0):
        t['xMin'] += Xshift*moments['sysXmax_Up'][6:]
        t['xMax'] += Xshift*moments['sysXmax_Up'][6:]
    if (Xshift<0):
        t['xMin'] += Xshift*moments['sysXmax_Low'][6:]
        t['xMax'] -= Xshift*moments['sysXmax_Low'][6:]

    print(t['xMin'], " ", t['xMax'])
    #exit()
    '''------------- Shift ------------- '''
    k = t[subcolumn]
    arr = np.zeros((len(t['Index']),100))
    for j in range(len(t['Index'])):
        list = []
        for i in range(len(k[j])):
            list.append(k[j][i])

        arr_try = np.array(list)
        arr[j] = arr_try
        # Consistency checks
        if np.sum(arr_try)-t['nEntries'][j] != 0:
            print( "err in xmaxHisto.txt")
            sys.exit()

    return t, arr


def deviance_Xmax_distr(data, model, nentries):
    ''' Compute the deviance of the Xmax distribution

    Parameters
    ----------
    data : `list`
        Experimental data
    model : `list`
        Expected model
    nentries : `list`
        Number of entries (at each energy bin) for the experimental distributions

    Returns
    -------
    X,r: 'float'
        return the acceptance function (x-y axis)
    '''
    idk = data > 0
    llsat = np.sum(np.log(data[idk]/nentries)*data[idk])
    llik = np.sum(np.log(model[idk])*data[idk])
    Dev = -2 * (llik - llsat)

    return Dev


def convolve_diff_fast(xmax,model,res,x):
    ''' Convolve a gumbel function (fast method)

    Parameters
    ----------
    xmax : `list`
        energy
    model : `list`
        Pure gumbel
    res,x : `list`
        Resolution function (x and y axis)

    Returns
    -------
    total: 'list'
        return the convoluted gumbel
    '''
    reso_function = interpolate.interp1d(x, res)
    meanxmax = xmax
    conv_func =np.zeros((len(xmax), len(xmax)), dtype = float)

    XX,YY = np.meshgrid(meanxmax, meanxmax)
    xmax_grid = XX-YY
    conv_func = model* reso_function(xmax_grid)
    total = np.sum(conv_func, axis = 1)

    return total


#@jit
def convolve_diff_slow(xmax,model,res,x, acc):
    ''' Convolve a gumbel function (slow method)

    Parameters
    ----------
    xmax : `list`
        energy
    model : `list`
        Pure gumbel
    res,x : `list`
        Resolution function (x and y axis)
    acc : `list`
        Acceptance function

    Returns
    -------
    total: 'list'
        return the convoluted gumbel and corrected for the acceptance
    '''
    conv_func =np.zeros((len(xmax), len(xmax)), dtype = float)
    reso_function = interpolate.interp1d(x, res)

    for j,e in enumerate(xmax):
        meanxmax = xmax[j]
        for jx,je in enumerate(xmax):
            meanxmax_2 = xmax[jx]
            window = np.linspace(meanxmax_2-meanxmax-(0.5*(constant.dx)),meanxmax_2-meanxmax+(0.5*(constant.dx)))
            resfun = np.trapz(reso_function(window), window)

            conv_func[jx][j] = model[j]*constant.dx*resfun*acc[j]

    sum = np.sum(conv_func, axis = 1)
    return sum


def Convolve_all(xmax,A_tot, arr, model):
    ''' Convolve all the gumbel for each energy and mass (at the beginning of the fit)

    Parameters
    ----------
    xmax : `table`
        xmax table (for minlgE, maxlgE e meanlgE)
    A_tot : `list`
        mass number at the top of the atmosphere
    arr : `list`
        experimental xmax distribution
    model : `string`
        hadronic interaction model

    Returns
    -------
    total: 'list'
        return the convoluted gumbel and corrected for the acceptance
    '''

    meanLgE =  xmax['meanlgE']
    filename = os.path.join(COMBINED_FIT_BASE_DIR,'../Private_data/ICRC2017/acceptance_resolution_2017.txt')
    data = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)
    gumbel_tot = []
    for i in range( len(meanLgE)):
        #print(i, " ", arr[i])
        p_acc = [data['x1'][i], data['x2'][i], data['l1'][i], data['l2'][i] ]
        a =  acceptance_func(p_acc,arr[i])

        p_reso = [data['resparam0'][i], data['resparam2'][i], data['resparam4'][i]]
        x, r = resolution_func(p_reso)
        sum = np.zeros((len(A_tot),len(arr[i])))

        for j in range(len(A_tot)):
            if A_tot[j] > 0:
                min = int((np.min(arr[i])-(constant.dx/2))/constant.dx)
                max = int((np.max(arr[i])+(constant.dx/2))/constant.dx)
                g = gumbel.Gumbel_function(A_tot[j], meanLgE[i], model)
                g.xmax = g.xmax[min:max]
                g.y = g.y[min:max]

                sum[j] = convolve_diff_slow(g.xmax, g.y, r,x,a)
                sum[j] = sum[j]/np.sum(sum[j])

        gumbel_tot.append(sum)
    #exit()
    return gumbel_tot
