import numpy as np
import math
import sys
from combined_fit import xmax_tools as xmax_tls

# EPOS-LHC = {'a_mu': [775.457, -10.3992, -1.75256],}
a_mu = {'EPOS-LHC':[775.457, -10.3992, -1.75256],
        'QGSJETII04':[758.65, -12.3571, -1.24538],
        'Sibyll2.3d':[785.852,-15.5994, -1.06906]}
b_mu = {'EPOS-LHC':[58.5286, -0.825599, 0.231713],
        'QGSJETII04':[56.5943 -1.01236, 0.228675],
        'Sibyll2.3d':[60.5929, -0.786014, 0.200728]}
c_mu = {'EPOS-LHC':[-1.40843, 0.226392, -0.100243], 'QGSJETII04':[-0.534763, -0.172777, -0.0191698], 'Sibyll2.3d':[-0.959349, 0.211336, -0.0647766]}
a_sigma = {'EPOS-LHC':[32.2628, 3.943,-0.864423], 'QGSJETII04':[35.4235, 6.75919, -1.46182], 'Sibyll2.3d':[41.0345, -2.17329, -0.306202]}
b_sigma = {'EPOS-LHC':[1.27601, -1.81337, 0.231914], 'QGSJETII04':[-0.796074, 0.20184, -0.0142566], 'Sibyll2.3d':[-0.309466, -1.16496, 0.225445]}
a_lambda = {'EPOS-LHC':[0.64108, 0.219734, 0.171156], 'QGSJETII04':[0.671547, 0.373902, 0.0753259], 'Sibyll2.3d':[0.799493, 0.235235, 0.00856884]}
b_lambda = {'EPOS-LHC':[0.0725804, 0.0353218, -0.0130931], 'QGSJETII04':[0.030434, 0.0473988, -0.000563764], 'Sibyll2.3d':[0.0632135, -0.0012847,0.000330525]}


def gumbelFun(sigma, lambd,mu):
    '''Implement the gumbel function

    see https://arxiv.org/pdf/1305.2331.pdf

    Parameters
    ----------
    sigma : `float`
        Spread of the distribution
    lambd : `float`
        Interaction lenght for the particle
    mu : `float`
        Mean of the distribution

    Returns
    -------
    x : 'list'
        gumbel function along x axis
    gumbel : 'list'
        gumbel functions along y axis
    '''
    x = np.arange(0,2000,20)
    x = x+ 10
    arg =  -lambd*((x - mu)/sigma)-lambd*np.exp(-((x - mu) / sigma))
    gumbel = 1./sigma * (lambd**lambd)/math.gamma(lambd)*math.e**arg
    return x,gumbel


class Gumbel_function:
    '''Compute the gumbel fuction for the chosen HIM

    Parameters
    ----------
    A : `int`
        Mass of the interacting particle
    E : `float`
        lg(E) of the interacting particle
    hadr : `float`
        hadronic interaction model

    Returns
    -------
    None
    '''
    def __init__(self, A, E, hadr):
        '''Gumbel_function class constructor
        '''
        self.A = A
        self.logE = E
        self.hadr = hadr
        if hadr not in xmax_tls.HIM_list:
            print("Hadronic interaction model not valid! ", hadr)
            sys.exit(0)


        lE = self.logE-19
        f_lnA = np.log(self.A)
        fA = [1,f_lnA, f_lnA*f_lnA]

        p0_mu = np.dot(a_mu[hadr],(fA))
        p1_mu = np.dot(b_mu[hadr],(fA))
        p2_mu = np.dot(c_mu[hadr],(fA))

        p0_sigma = np.dot(a_sigma[hadr],(fA))
        p1_sigma = np.dot(b_sigma[hadr],(fA))
        p0_lambda = np.dot(a_lambda[hadr],(fA))
        p1_lambda = np.dot(b_lambda[hadr],(fA))

        mu = p0_mu + p1_mu*lE + p2_mu*lE*lE
        sigma = np.around(p0_sigma + p1_sigma*lE, decimals = 4)
        lambd = p0_lambda + p1_lambda*lE

        self.xmax,self.y = gumbelFun(sigma, lambd,mu)
