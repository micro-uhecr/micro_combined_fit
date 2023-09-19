import numpy as np
from scipy import interpolate, integrate

########### Constants
colors =['tab:red', 'tab:gray', 'tab:green','tab:cyan', 'tab:blue', 'tab:brown', 'tab:pink','tab:olive']*3
colors_alt =['tab:orange', 'xkcd:silver', 'tab:olive','xkcd:azure', 'darkblue', 'tab:brown', 'tab:pink','tab:olive']*3
Masses = ['H','He', 'N', 'Si', 'Fe']

annotations = [#ra, dec, name
#                [80.894,-69.756,"LMC","u"],#0-1 Mpc
#                [11.893, -25.292,"NGC253","u"],#NGC0253 #1-10 Mpc
#                [0.5*(148.89+148.975), 0.5*(69.067+69.682),"M81/82","u"],#NGC3031, NGC3034
#                [204.25, -29.868,"M83","u"],#NGC5236
#                [67.705, 64.848, "NGC1569","u"],#NGC1569
#                [196.359, -49.471,"NGC4945","u"],#NGC4945
#                [39.148, 59.655, "Maffei 1","l"],#PGC009892
#                [201.37, -43.017,"Cen A","u"],#NGC5128
#                [184.74, 47.304, "M106","ll"],#NGC4258
#                [202.47, 47.234, "M51","u"],#NGC5194
#                [187.5, 8.9, "Virgo", "u"],#10-350 Mpc
#                [200.8, -33.3, "Shapley", "u"],# 0.0163  0.1456
#                [274.7, -62.3, "Laniakea", "u"],# 0.0112  0.0861
#                [46.3, 40.2, "Perseus-Pisces", "u"]
                ]

_OmegaM = 0.3
_OmegaLambda = 1-_OmegaM

_c = 299792.458#km/s
_c_ov_4pi = 23856.726#km/(s.sr)
_H0 = 70#km/s/Mpc

_RH = _c/_H0
_mp_eV = 940E6#eV

_Mpc_2_km = 3.08568E19
_yr_to_sec = 3.15576E7
uH_Gyr = _Mpc_2_km*1E-9/_yr_to_sec#s*Mpc/km -> Gyr
_erg_to_EeV = 0.624151E-6
_erg_to_eV = 0.624151E12

E_z_m1 = lambda z: 1./np.sqrt(_OmegaM*(1.+z)**3+1-_OmegaM)

_zmin,_zmax = 0, 15.
_z = np.linspace(_zmin,_zmax, 100000)
_DL = (1+_z)*integrate.cumtrapz(E_z_m1(_z),_z,initial=0)*_RH
_t = integrate.cumtrapz(E_z_m1(_z)/(1.+_z),_z,initial=0)*uH_Gyr/_H0
_dzdt = lambda z: _H0*(1+z)*np.sqrt(_OmegaLambda + _OmegaM*(1.+z)**3)
_dtdz_s = lambda z: 1E9*uH_Gyr*_yr_to_sec/_dzdt(z)

_fDL_z = interpolate.interp1d(_DL, _z)
_fz_DL = interpolate.interp1d(_z,_DL)
_ft_z = interpolate.interp1d(_t,_z)
_fz_t = interpolate.interp1d(_z,_t)
_fDL_t = interpolate.interp1d(_DL,_t)
_ft_DL = interpolate.interp1d(_t,_DL)

dx = 20


def load_virgo_properties(galCoord):
    dist0, R_500 = 15, 0.8#Mpc
    logM500 = np.log10(1.2E14)
    if galCoord: l0, b0 = 279.676173, 74.459570
    else: l0, b0 = 186.633700, 12.723300
    return dist0, l0, b0, R_500, logM500


def transparency_cluster(logM500, isProton):
    logM_free, sigma = 14.4, 0.25
    if isProton: lrho0, xi = 20.0, 0.6
    else: lrho0, xi = 24.3, 1.7

    lrho = lrho0 + xi*(logM500-15)
    Gamma = 2/(1+10**(-sigma*(logM500-logM_free)))

    logR = np.linspace(17,23, 60)
    logf = np.zeros_like(logR)#TBD: check with Antonio his actual formula (10^1 = 10... not 1)
    sel = logR < lrho
    logf[sel] += Gamma*(logR[sel]-lrho)

    return interpolate.interp1d(logR, 10**logf)


# Magnetic field
B_GMF_nG, lc_GMF_kpc, Lmax_GMF_Mpc = 1E3, 0.1, 10E-3
B_LGMF_nG, lc_LGMF_kpc, Lmax_LGMF_Mpc = 10, 10, 1
B_IGMF_nG, lc_IGMF_kpc, Lmax_IGMF_Mpc = 1E-4, 1E3, 1e10
B_default = [[B_GMF_nG, lc_GMF_kpc, Lmax_GMF_Mpc],\
            [B_LGMF_nG, lc_LGMF_kpc, Lmax_LGMF_Mpc],\
            [B_IGMF_nG, lc_IGMF_kpc, Lmax_IGMF_Mpc]]

def logRcut_transient(k, dist, lum):
    lambda_th = 0.7#median/mean of Poisson ~ 0 below lambda_th, ~1 above - see small script below
    '''
	lambda_poiss = np.logspace(-1,1,1000)
	med_ov_mean = []
	for l in lambda_poiss:
		realisation = np.random.poisson(l, size=1000)
		med_ov_mean.append(np.median(realisation)/np.mean(realisation))
	lambda_th = 0.7
	approx_fun = lambda l: l>lambda_th

	plt.xscale('log')
	plt.plot(lambda_poiss, med_ov_mean, label="Poisson median / mean")
	plt.plot(lambda_poiss, approx_fun(lambda_poiss), label="Approx with lambda_th="+str(lambda_th))
	plt.show()
	'''
    lambda_poiss_19 = k*lum*delta_t19_yr(dist)
    logRcut = 19 + 0.5*np.log10(lambda_poiss_19/lambda_th)#lambda = lambda_19 (R/1E19)-2 > lambda_th
    return logRcut


def delta_t19_yr(dist, B = B_default):
    '''interpolation function'''
    D = np.logspace(-2, 3)
    tau = tau19_propa_custom_yr(D, B)
    finterp = interpolate.interp1d(D, tau)
    return finterp(dist)


def tau19_propa_custom_yr(D, B):
    '''Propagation-induced delay at 10 EeV in yr'''
    '''Based on Achterberg et al. 1998 (see also S. Marafico's thesis, Eq. B.39)'''

    tau2 = np.zeros_like(D)
    for MF in B:
        B_nG, lc_kpc, Lmax_Mpc = MF[0], MF[1], MF[2]
        tau_max = 4.4E3 * (B_nG/10)**2 * (lc_kpc/10) * (Lmax_Mpc/1)**2 * np.ones_like(D)
        sel_below = D<Lmax_Mpc
        tau_max[sel_below] = tau_max[sel_below]*(D[sel_below]/Lmax_Mpc)**2
        tau2+=tau_max*tau_max

    return np.sqrt(tau2)


def theta_mean_deg(Rmean, B = B_default):
    '''Typical deflection angle at plateau distance'''
    theta = delta_theta19_deg(Lmax_LGMF_Mpc, B)/(Rmean/1E19)
    return theta


def delta_theta19_deg(dist, B = B_default):
    '''interpolation function'''
    D = np.logspace(-2, 3)
    theta = theta19_propa_custom_yr(D, B)
    finterp = interpolate.interp1d(D, theta)
    return finterp(dist)


def theta19_propa_custom_yr(D, B):
    '''Propagation-induced deflection at 10 EeV in deg'''
    '''Based on Achterberg et al. 1998 (see also S. Marafico's thesis, Eq. B.44)'''

    theta2 = np.zeros_like(D)
    for MF in B:
        B_nG, lc_kpc, Lmax_Mpc = MF[0], MF[1], MF[2]
        theta_max = 3.4 * (B_nG/10) * np.sqrt((lc_kpc/10) * (Lmax_Mpc/1)) * np.ones_like(D)
        sel_below = D<Lmax_Mpc
        theta_max[sel_below] = theta_max[sel_below]*np.sqrt(D[sel_below]/Lmax_Mpc)
        theta2+=theta_max*theta_max

    return np.sqrt(theta2)
