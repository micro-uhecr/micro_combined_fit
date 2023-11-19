import numpy as np
from scipy import interpolate, integrate
from combined_fit import constant


def load_cluster_properties(galCoord):
    if galCoord:
         l0 = constant.l0_gal #279.676173, 74.459570
         b0 = constant.b0_gal
    else:
        l0 = constant.l0_eq #186.633700, 12.723300
        b0 = constant.b0_eq
    return constant.dist_clusters, l0,b0, constant.R_500, constant.logM500


def transparency_cluster(logM500, isProton):
    logM_free, sigma = 14.4, 0.25
    if isProton: lrho0, xi = 20.0, 0.6
    else: lrho0, xi = 24.3, 1.7

    lrho = lrho0 + xi*(logM500-15)
    Gamma = 2/(1+10**(-sigma*(logM500-logM_free)))

    logR = np.linspace(17,23, 60)
    logf = np.zeros_like(logR)
    sel = logR < lrho
    logf[sel] += Gamma*(logR[sel]-lrho)

    return interpolate.interp1d(logR, 10**logf)



def logRcut_transient(k, dist, lum):
    lambda_th = 0.7#median/mean of Poisson ~ 0 below lambda_th, ~1 above - see small script below
    """
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
	"""
    lambda_poiss_19 = k*lum*delta_t19_yr(dist)
    logRcut = 19 + 0.5*np.log10(lambda_poiss_19/lambda_th)#lambda = lambda_19 (R/1E19)-2 > lambda_th
    return logRcut


def delta_t19_yr(dist, B = constant.B_default):
    """interpolation function"""
    D = np.logspace(-2.5, 3)
    tau = tau19_propa_custom_yr(D, B)
    finterp = interpolate.interp1d(D, tau)
    return finterp(dist)


def tau19_propa_custom_yr(D, B):
    """Propagation-induced delay at 10 EeV in yr"""
    """Based on Achterberg et al. 1998 (see also S. Marafico's thesis, Eq. B.39)"""

    tau2 = np.zeros_like(D)
    for MF in B:
        B_nG, lc_kpc, Lmax_Mpc = MF[0], MF[1], MF[2]
        tau_max = 4.4E3 * (B_nG/10)**2 * (lc_kpc/10) * (Lmax_Mpc/1)**2 * np.ones_like(D)
        sel_below = D<Lmax_Mpc
        tau_max[sel_below] = tau_max[sel_below]*(D[sel_below]/Lmax_Mpc)**2
        tau2+=tau_max*tau_max

    return np.sqrt(tau2)


def theta_mean_deg(Rmean, B = constant.B_default):
    """Typical deflection angle at plateau distance"""
    theta = delta_theta19_deg(constant.Lmax_LGMF_Mpc, B)/(Rmean/1E19)
    return theta


def delta_theta19_deg(dist, B = constant.B_default):
    """interpolation function"""
    D = np.logspace(-2, 3)
    theta = theta19_propa_custom_yr(D, B)
    finterp = interpolate.interp1d(D, theta)
    return finterp(dist)


def theta19_propa_custom_yr(D, B):
    """Propagation-induced deflection at 10 EeV in deg"""
    """Based on Achterberg et al. 1998 (see also S. Marafico's thesis, Eq. B.44)"""

    theta2 = np.zeros_like(D)
    for MF in B:
        B_nG, lc_kpc, Lmax_Mpc = MF[0], MF[1], MF[2]
        theta_max = 3.4 * (B_nG/10) * np.sqrt((lc_kpc/10) * (Lmax_Mpc/1)) * np.ones_like(D)
        sel_below = D<Lmax_Mpc
        theta_max[sel_below] = theta_max[sel_below]*np.sqrt(D[sel_below]/Lmax_Mpc)
        theta2+=theta_max*theta_max

    return np.sqrt(theta2)
