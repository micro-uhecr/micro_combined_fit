import numpy as np
from scipy import interpolate, integrate


########### Constants
colors = ['tab:red', 'tab:grey', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','tab:blue']*3
Masses = ['H','He', 'N', 'Si', 'Fe']

# Sizes used for plotting (see first lines of lib.py)
SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

_OmegaM = 0.3
_OmegaLambda = 1-_OmegaM

_c = 299792.458#km/s
_H0 = 70#km/s/Mpc
_G = 6.67E-11 #m^3/kg/s^2
_SolarMass =  1.98E30 #kg

_RH = _c/_H0

_mp_eV = 940E6#eV

_psi0 = 1.5E-2#M*/yr/Mpc^3

_Mpc_2_km = 3.08568E19
_yr_to_sec = 3.15576E7
uH_Gyr = _Mpc_2_km*1E-9/_yr_to_sec#s*Mpc/km -> Gyr
_erg_to_EeV = 0.624151E-6

_RhoC = 3*(_H0*1e3)**2 * _Mpc_2_km*1e3 / (_G  * 8 * np.pi) / _SolarMass

E_z_m1 = lambda z: 1./np.sqrt(_OmegaM*(1.+z)**3+1-_OmegaM)

_zmin,_zmax = 0, 15.
_z = np.linspace(_zmin,_zmax, 100000)
_DL = (1+_z)*integrate.cumtrapz(E_z_m1(_z),_z,initial=0)*_RH
_t = integrate.cumtrapz(E_z_m1(_z)/(1.+_z),_z,initial=0)*uH_Gyr/_H0

_fDL_z = interpolate.interp1d(_DL, _z)
_fz_DL = interpolate.interp1d(_z,_DL)
_ft_z = interpolate.interp1d(_t,_z)
_fz_t = interpolate.interp1d(_z,_t)
_fDL_t = interpolate.interp1d(_DL,_t)
_ft_DL = interpolate.interp1d(_t,_DL)


annotations = [#ra, dec, name
				[80.894,-69.756,"LMC","u"],#0-1 Mpc
				[11.893, -25.292,"NGC253","u"],#NGC0253 #1-10 Mpc
				[0.5*(148.89+148.975), 0.5*(69.067+69.682),"M81/82","u"],#NGC3031, NGC3034
				[204.25, -29.868,"M83","u"],#NGC5236
				[3.785, -39.22, "NGC55","u"],#NGC0055
				[67.705, 64.848, "NGC1569","u"],#NGC1569
				[196.359, -49.471,"NGC4945","u"],#NGC4945
				[39.148, 59.655, "Maffei 1","l"],#PGC009892
				[201.37, -43.017,"Cen A","u"],#NGC5128
				[184.74, 47.304, "M106","ll"],#NGC4258
				[202.47, 47.234, "M51","u"],#NGC5194
				[187.5, 8.9, "Virgo", "u"],#10-350 Mpc
				[200.8, -33.3, "Shapley", "u"],# 0.0163  0.1456
				[266.6, -49.4, "clone", "u"],# 0.0118  0.0926
				[298.9, -55.3, "C28b?", "u"],# 0.0117   0.091
				[274.7, -62.3, "Laniakea", "u"],# 0.0112  0.0861
				[46.3, 40.2, "Perseus-Pisces", "u"]
				]

# Magentic field
B_GMF_nG, lc_GMF_Mpc, Lmax_GMF_Mpc = 1E3, 100E-6, 10E-3#1 uG, 0.1 kpc, 10 kpc
B_LGMF_nG, lc_LGMF_Mpc, Lmax_LGMF_Mpc = 25, 10E-3, 1#25 nG, 10 kpc, 1 Mpc
B_IGMF_nG, lc_IGMF_Mpc, Lmax_IGMF_Mpc = 1e-4, 1, 1e10
B_default = [[B_GMF_nG, lc_GMF_Mpc, Lmax_GMF_Mpc],\
			[B_LGMF_nG, lc_LGMF_Mpc, Lmax_LGMF_Mpc],\
			[B_IGMF_nG, lc_IGMF_Mpc, Lmax_IGMF_Mpc]]
