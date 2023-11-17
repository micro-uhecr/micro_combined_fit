import numpy as np
from scipy import interpolate, integrate

########### Constants
colors = ['tab:red', 'tab:gray', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:brown', 'tab:pink', 'tab:olive']*3
colors_alt = ['tab:orange', 'xkcd:silver', 'tab:olive', 'xkcd:azure', 'darkblue', 'tab:brown', 'tab:pink', 'tab:olive']*3
Masses = ['H','He', 'N', 'Si', 'Fe']

annotations = [#ra, dec, name
#                [266.417,-29.008,"Sgr A*","ll"],#0-1 Mpc

                [23.462,30.660,"Triangulum","u"],#0-1 Mpc
                [10.685,41.269,"Andromeda","u"],#0-1 Mpc

                [3.723, -39.197,"NGC55","uu"],#NGC055 #1-10 Mpc
                [11.893, -25.292,"NGC253","uu"],#NGC0253 #1-10 Mpc
                [67.705, 64.848, "NGC1569","u"],#NGC1569

                [204.25, -29.868,"M83","u"],#NGC5236
                [201.37, -43.017,"Cen A","u"],#NGC5128
                [196.359, -49.471,"NGC4945","u"],#NGC4945
                [39.148, 59.655, "Maffei 1","ll"],#PGC009892
                [184.74, 47.304, "M106","ll"],#NGC4258
                [0.5*(148.89+148.975), 0.5*(69.067+69.682),"M81/82","u"],#NGC3031, NGC3034
                
                [270., -60., "Laniakea supercluster", "u"]# 0.0112  0.0861
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


# Magnetic field
B_GMF_nG, lc_GMF_kpc, Lmax_GMF_Mpc = 1E3, 0.1, 10E-3
B_LGMF_nG, lc_LGMF_kpc, Lmax_LGMF_Mpc = 10, 10, 2
B_IGMF_nG, lc_IGMF_kpc, Lmax_IGMF_Mpc = 1E-4, 1E3, 1e10
B_default = [[B_GMF_nG, lc_GMF_kpc, Lmax_GMF_Mpc],\
            [B_LGMF_nG, lc_LGMF_kpc, Lmax_LGMF_Mpc],\
            [B_IGMF_nG, lc_IGMF_kpc, Lmax_IGMF_Mpc]]


dist_clusters = 15,70 #Virgo, Perseus
logM500 = np.log10(1.2E14), np.log10(5.8E14)
l0_gal = 279.676173, 150.572468
b0_gal = 74.459570, -13.261685
l0_eq = 186.633700, 49.9467009
b0_eq = 12.723300, 41.513100
R_500 = 0.77,1.28
