import sys
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from iminuit import minimize
from scipy import integrate
import healpy as hp

from combined_fit import spectrum as sp
from combined_fit import minimizer as minim
from combined_fit import tensor as ts
from combined_fit import mass
from combined_fit import constant
from combined_fit import xmax_distr
from combined_fit import draw
from combined_fit import map

dzdt = lambda z: constant._H0*(1+z)*np.sqrt(constant._OmegaLambda + constant._OmegaM*(1.+z)**3)


### Main ##########################################################
if __name__ == "__main__":



	################################# Inputs ##################################
	###########################################################################

	#Injected masses
	A	= [	1,	 4,	14,	28,	56]
	Z	= [	1,	 2,  7,	14, 26]

	logEmin = 18.9# Must be a 1 decimal value (18.2,18.3,etc.) 8EeV -> 18.9, 20EeV -> 19.3, 30 -> 19.5, 50 -> 19.7, 80 -> 19.9
	logEmin_CF = 18.7 # Value above which the emissivity is computed in the Combined Fit.
	SFR = True # Or False for SMD

	#Distribution of Sources
	dist_cut = 350#Mpc | Distance where the catalog ends
	zmax = 1
	nside = 64

	#Plot parameters
	smooth= "fisher" # Can be "fisher" or "top-hat"
	radius = 45 #deg
	galCoord= False # If true, the map will be plot in galactic coordinates


	if SFR==False:
		logRcut = 18.3797
		gamma = -0.369097
		L = np.array([ 1.03326e+42, 1.08766e+44, 2.18835e+44, 3.15289e+43, 8.84709e+42])
		f_z = ts.Load_evol(file_sfrd = "smd_local.dat", key="smd")
		trac = "logM*"
	else:
		logRcut = 18.2038
		gamma = -1.96024
		L = np.array([ 1.21357e+29, 3.06319e+44, 1.91915e+45, 3.88881e+43, 1.02726e+44])
		f_z = ts.Load_evol(file_sfrd = "sfrd_local.dat")
		trac = "logSFR"


	##### Load tensor and compute the bin which linked galaxy and isotropy ####
	###########################################################################

	#Loading  tensor
	Tensor=[]
	Tensor = ts.upload_Tensor(logEmin=logEmin)

	#Loading  tensor for computing the alpha coefficient
	Tensor_alpha=[]
	Tensor_alpha = ts.upload_Tensor(logEmin=1)


	### Load the binning in reshift
	bin_z = np.array(Tensor[0].z)
	delta_z = np.array(Tensor[0].delta_z)
	bin_dist = constant._fz_DL(bin_z)

	z_cut = constant._fDL_z(dist_cut)
	zcu = ts.find_n_given_z(z_cut, zmax=zmax) #Find the closest bin_z value before z_cut
	zc = zcu + 1
	over, under = bin_z[zc], bin_z[zcu]
	iso_bin_z_over = bin_z[zc:]

	# Safety net
	if over< z_cut or under>z_cut:
		print("WATCHOUT !!! Problem when using function find_n_given_z")

	### Insert a new bin or modify the first bin of the tensor in order for istropic background to start at z_cut
	NewBin=False
	if z_cut < (bin_z[zc]-delta_z[zc]/2):
		NewBin=True
		mini_bin_delta_z = (bin_z[zc]-delta_z[zc]/2)-z_cut
		new_zBin = ((bin_z[zc]-delta_z[zc]/2)+z_cut)/2
		iso_bin_z = np.insert(iso_bin_z_over, 0, new_zBin, axis=0)
	else:
		delta_z[zc] -= z_cut-(bin_z[zc]-delta_z[zc]/2)
		iso_bin_z = iso_bin_z_over


	###### Compute injected specrum and associate to galaxy and isotropy ######
	###########################################################################

	start = time.time()
	#Injected spectrum galaxies
	w_R = lambda ZA, logR: sp.Spectrum_Energy(ZA, logR, gamma, logRcut, logEmin=logEmin_CF)
	w_zR_Gal = lambda ZA, z, logR: w_R(ZA, logR)

	#Injected spectrum isotropic background (D>dist_cut)
	w_zR_Background = lambda ZA, z, logR: w_R(ZA, logR)/dzdt(z)*f_z(z)
	S_z = lambda z : 1/dzdt(z)*f_z(z)
	dtdz = lambda z: 1/dzdt(z)

	norm = integrate.quad(S_z, 0, 2.5)[0]/ integrate.quad(dtdz, 0, 2.5)[0] / (1/constant._Mpc_2_km)**2
	E_times_k = np.array(L/norm)

	dist, l, b, Cn, tracer, Rmin = map.load_Catalog(galCoord, Dmin=1., Dmax=dist_cut, tracer=trac, fluxmin=0.) # logSFR can be changed to logM*

	res =  np.array([ts.Return_lnA(Tensor, E_times_k, zmax, i, A, Z, w_zR_Gal) for i in tqdm(range(len(bin_z)))])
	#For each galaxy, compute the flux (result[Glaxy_Number,1] for a detected nuclei result[Glaxy_Number,0])
	result = map.match_Gal(res, dist, bin_dist, zmax)

	#Select tensor above dist_cut to compute isotropic background contribution
	iso_result_over = np.copy(res[zc:, :, :])
	iso_result_over[:, 1, :] = np.transpose(np.transpose(iso_result_over[:, 1, :])*delta_z[zc:])

	if NewBin:
		mini_bin = ((z_cut-under)*res[zc, :, :] + (over-z_cut)*res[zcu, :, :])/(over-under)
		mini_bin *= mini_bin_delta_z
		iso_result = np.insert(iso_result_over, 0, mini_bin, axis=0)
	else:
		iso_result = iso_result_over


	#Get the percentage 'alpha' of isotropic background.
	alpha_fact = map.alpha_factor(z_cut, logEmin, Tensor_alpha, A, Z, E_times_k, w_zR_Background, zmax)
	print("The percentage of isotropic background is alpha=", alpha_fact)

	# Load glaxies
	data = map.LoadShapedData(galCoord, dist*constant._Mpc_2_km, Cn, tracer, l, b)

	# Compute one map per detected nucleus for the foreground
	lnA_map = np.transpose(map.LoadlnAMap(data, nside, result[:,1]))/constant._c

	# Get one map per detected nucleus for the isotropic background
	iso_lnA_map = map.LoadIsolnAMap(nside, iso_result, iso_bin_z, lnA_map, alpha_fact, S_z)

	#Compute the the full maps (foreground + background) and smoothed it
	lnA_map_tot_unsmoothed = lnA_map + iso_lnA_map
	lnA_map_tot = map.LoadSmoothedMap(lnA_map_tot_unsmoothed, radius, nside, smoothing=smooth)

	# Compute and normalized the flux map
	int_flux = map.Compute_integrated_Flux(Tensor, E_times_k, A, Z, w_zR_Background)
	print("int_flux = ",  int_flux)
	Flux_map = np.sum(lnA_map_tot, axis=0)
	Flux_map = Flux_map / np.sum(Flux_map/hp.nside2npix(nside)) * int_flux
	end = time.time()
	print("Elapsed time for computing the map ", end - start)


	################################## Plots ##################################
	###########################################################################

	if galCoord:
		ax_title = "Galactic coordinates"
	else:
		ax_title = "Equatorial coordinates"

	fig_name = "Flux_map"

	print("Min flux : " , np.min(Flux_map), r" $km^{-2} \, yr^{-1} \, sr^{-1}$ | " ,"Max flux : " , np.max(Flux_map), r" $km^{-2} \, yr^{-1} \, sr^{-1}$" )
	title = r"Flux Map, $\Phi(\log_{10} (E/{\rm eV}) >$" + str(logEmin) + r"$)$ - Fisher smoothing with "+ str(radius) +"° radius"
	color_bar_title = r"Flux $[\rm km^{-2} \, yr^{-1} \, sr^{-1}]$"
	map.PlotHPmap(Flux_map, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, projection="mollweide", cmap='twilight_shifted', vmin=-1, vmax=-1)


	plt.show()
