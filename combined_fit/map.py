from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS, Galactic
import numpy as np
import time
from combined_fit import tensor as ts
from combined_fit import constant
from scipy import interpolate
import healpy as hp
import matplotlib.pyplot as plt
import os
import pathlib
import pandas as pd
import astropy.units as u
from scipy.sparse import csr_matrix, csc_matrix
import sparse
from tqdm import tqdm

# This gives the combined_fit/combined_fit directory
COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

def PlotHPmap(HPmap, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, projection="mollweide", cmap='afmhot', vmin=0, vmax=-1):
	''' Plot a healpy max

	Parameters
	----------
	HPmap : `numpy array`
		HealPy map
	nside : `int`
		nside associated to HPmap
	galCoord : `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	title : `string`
		title of the graph
	color_bar_title : `string`
		title of the color bar
	ax_title : `string`
		title of the ax
	fig_name : `string`
		name of the fig
	plot_zoa : `bool`
		if true the Zone Of Avaidance is plot
	write : `bool`
		if true will save the figure
	projection : `string`
		projection used (cf. Healpy)
	cmap : `string`
		color mar used
	vmin : `float`
		minimum value used to plot the map (if -1 take the minimum value of HPmap)
	vmax : `float`
		maximum value used to plot the map (if -1 take the maximum value of HPmap)


	Returns
	-------
	None
	'''

	# Transform healpix map into matplotlib map (grid_map)
	xsize = 2048# grid size for matplotlib
	ysize = int(xsize/2.)
	theta = np.linspace(np.pi, 0, ysize)
	phi   = np.linspace(-np.pi, np.pi, xsize)
	PHI, THETA = np.meshgrid(phi, theta)
	grid_pix = hp.ang2pix(nside, THETA, PHI)
	grid_map = HPmap[grid_pix]

	# Define the figure
	fig = plt.figure(figsize=(10,6))
	fig.suptitle(title, size=18)
	ax = fig.add_subplot(111,projection=projection)
	plt.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.02)
	labelsize = 12
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)

	# Set the size of the other fonts
	fontsize = 20
	font = {'size'   : fontsize}
	plt.rc('font', **font)

	# minimum and maximum values along the z-scale (colorbar)
	if vmax ==-1 :
		vmax = np.max(HPmap)
	if vmin ==-1 :
		vmin = np.min(HPmap)

	# Plot the map
	l = np.radians(np.linspace(-180, 180, xsize))
	b = np.radians(np.linspace(-90, 90, ysize))
	image = plt.pcolormesh(l[::-1], b, grid_map, rasterized=True,cmap=plt.get_cmap(cmap), shading='auto', vmin=vmin, vmax=vmax)

	lstyle = '-'
	if(not plot_zoa): lstyle = '--'

	#Zone of Avoidance
	def b_ZoA(l):
		return np.radians(20)*(np.abs(l)<np.radians(30))+np.radians(3)*(np.abs(l)>=np.radians(30))

	if(galCoord):
		l = np.radians(np.linspace(-180, 180, xsize))
		b = b_ZoA(l)
		plt.plot(l,b, color="gray",linewidth=1)
		plt.plot(l,-b, color="gray",linewidth=1)
	else:
		#upper enveloppe
		l = np.radians(np.linspace(-180, 180, xsize))
		l = (l + np.pi) % 2*np.pi - np.pi
		b_zoa = b_ZoA(l)
		c = SkyCoord(l=l, b=b_zoa, frame='galactic', unit="rad")
		ra_uzoa, dec_uzoa = 180.-c.icrs.ra.degree, c.icrs.dec.degree
		ra_uzoa = (ra_uzoa + 180.) % 360. - 180.
		#breaks the array if angular distance is large
		i_br = np.where(np.abs(ra_uzoa[:-1]-ra_uzoa[1:])>30)[0]
		plt.plot(np.radians(ra_uzoa[i_br[0]+1:i_br[1]]),np.radians(dec_uzoa[i_br[0]+1:i_br[1]]), color="gray",linewidth=1,linestyle=lstyle)

		#lower enveloppe
		l = np.radians(np.linspace(-180, 180, xsize))
		l = (l + np.pi) % 2*np.pi - np.pi
		mb_zoa = -b_ZoA(l)
		c = SkyCoord(l=l, b=mb_zoa, frame='galactic', unit="rad")
		ra_lzoa, dec_lzoa = 180.-c.icrs.ra.degree, c.icrs.dec.degree
		ra_lzoa = (ra_lzoa + 180.) % 360. - 180.
		#breaks the array if angular distance is large
		i_br = np.where(np.abs(ra_lzoa[:-1]-ra_lzoa[1:])>30)[0]
		plt.plot(np.radians(ra_lzoa[i_br[0]+1:i_br[1]]),np.radians(dec_lzoa[i_br[0]+1:i_br[1]]), color="gray",linewidth=1,linestyle=lstyle)


	#Supergalactic plane
	if(galCoord):
		l = np.radians(np.linspace(-180, 180, xsize))
		l = (l + np.pi) % 2*np.pi - np.pi
		b = np.zeros_like(l)
		c = SkyCoord(sgl=l, sgb=b, frame='supergalactic', unit="rad")
		l_sgp, b_sgp = -c.galactic.l.degree, c.galactic.b.degree
		l_sgp = (l_sgp + 180.) % 360. - 180.
		#breaks the array if angular distance is large
		i_br = np.where(np.abs(l_sgp[:-1]-l_sgp[1:])>30)[0]
		plt.plot(np.radians(l_sgp[i_br[0]+1:i_br[1]]),np.radians(b_sgp[i_br[0]+1:i_br[1]]), color="tab:red",linewidth=1)
	else:
		l = np.radians(np.linspace(-180, 180, xsize))
		l = (l + np.pi) % 2*np.pi - np.pi
		b = np.zeros_like(l)
		c = SkyCoord(sgl=l, sgb=b, frame='supergalactic', unit="rad")
		ra_sgp, dec_sgp = 180.-c.icrs.ra.degree, c.icrs.dec.degree
		ra_sgp = (ra_sgp + 180.) % 360. - 180.
		#breaks the array if angular distance is large
		i_br = np.where(np.abs(ra_sgp[:-1]-ra_sgp[1:])>30)[0]
		plt.plot(np.radians(ra_sgp[i_br[0]+1:i_br[1]]),np.radians(dec_sgp[i_br[0]+1:i_br[1]]), color="tab:red",linewidth=1)

	#kwargs = {'format': '%.2f'}
	#if(vmax>10): kwargs = {'format': '%.0f'}
	#elif(vmax>1):  kwargs = {'format': '%.1f'}
	cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.05)
	cb.set_label(color_bar_title, size=13)
	plt.tick_params(axis='x', colors='gray')
	# Plot the labels considering if it is galactic or equatorial coordinates
	ax.set_title(ax_title, fontdict={'fontsize': 13},)
	if(galCoord):
		ax.set_xticklabels([r"150$\degree$", r"120$\degree$", r"90$\degree$", r"60$\degree$", r"30$\degree$", r"GC", r"330$\degree$", r"300$\degree$", r"270$\degree$", r"240$\degree$", r"210$\degree$"])
		plt.xlabel('l', size=labelsize, color='gray')
		plt.ylabel('b', size=labelsize)
	else:
		ax.set_xticklabels([r"330$\degree$", r"300$\degree$", r"270$\degree$", r"240$\degree$", r"210$\degree$", r"180$\degree$", r"150$\degree$", r"120$\degree$", r"90$\degree$", r"60$\degree$", r"30$\degree$"])
		plt.xlabel('R.A.', size=labelsize, color='gray')
		plt.ylabel('Dec.', size=labelsize)

	# Annotations
	if galCoord==False:
		for annot in constant.annotations:

			ra_deg, dec_deg, text, u = annot
			theta_annot, phi_annot = MapToHealpyCoord(galCoord, np.radians(ra_deg), np.radians(dec_deg))
			npix = hp.nside2npix(nside)
			ind = hp.ang2pix(nside, phi_annot, theta_annot)

			dx, dy = 0.5, 0.5#deg
			if(u=="l"):
				dy=-8*dy
				dx=-15*dx
			elif(u=="ll"):
				dy=-12*dy
			elif(u=="uu"):
				dy=5*dy
				dx=10*dx

			x = np.radians(180.-ra_deg)
			y = np.radians(dec_deg)

			plt.plot([x], [y], marker='+', markersize=4, color='gray')
			x+=np.radians(dx)
			y+=np.radians(dy)
			plt.annotate(text, xy=(x, y), xycoords='data', xytext=(x, y), textcoords='data', color='gray', fontsize=14)

		# Local void
		ra, dec = 294.0, 15.0
		x = np.radians(180.-ra)
		y = np.radians(dec)
		plt.plot([x], [y], marker='o', mfc='none', markersize=4, color='white')
		dx, dy = 0.5, 0.5#deg
		x+=np.radians(dx)
		y+=np.radians(dy)
		plt.annotate("Local Void", xy=(x,y), xycoords='data', xytext=(x,y), textcoords='data', color='white', fontsize=14)

		plt.grid(True, alpha=0.25)

	if(write): plt.savefig(fig_name)



def match_Gal(result, dist, bin_dist, zmax):
	''' Associate each galaxy with a flux

	Parameters
	----------
	result : `numpy array`
		Numpy array return by ts.Flux_Per_A_Detected()
	dist : `numpy array`
		array of len `number of galaxy` with the distance of each galaxy in Mpc
	bin_dist : `numpy array`
		array in distance of the tensor (converted from redshift to luminosity distance)
	zmax : `float`
		maximum distance of the tensor

	Returns
	-------
	res : `numpy array`
		The flux of each galaxy in the same format as result
	'''

	start_time = time.time()
	idx = ts.find_n_given_z(constant._fDL_z(dist), zmax=zmax)
	weights = np.array([dist-bin_dist[idx], bin_dist[idx+1]-dist])
	den = np.sum(weights, axis=0)

	res = (result[idx, :, :]*weights[1, :, None, None] + result[idx+1, :, :]*weights[0, :, None, None])/den[:, None, None]
	print("Elapsed time for associating flux to galaxy ", time.time() - start_time)

	return res



def match_Gal_Deltat(res, Delta_t, k, tracer, dist, bin_dist, bin_logE, zmax):
	''' Associate each galaxy with a flux in a transient scenario

	Parameters
	----------
	res : `numpy array`
		Numpy array return by ts.Flux_Per_ZA_Energy_Detected()
	Delta_t : `function`
		Function to compute Delta_t
	k : `float`
		parameter k
	dist : `numpy array`
		array of len `number of galaxy` with the distance of each galaxy in Mpc
	bin_dist : `numpy array`
		array in distance of the tensor (converted from redshift to luminosity distance)
	bin_logE : `numpy array`
		array in logE of the tensor	(detected energy)
	zmax : `float`
		maximum distance of the tensor

	Returns
	-------
	Shaped_Final_Result : `numpy array`
		The flux of each galaxy in the same format as res
	'''

	# Find idx to associate to each galaxy
	idx = ts.find_n_given_z(constant._fDL_z(dist), zmax=zmax)
	weights = np.array([dist-bin_dist[idx], bin_dist[idx+1]-dist])
	den = np.sum(weights, axis=0)

	# Compute the rigidity for all detected energy and detected charge
	charge = np.array([x[1] for x in res[0,0]])[:,0]
	charge = charge[charge >0] # modification by A.C. 12/06/2023
	energy = [bin_logE]*np.size(charge)

	# Flatten and regroup them in order to compute once and for all
	rigidity = np.transpose(np.transpose(energy)-np.log10(charge))
	rig_order, rig_order_idx = np.unique(rigidity, return_index=True)

	# Compute the time Delta tau for each rigidity and for each source
	Delta_Deltat = np.array([tau_propa_custom_yr(dist,  constant.B_default, rig_order[::-1][i]) - tau_propa_custom_yr(dist,  constant.B_default,np.append(100,rig_order[::-1])[0:-1][i]) for i in range(len(rig_order))])

	# Compute Poisson parameter and drawn at random for each galaxy and each rigidity (quick)
	lambd = k*tracer*Delta_Deltat
	Nburst_drawn = np.zeros_like(lambd)
	ind_poisson = lambd <=1000
	ind_gaussian = lambd >1000
	Nburst_drawn[ind_poisson] = np.random.poisson(lambd[ind_poisson])
	Nburst_drawn[ind_gaussian] = np.random.normal(lambd[ind_gaussian], np.sqrt(lambd[ind_gaussian]))

	# Compute cumulative burst and divide it by the sum of lambda
	Cum_Number_Bursts = np.cumsum(Nburst_drawn, axis=0) / (k*tracer*np.array([tau_propa_custom_yr(dist, constant.B_default, rig_order[::-1][i]) for i in range(len(rig_order))]))
	Cum_Number_Bursts = np.transpose(Cum_Number_Bursts)

	# Associate to each of the rig_order an array of idx correspondig in rigidity
	ind_lambda = [np.where(rigidity==rig_order[::-1][i]) for i in range(len(rig_order))]

	coords = []
	value = []

	ind = np.transpose(np.where(Cum_Number_Bursts>0))


	for j, gal in (enumerate(ind)): #Loop over all galaxies and non zero rigidity

		# Since one rigidity can correspond to multiply E/Z combinaisation. A loop is done over all E/Z that are equal to the rigidity computed
		for k in range(len(ind_lambda[gal[1]][0])):
			bin_redshift = idx[gal[0]]
			bin_ZA_detected = ind_lambda[gal[1]][0][k]
			bin_energy_detected = ind_lambda[gal[1]][1][k]
			Flux_below = res[bin_redshift, 1, bin_ZA_detected, bin_energy_detected]
			Flux_above = res[bin_redshift+1, 1, bin_ZA_detected, bin_energy_detected]

			if Flux_below > 0 or Flux_above>0:
				#Store the coordinates with non zeros values
				coords.append([gal[0], bin_ZA_detected, bin_energy_detected])

				#The flux of each galaxy is multiply by the number of bursts
				if bin_redshift == -1:
					value.append(Cum_Number_Bursts[gal[0], gal[1]] * Flux_above )
				else:
					value.append(Cum_Number_Bursts[gal[0], gal[1]] * (Flux_below*weights[1,idx[gal[0]]] + Flux_above*weights[0,idx[gal[0]]] )/den[idx[gal[0]]] )

	# Store it in a sparce matrix
	Flux_matrix = sparse.COO(np.transpose(coords), value, shape=np.shape(res[idx, 1, :, :]))

	# Sum for all energies and convert it into a normal numpy array
	Flux_Matrix_Sum_Energy = Flux_matrix.sum(axis=2).todense()

	# Shape the result in order to match
	Shaped_Final_Result = Flux_Matrix_Sum_Energy[:, :, None]
	ZA_detected = np.array([res[0, 0, :, 0][:, None]]*len(idx))
	Shaped_Final_Result= np.append(ZA_detected, Shaped_Final_Result, axis=2)
	Shaped_Final_Result = np.array(np.swapaxes(Shaped_Final_Result, 1,2))

	return Shaped_Final_Result

def load_Catalog(galCoord=True, Dmin=1, Dmax=350, tracer="logSFR", fluxmin=0):
	''' Associate each galaxy with a flux

	Parameters
	----------
	galCoord : `bool`
		if true load galactic coordinates, if not, load equatorial coordinates
	Dmin : `float`
		minimum distance of injection in Mpc
	Dmax : `float`
		maximum distance of injection in Mpc
	tracer : `string`
		logarithm of the tracer consider (either logSFR or logM* here)
	fluxmin : `float`
		galaxies with a minimum value((10**t[tracer]/t['d']**2)>fluxmin) are considered

	Returns
	-------
	dist : `array`
		distance of the galaxy in Mpc
	l : `array`
		longitude
	b : `array`
		latitude
	Cn : `array`
		Correction factor (cf. arXiv:2105.11345)
	tracer_Of_UHECR : `array`
		value of the SFR (M_solar Mpc^-3 yr^-1) or Stellar mass (M_solar Mpc^-3)
	'''
	file_Catalog = os.path.join(COMBINED_FIT_BASE_DIR,"../Catalog/light_sm_sfr_baryon_od_Resc_Rscreen_merged.dat")
	t = Table.read(file_Catalog, format='ascii.basic', delimiter=" ", guess=False)
	choosen_Galaxies = (t['d']<=Dmax)*(t['d']>=Dmin)*((10**t[tracer]/t['d']**2)>fluxmin)
	tsel = t[choosen_Galaxies]
	tsel.sort("logSFR")
	dist = tsel['d']
	if galCoord:
		l = tsel['glon']
		b = tsel['glat']
	else:
		l = tsel['ra']
		b = tsel['dec']
	Cn = tsel['cNs']

	tracer_Of_UHECR = np.power(10, tsel[tracer])
	return dist, l, b, Cn, tracer_Of_UHECR

def alpha(Table, A, Z, E_times_k, z_cut, w_zR):
	''' Compute the contribution of the background compare to the foreground

	Parameters
	----------
	Table : `Tensor`
		Tensor loaded
	A : `numpy array`
		Mass injected
	Z : `numpy array`
		Charge injected
	E_times_k : `numpy array`
		Value of the parameter E_times_k
	z_cut : `numpy array`
		Redshift that split the foreground and the background
	w_zR : `function`
		Function that describes the evolution of source in redshift (z) and injected spectrum in Rigidity (R)

	Returns
	-------
	logE : `numpy array`
		Detected energy
	alpha : `numpy array`
		Contribution of the background above an energy logE
	alpha_Per_A_inj : `numpy array`
		Contribution of the background above an energy logE for each injected mass
	'''

	# uploading tensor
	all_Ts = []
	all_iso_Ts = []

	for i, a in enumerate(A):
		t = Table[i]

		zz, rr = np.meshgrid(t.z, t.logRi)
		weights = w_zR(Z[i], zz, rr)*t.delta_z

		target = t.tensor_stacked[:, :z_cut, :]
		iso_target = t.tensor_stacked[:, z_cut:, :]

		weight = weights[:, :z_cut]
		iso_weight = weights[:, z_cut:]

		iz = np.where(np.array(target.shape) == t.z[:z_cut].size)
		iR = np.where(np.array(target.shape) == t.logRi.size)

		T = E_times_k[i]*np.tensordot(target, weight, axes=([iR[0][0],iz[0][0]],[0,1]))
		iso_T = E_times_k[i]*np.tensordot(iso_target, iso_weight, axes=([iR[0][0],iz[0][0]],[0,1]))

		all_Ts.append(T/(10**t.logE * (t.logE[1]-t.logE[0]) * np.log(10)))
		all_iso_Ts.append(iso_T/(10**t.logE * (t.logE[1]-t.logE[0]) * np.log(10)))


	M = np.sum(all_Ts, axis=0)
	iso_M = np.sum(all_iso_Ts, axis=0)

	Flux_above_E = np.zeros_like(M)
	iso_Flux_above_E = np.zeros_like(M)

	Flux_Per_A_inj = np.zeros_like(all_Ts)
	denom = np.zeros_like(all_Ts)

	for i in range(len(M)):
		Flux_above_E[i]=np.sum(M[i:])
		iso_Flux_above_E[i]=np.sum(iso_M[i:])
		for h, a in enumerate(A):
			Flux_Per_A_inj[h][i] = np.sum(all_iso_Ts[h][i:])
			denom [h][i] = Flux_Per_A_inj[h][i] + np.sum(all_Ts[h][i:])

	num = iso_Flux_above_E
	denum = iso_Flux_above_E+Flux_above_E
	logE, alpha, alpha_Per_A_inj = [], [],  []

	for idx in range(len(denum)):
		#if the flux is above 1e-15 of maximum flux compute alpha.
		if denum[idx]>np.max(denum)*1e-15 :
			logE.append(Table[0].logE[idx])
			alpha.append(num[idx]/denum[idx])
			alpha_Per_A_inj.append(Flux_Per_A_inj[:,idx]/denum[idx])
	alpha_Per_A_inj = np.transpose(alpha_Per_A_inj)


	return logE, alpha, alpha_Per_A_inj

def alpha_factor(z_cut, logEmin, Table, A, Z, E_times_k, w_zR_alpha, zmax):
	''' Return the alpha factor that give the proportion of
	the background compare to the foreground above logEmin

	Parameters
	----------
	z_cut : `numpy array`
		Redshift that split the foreground and the background
	logEmin : `float`
		Above a minimum energy
	Table : `Tensor`
		Tensor loaded
	A : `numpy array`
		Mass injected
	Z : `numpy array`
		Charge injected
	E_times_k : `numpy array`
		Value of the parameter E_times_k
	w_zR_alpha : `function`
		Function that describes the evolution of source in redshift (z) and injected spectrum in Rigidity (R)
	zmax : `float`

	Returns
	-------
	fact : `float
		Alpha coefficient
	'''
	zcu = ts.find_n_given_z(z_cut, zmax=zmax)
	zc = zcu + 1
	over, under = Table[0].z[zc], Table[0].z[zcu]
	mini_w = [over-z_cut, z_cut-under]

	logE_alpha_array, alpha_array_over, useless = alpha(Table, A, Z, E_times_k, zc, w_zR_alpha)
	logE_alpha_array, alpha_array_under, useless = alpha(Table, A, Z, E_times_k, zcu, w_zR_alpha)
	alpha_GE_o = interpolate.interp1d(logE_alpha_array, alpha_array_over)
	alpha_GE_u = interpolate.interp1d(logE_alpha_array, alpha_array_under)
	fact_o = alpha_GE_o(logEmin)
	fact_u = alpha_GE_u(logEmin)
	fact = (fact_o*mini_w[1] + fact_u*mini_w[0])/np.sum(mini_w)
	return fact

def MapToHealpyCoord(galCoord, l, b):
	''' Convert RA/DEC or or lonGalactic/latGalactic
	into Healpy coordiantes

	Parameters
	----------
	galCoord : `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	l : `float or numpy array`
		Right Ascension or galactic longitude
	b : `float or numpy array`
		Declination or galactic latitude

	Returns
	-------
	phi : `float or numpy array`
		Phi healpy coordinates
	theta : `float or numpy array`
		Theta healpy coordinates
	'''

	theta = np.pi/2-b
	phi  = l

	if(not galCoord): # If equatorial coordinates
		phi -= np.pi
	return phi, theta

def LoadShapedData(galCoord, Dist, Cn, tracer, l, b):
	''' Shape the data into a numpy array

	Parameters
	----------
	galCoord : `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	Dist : `numpy array`
		Distance of the source in Mpc
	Cn : `numpy array`
		Correction factor (cf. arXiv:2105.11345)
	tracer : `numpy array`
		logarithm of the tracer consider (either logSFR or logM* here)
	Returns
	-------
	dataset : `numpy array`
		5D array with theta, phi, distance, Correcting factor and the tracer
	'''


	#Coordinate conversion
	phiData, thetaData = MapToHealpyCoord(galCoord, np.radians(l), np.radians(b))

	# dataset = np.asarray([thetaData, phiData, Dist, Cn, tracer])
	dataset = {'Theta': thetaData, 'Phi': phiData, 'Dist':  Dist, 'Cn': Cn, 'Tracer':  tracer }
	dataset_panda = pd.DataFrame(data=dataset)

	return dataset_panda

def Load_Map_A_Detected(dataset, nside, Mass_flux):
	''' Produce a array that can be plot using HealPy

	Parameters
	----------
	dataset : `numpy array`
		Dataset from LoadShapedData function
	nside : `int`
		nside parameter for Healpy map
	Mass_flux : `numpy array`
		Flux for each galaxy
	Returns
	-------
	Map_A_Detected : `numpy array`
		Map in a Healpy format
	'''

	# Pixel Not_Zero_exposureex for each events of coordinates (theta, phi)
	Galaxies = hp.ang2pix(nside, dataset['Theta'].to_numpy(), dataset['Phi'].to_numpy())
	dataset['Pixel'] = Galaxies
	#print(Mass_flux, " ", Galaxies)
	# Count map of parameter nside
	npix = hp.nside2npix(nside)# npix corresponds to the number of pixel associated to a NSIDE healpy map
	size_reshape = list(np.shape(Mass_flux)[1:])+[1]
	fact = np.tile(dataset['Tracer']/(dataset['Dist']**2 * dataset['Cn']), tuple(size_reshape)).T.astype('float64')
	fact *= Mass_flux.astype('float64')

	Map_A_Detected = np.zeros((npix, np.size(Mass_flux[0])))

	for i in range(len(Galaxies)):
		Map_A_Detected[Galaxies[i]] = Map_A_Detected[Galaxies[i]] + fact[i]

	#print(Map_A_Detected, " ", np.shape(Map_A_Detected), " ",np.shape(fact), " ", np.shape(Mass_flux) )


	return Map_A_Detected

def Load_IsoMap_A_Detected(nside, iso_result, iso_bin_z, Map_A_Detected, alpha_fact, f_z, coefficient=0.):
	''' Produce an istropic Healpy map coming from the background normalized
	regarding the alpha_factor and the map from the foreground (Map_A_Detected)

	Parameters
	----------
	nside : `int`
		nside parameter for Healpy map
	iso_result : `numpy array`
		Isotropic flux for each distance iso_bin_z
	iso_bin_z : `numpy array`
		Array in redshift
	Map_A_Detected : `numpy array`
		Foreground map
	alpha_fact : `float`
		Proportion of background compared to foreground
	f_z : `function`
		Evolution of sources
	Returns
	-------
	lnA_iso_map : `numpy array`
		Map in a Healpy format
	'''

	# Count map of parameter nside
	npix = hp.nside2npix(nside)# npix corresponds to the number of pixel associated to a NSIDE healpy map
	Sz = f_z(iso_bin_z)
	ax = np.where(np.array(iso_result[:, 1, :].shape)==Sz.size)
	var = np.tensordot(iso_result[:, 1, :], Sz, axes=(ax[0],[0]))

	# Construct map in Healpy format
	iso_results = [var]*npix
	iso_Map_A_Detected = np.transpose(iso_results)

	int = np.sum(Map_A_Detected)
	iso_int = np.sum(iso_Map_A_Detected)

	# Normalize it
	if iso_int == 0: #This is a simple patch to override division by zero at high logEmin
		lnA_iso_map = 0
	else:
		param = alpha_fact/(1-alpha_fact)
		if coefficient == 0:
			coefficient = param*(int/iso_int)
		lnA_iso_map = coefficient*iso_Map_A_Detected

	return lnA_iso_map

def top_hat_beam(radius, nside):
	''' Top hat smoothing function to be used using healpy smooth function

	Parameters
	----------
	nside : `int`
		nside parameter for Healpy map
	radius : `float`
		radius in radians for smoothing

	Returns
	-------
	hp.sphtfunc.beam2bl function
	'''
	b = np.linspace(0.0,np.pi,10000)
	bw = np.where(abs(b)<=radius, 1, 0)
	return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space

def fisher_beam(radius, nside):
	''' Fisher smoothing function to be used using healpy smooth function

	Parameters
	----------
	nside : `int`
		nside parameter for Healpy map
	radius : `float`
		radius in radians for smoothing

	Returns
	-------
	hp.sphtfunc.beam2bl function
	'''
	#print("rad  ", radius, " ", 1/radius**2)
	b = np.linspace(0.0,np.pi,10000)
	bw = np.float128(np.exp(np.float128(np.cos(b)/radius**2)))

	return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space

def LoadSmoothedMap(hp_map, radius_deg, nside, smoothing="fisher"):
	''' Perform a smoothing on a healpy map

	Parameters
	----------
	hp_map : `numpy array`
		healpy map
	radius : `float`
		radius in degree for smoothing
	nside : `int`
		nside parameter for Healpy map
	smoothing : `string`
		Function used: top_hat or fisher

	Returns
	-------
	Smoothed_map : `numpy array`
		Healpy map smoothed
	'''
	radius = np.radians(radius_deg)
	if smoothing == "fisher":
		beam_function = fisher_beam
	elif smoothing == "top-hat":
		beam_function = top_hat_beam

	if isinstance(hp_map[0],np.ndarray):
		Smoothed_map = [hp.smoothing(hp_map[i], beam_window=beam_function(radius, nside)) for i in range(len(hp_map))]
	else:
		Smoothed_map = hp.smoothing(hp_map, beam_window=beam_function(radius, nside))
	#print("end ", np.type(Smoothed_map))
	#exit()

	return Smoothed_map



'''def tau_propa_custom_yr(D, B, logR):  #  The v2 version can ingest D and R with different sizes
	Propagation-induced delay in yr
	Based on Eq. 5 in Stanev 2008 arXiv:0810.2501 assuming delay = spread
	if np.isscalar(D):
		tau2=0
		for MF in B:
			B_nG, lc_Mpc, Lmax_Mpc = MF[0], MF[1], MF[2]
			tau_max = 4.4E5 * B_nG**2 *  lc_Mpc * (Lmax_Mpc/100)**2
			if(D<Lmax_Mpc):
				tau_max = (D/Lmax_Mpc)**2
			tau2+=tau_max*tau_max
	else:
		tau2 = np.zeros_like(D)
		for MF in B:
			B_nG, lc_Mpc, Lmax_Mpc = MF[0], MF[1], MF[2]
			factor = 4.4E5 * B_nG**2 *  lc_Mpc * (Lmax_Mpc/100)**2
			tau_max = factor * np.ones_like(D)
			ind_below = D<Lmax_Mpc
			tau_max[ind_below] = tau_max[ind_below]*(D[ind_below]/Lmax_Mpc)**2
			tau2+=tau_max*tau_max

	return np.sqrt(tau2)'''# * (10**(logR-18)/100)**(-2) #yr

def tau_propa_custom_yr(D, B, logR):
	'''Propagation-induced delay in yr'''
	'''Based on Eq. 5 in Stanev 2008 arXiv:0810.2501 assuming delay = spread'''
	if np.isscalar(D):
		tau2=0
	else:
		tau2 = np.zeros_like(D)

	for MF in B:
		B_nG, lc_Mpc, Lmax_Mpc = MF[0], MF[1], MF[2]
		tau_max = np.zeros_like(D)
		if np.isscalar(D):
			if(D<Lmax_Mpc):
				tau_max = 4.4E5 * B_nG**2 * (D/100)**2 * lc_Mpc * (10**(logR-18)/100)**(-2) #yr
			else:
				tau_max = 4.4E5 * B_nG**2 * (Lmax_Mpc/100)**2 * lc_Mpc * (10**(logR-18)/100)**(-2) #yr
			tau2+=tau_max*tau_max
		else:
			ind_below = np.where(D<Lmax_Mpc)
			ind_above = np.where(D>=Lmax_Mpc)
			tau_max[ind_below] = 4.4E5 * B_nG**2 * (D[ind_below]/100)**2 * lc_Mpc * (10**(logR-18)/100)**(-2) #yr
			tau_max[ind_above] = 4.4E5 * B_nG**2 * (Lmax_Mpc/100)**2 * lc_Mpc * (10**(logR-18)/100)**(-2) #yr
			tau2+=tau_max*tau_max

	#return np.sqrt(tau2)
	return np.sqrt(tau2)#
def tau_propa_custom_yr_v2(D, B, logR):
	'''Propagation-induced delay in yr'''
	'''Based on Eq. 5 in Stanev 2008 arXiv:0810.2501 assuming delay = spread'''

	tau2 = np.zeros((len(D),len(logR)))*0.
	dist  = np.array([D] * len(logR)).T
	for MF in B:
		B_nG, lc_Mpc, Lmax_Mpc = MF[0], MF[1], MF[2]
		factor = 4.4E5 * B_nG**2 *  lc_Mpc * (Lmax_Mpc/100)**2
		tau_max = factor * np.ones((len(D),len(logR)))
		ind_below = np.where(dist<Lmax_Mpc)
		tau_max[ind_below] = tau_max[ind_below]*(dist[ind_below]/Lmax_Mpc)**2
		tau2+=tau_max*tau_max

	return np.sqrt(tau2) * (10**(logR-18)/100)**(-2)

def found_gal(dist0, l0, b0, dist, l, b, R):
	c1 = SkyCoord(l0*u.degree, b0*u.degree, frame='icrs')
	c2 = SkyCoord(l*u.degree, b*u.degree, frame='icrs')
	sep = c1.separation(c2)

	ind =  np.where((sep.radian<np.arctan(R/dist0)))[0]

	legnth = dist_Gal_cluster(dist0, R, sep.radian, dist)

	return ind, legnth[ind]

def dist_Gal_cluster(dist0, R, theta, dist):

	delta = 4 * dist0**2 - 4 *(1+np.tan(theta)**2)*(dist0**2-R**2)

	soluce_1, soluce_2 = np.zeros_like(dist), np.zeros_like(dist)


	ind = np.where(delta>0)[0]

	soluce_1[ind] = (2  * dist0 + np.sqrt(delta[ind]) ) / (2*(1+np.tan(theta[ind])**2))
	soluce_2[ind] = (2  * dist0 - np.sqrt(delta[ind]) ) / (2*(1+np.tan(theta[ind])**2))

	ind2 = np.where(soluce_1/np.cos(theta) > dist)
	soluce_1[ind2] =  dist[ind2]*np.cos(theta[ind2])

	ind3 = np.where(soluce_2/np.cos(theta) > dist)
	soluce_1[ind3] =  0
	soluce_2[ind3] =  0


	distance = np.zeros_like(dist)
	distance[ind] = (soluce_1[ind] - soluce_2[ind])/np.cos(theta[ind])

	return distance
def logR_escape(dist, size_cluster):
	r_array = np.linspace(0,size_cluster,1000)
	mean_B0 = np.mean(magnetic_field(r_array))

	logR = np.linspace(10, 30,1000)

	tau_at_dist = tau_propa_custom_yr(dist, [[mean_B0, 10e-3, 1e10]], logR)/np.sqrt(2)

	f = interpolate.interp1d(tau_at_dist, logR)
	bal_time =  dist*constant._Mpc_2_km/constant._c / constant._yr_to_sec
	#print(tau_at_dist, " ", bal_time)
	#if(dist != 0): exit(1)
	result =0

	if dist>1e-10:
		result = f(bal_time)
	print(dist," ", result)
	return result

def logR_escape_antonio(dist, size_cluster):
	def Cluster_f(a, x):
	    logM = 14.1
	    logM_free = 14.4
	    sigma = 4.39 #for protons
	    logR0 = 24.3 #20 for protons
	    xi = 1.7 #0.55 for protons
	    logRho = logR0 + xi*(logM-15)
	    Gamma = 2/(1+10**((logM_free-logM)/sigma))
	    logf = (Gamma*(x-logRho))
	    omega = 5
	    res = 1/(1+10**(-omega*logf))**(1/omega);

	    return res

	r_array = np.linspace(0,size_cluster,1000)
	mean_B0 = np.mean(magnetic_field(r_array))

	logR = np.linspace(10, 30,1000)
	a = 0
	res = Cluster_f(a,logR)
	val = interpolate.interp1d(res, logR)

	result =0

	if dist >size_cluster/2:
		result = val(0.5)
	#print(dist," ", result)
	return result
def Return_lnA_With_Rmin_With_Deltat(t, frac, zmax, bin_z, iRmin, A, Z, w_zR):
	#Computation##################################
	sel_A, sel_Z, sel_ZA, fractions = [], [], [], []


	for i, a in enumerate(A):
		je = t[i].Contract_Ri_Given_z_and_Rmin(t[i].tensor, w_zR, Z[i], bin_z, iRmin)/t[i].delta_z[bin_z]
		fractions.append(frac[i]*je)
		sel_ZA.append(t[i].A)

	ZA = np.concatenate(sel_ZA)
	frac_tot = np.concatenate(fractions, axis=0)

	uZA = np.unique(ZA)
	ZA_arr, stacked_frac_tot = [], []
	for za in uZA:
		sel = np.where(ZA == za)
		ZA_arr.append(za)
		stacked_frac_tot.append(np.sum(frac_tot[sel], axis=0))

	stacked_frac_tot = np.sum(np.array(stacked_frac_tot), axis=1)
	stacked_ZA = np.array(ZA_arr)

	stacked_ZA = np.transpose(stacked_ZA)
	#print(stacked_ZA, " ",stacked_frac_tot, " ", iRmin )
	#exit()

	return stacked_ZA, stacked_frac_tot

def magnetic_field(r):
	return constant.B0*(thermal_distrib(r)/constant.n0)**constant.eta

def thermal_distrib(r):
	return constant.n0*(1+(r/constant.rc)**2)**(-1.5*constant.beta)

def match_Gal_With_Rmin_With_Deltat(result, dist, bin_dist, Rmin, bin_logRi, zmax) :

	start_time = time.time()
	idx_z = ts.find_n_given_z(constant._fDL_z(dist), zmax=zmax)
	idx_Rmin = ts.find_nearest_Ri_bin(bin_logRi, Rmin)

	# Bilinear interpolation on (z, Ri) grid
	weights = np.array([(dist-bin_dist[idx_z])*(Rmin-bin_logRi[idx_Rmin]), (dist-bin_dist[idx_z])*(bin_logRi[idx_Rmin+1]-Rmin), (bin_dist[idx_z+1]-dist)*(Rmin-bin_logRi[idx_Rmin]), (bin_dist[idx_z+1]-dist)*(bin_logRi[idx_Rmin+1]-Rmin)])

	div = np.sum(weights, axis=0)


	final_res = (result[idx_Rmin, idx_z, :, :]*weights[3, :, None, None] + result[idx_Rmin+1, idx_z, :, :]*weights[2, :, None, None] + result[idx_Rmin, idx_z+1, :, :]*weights[1, :, None, None] + result[idx_Rmin+1, idx_z+1, :, :]*weights[0, :, None, None])/div[:, None, None]

	return final_res

'''def LoadlnAMap(dataset, nside, Mass_flux):

	# Pixel Not_Zero_exposureex for each events of coordinates (theta, phi)
	Galaxies = hp.ang2pix(nside, dataset[0], dataset[1])

	# Count map of parameter nside
	npix = hp.nside2npix(nside)# npix corresponds to the number of pixel associated to a NSIDE healpy map

	lnA_map = np.zeros((npix, np.size(Mass_flux[0])))

	for i in range(len(Galaxies)):
		lnA_map[Galaxies[i]] = lnA_map[Galaxies[i]] + Mass_flux[i]/(dataset[2][i]**2*dataset[3][i])*dataset[4][i]

	return lnA_map

def LoadShapedData(galCoord, Dist, Cn, M_star, l, b):
	#If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise. Minimum energy in EeV
	#Coordinate conversion

	phiData, thetaData = MapToHealpyCoord(galCoord, np.radians(l), np.radians(b))

	dataset = np.asarray([thetaData, phiData, Dist, Cn, M_star])
	return dataset

def match_Gal_Deltat(res, Delta_t, k, tracer, dist, bin_dist, bin_logE, zmax):

	start_time = time.time()

	idx = ts.find_n_given_z(constant._fDL_z(dist), zmax=zmax)
	ind_of_idx = np.where(idx == -1)
	idx[ind_of_idx] =0
	weights = np.array([dist-bin_dist[idx], bin_dist[idx+1]-dist])
	den = np.sum(weights, axis=0)

	mass = np.array([x[1] for x in res[0,0]])[:,0]
	rigidity = [bin_logE]*np.size(mass)
	rigidity = np.transpose(np.transpose(rigidity)-np.log10(mass))

	rig_order, rig_order_idx = np.unique(rigidity, return_index=True)


	Delta_Deltat = np.array([Delta_t(constant._fDL_z(dist),rig_order[::-1][i])-Delta_t(constant._fDL_z(dist),np.append(100,rig_order[::-1])[0:-1][i]) for i in range(len(rig_order))])

	lambd = k*tracer*Delta_Deltat
	random = np.zeros_like(lambd)
	ind_poisson = np.where(lambd <=1000)
	ind_gaussian = np.where(lambd >1000)
	random[ind_poisson] = np.random.poisson(lambd[ind_poisson])
	random[ind_gaussian] = np.random.normal(lambd[ind_gaussian], np.sqrt(lambd[ind_gaussian]))

	lambda_Values= np.cumsum(random, axis=0) / (k*tracer*np.array([Delta_t(constant._fDL_z(dist),rig_order[::-1][i]) for i in range(len(rig_order))]))
	lambda_Values = np.transpose(lambda_Values)


	lambda_v= np.zeros_like(res[idx, 1, :, :])
	ind_lambda = [np.where(rigidity==rig_order[::-1][i]) for i in range(len(rig_order))]

	for j in (range(len(idx))):
		for i in range(len(rig_order)):
			lambda_v[j][ind_lambda[i]] = lambda_Values[j][i]


	result_test = np.multiply(lambda_v, res[idx, 1, :, :])

	result_test = np.sum(result_test, axis=2)


	result_final = result_test[:, :, None]
	test = np.array([res[0, 0, :, 0][:, None]]*len(idx))

	result_final= np.append(test, result_final, axis=2)
	result_final = np.array(np.swapaxes(result_final, 1,2))

	return result_final'''

def Cluster_f(a, x):
	M_free = 10**14.4
	sigma = 0.25
	logR0 =20
	xi = 0.6
	logRho = logR0 + xi*(M/15)
	Gamma = 2/(1+(M/M_free)**(-sigma))
	logf = 1+Gamma*R/logRho
	return logf
