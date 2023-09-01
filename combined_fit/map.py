import os
import pathlib
import healpy as hp
import matplotlib.pyplot as plt

import astropy.units as u
import numpy as np
from scipy import interpolate
from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS, Galactic

from combined_fit import constant, draw, utilities
from combined_fit import spectrum as sp

# This gives the combined_fit/combined_fit directory
COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()


def PlotHPmap(HPmap, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=True, write=False, projection="hammer", cmap='afmhot', vmin=-1, vmax=-1):
	""" Plot a healpy max

	Parameters
	----------
	HPmap: `numpy array`
		HealPy map
	nside: `int`
		nside associated to HPmap
	galCoord: `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	title: `string`
		title of the graph
	color_bar_title: `string`
		title of the color bar
	ax_title: `string`
		title of the ax
	fig_name: `string`
		name of the fig
	plot_zoa: `bool`
		if true the Zone Of Avaidance is plot
	write: `bool`
		if true will save the figure
	projection: `string`
		projection used (cf. Healpy)
	cmap: `string`
		color mar used
	vmin: `float`
		minimum value used to plot the map (if -1 take the minimum value of HPmap)
	vmax: `float`
		maximum value used to plot the map (if -1 take the maximum value of HPmap)


	Returns
	-------
	None
	"""

	# Transform healpix map into matplotlib map (grid_map)
	xsize = 2048# grid size for matplotlib
	ysize = int(xsize/2.)
	theta = np.linspace(np.pi, 0, ysize)
	phi   = np.linspace(-np.pi, np.pi, xsize)
	PHI, THETA = np.meshgrid(phi, theta)
	grid_pix = hp.ang2pix(nside, THETA, PHI)
	grid_map = HPmap[grid_pix]

	# Define the figure
	fig = plt.figure(figsize=(9.5,6))
	fig.suptitle(title, size=18)
	ax = fig.add_subplot(111,projection=projection)
	plt.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.02)
	labelsize = 12
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)

	# Set the size of the other fonts
	fontsize = 15
	font = {'size'  : fontsize}
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

	if plot_zoa:
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

	cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.04)
	cb.set_label(color_bar_title, fontsize=15)
	plt.tick_params(axis='x', colors='gray')
	# Plot the labels considering if it is galactic or equatorial coordinates
	ax.set_title(ax_title, fontdict={'fontsize': 14},)
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

	if(write): draw.MySaveFig(fig, fig_name)


def MapToHealpyCoord(galCoord, l, b):
	""" Convert RA/DEC or lonGalactic/latGalactic
	into Healpy coordiantes

	Parameters
	----------
	galCoord: `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	l: `float or numpy array`
		Right Ascension or galactic longitude
	b: `float or numpy array`
		Declination or galactic latitude

	Returns
	-------
	phi: `float or numpy array`
		Phi healpy coordinates
	theta: `float or numpy array`
		Theta healpy coordinates
	"""

	theta = np.pi/2-b
	phi  = l

	if(not galCoord): # If equatorial coordinates
		phi -= np.pi
	return phi, theta


def HealpyCoordToMap(galCoord, phi, theta):
	""" Convert Healpy coordiantes into RA/DEC or
	lonGalactic/latGalactic

	Parameters
	----------
	galCoord: `bool`
		If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise
	phi: `float or numpy array`
		Phi healpy coordinates
	theta: `float or numpy array`
		Theta healpy coordinates

	Returns
	-------
	l: `float or numpy array`
		Right Ascension or galactic longitude
	b: `float or numpy array`
		Declination or galactic latitude

	"""

	b = np.pi/2 - theta
	l = phi
	if(not galCoord): # If equatorial coordinates: l+π, projected to [0, 2*π]
		l = np.where(l < np.pi, l + np.pi, l - np.pi)
	return l, b


def top_hat_beam(radius, nside):
	""" Top hat smoothing function to be used using healpy smooth function

	Parameters
	----------
	nside: `int`
		nside parameter for Healpy map
	radius: `float`
		radius in radians for smoothing

	Returns
	-------
	hp.sphtfunc.beam2bl function
	"""
	b = np.linspace(0.0,np.pi,10000)
	bw = np.where(abs(b)<=radius, 1, 0)
	return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space


def fisher_beam(radius, nside):
	""" Fisher smoothing function to be used using healpy smooth function

	Parameters
	----------
	nside: `int`
		nside parameter for Healpy map
	radius: `float`
		radius in radians for smoothing

	Returns
	-------
	hp.sphtfunc.beam2bl function
	"""
	b = np.linspace(0.0,np.pi,10000)
	bw = np.float128(np.exp(np.float128(np.cos(b)/radius**2)))

	return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space


def LoadSmoothedMap(hp_map, radius_deg, nside, smoothing="fisher"):
	""" Perform a smoothing on a healpy map

	Parameters
	----------
	hp_map: `numpy array`
		healpy map
	radius: `float`
		radius in degree for smoothing
	nside: `int`
		nside parameter for Healpy map
	smoothing: `string`
		Function used: top_hat or fisher

	Returns
	-------
	smoothed_map: `numpy array`
		Healpy map smoothed
	"""
	radius = np.radians(radius_deg)
	if smoothing == "fisher": beam_function = fisher_beam
	elif smoothing == "top-hat": beam_function = top_hat_beam

	smoothed_map = hp.smoothing(hp_map, beam_window=beam_function(radius, nside))
	smoothed_map = smoothed_map*np.sum(hp_map)/np.sum(smoothed_map)

	return smoothed_map


def load_Catalog(galCoord=True, Dmin=1, Dmax=350, tracer="logSFR"):
	""" Associate each galaxy with a flux

	Parameters
	----------
	galCoord: `bool`
		if true load galactic coordinates, if not, load equatorial coordinates
	Dmin: `float`
		minimum distance of galaxies in Mpc
	Dmax: `float`
		maximum distance of galaxes in Mpc
	tracer: `string`
		logarithm of the tracer considered (either logSFR or logM* here)

	Returns
	-------
	name: `list`
		name of the galaxy
	dist: `array`
		distance of the galaxy in Mpc
	l or ra: `array`
		longitude or right ascension
	b or dec: `array`
		latitude or declination
	tracer_of_UHECR: `array`
		value of the SFR (M_solar Mpc^-3 yr^-1) or stellar mass (M_solar Mpc^-3), correction included
	"""

	#Load the catalog
	file_Catalog = os.path.join(COMBINED_FIT_BASE_DIR,"../Catalog/light_sfr_cleaned_corrected_cloned_LVMHL.dat")
	t = Table.read(file_Catalog, format='ascii.basic', delimiter=" ", guess=False)

	#Select those in the distance range
	choosen_Galaxies = (t['d']<=Dmax)*(t['d']>=Dmin)
	tsel = t[choosen_Galaxies]
	name = tsel['name']

	#Coordinates
	dist = tsel['d']
	if galCoord: l, b = tsel['glon'], tsel['glat']
	else: l, b = tsel['ra'], tsel['dec']

	#Flux
	if tracer == "logSFR": key_corr = 'cNs'
	else: key_corr = 'cNm'
	tracer_of_UHECR = np.power(10, tsel[tracer])/tsel[key_corr]

	return name, dist, l, b, tracer_of_UHECR


def load_Map_from_Catalog(galCoord, nside, l, b, weights_galaxies):
	""" Produce a array that can be plot using HealPy

	Parameters
	----------
	galCoord: `bool`
		if true load galactic coordinates, if not, load equatorial coordinates
	nside: `int`
		nside parameter for Healpy map
	l: `float`
		galactic longitude or right ascension in deg
	b: `float`
		galactic latitude or declination in deg
	weights_galaxies: `list`
		selected galaxies and associated spectral weights

	Returns
	-------
	Rmean: `float`
		mean rigidity above the threshold, useful for smoothing
	flux_map: `numpy array`
		flux_map in a Healpy format, units are arbitrary
	"""

	# find the pixel for each galaxy
	phi_gal, theta_gal = MapToHealpyCoord(galCoord, np.radians(l), np.radians(b))
	index_gal = hp.ang2pix(nside, theta_gal, phi_gal)
	npix = hp.nside2npix(nside)

	# sum over selections
	flux_maps = []
	cum_sumR, cum_weight = 0, 0
	for wgal in weights_galaxies:
		# load selected galaxies and associated spectral weights
		sel, wflux, cum_weighted_R, cum_weights = wgal
		flux_maps.append(np.histogram(index_gal[sel], bins=np.arange(npix + 1), weights = wflux)[0])
		cum_sumR += np.sum(cum_weighted_R)
		cum_weight += np.sum(cum_weights)

	return cum_sumR/cum_weight, np.sum(np.array(flux_maps), axis=0)


def sel_gal_behind(galCoord, dist0, l0, b0, dist, l, b, R):
	if galCoord: frame="galactic"
	else: frame="icrs"
	c1 = SkyCoord(l0*u.degree, b0*u.degree, frame=frame)
	c2 = SkyCoord(l*u.degree, b*u.degree, frame=frame)
	sep = c1.separation(c2)
	sel =  (sep.radian < np.arctan(R/dist0))*(dist>=dist0)
	return sel

def map_arbitrary_units_all_galaxies(galaxy_parameters, tensor_parameters, k_transient, galCoord, nside):
	name, dist, l, b, lum = galaxy_parameters
	Tensor, E_times_k, A, Z, logRcut, gamma_nucl, gamma_p = tensor_parameters
	weights_galaxies = []

	# select the galaxies beyond 1 Mpc
	sel = dist>1

	# load spectral model for all galaxies
	w_R = lambda Z, logR: sp.Spectrum_Energy(Z, logR, gamma_nucl, logRcut)
	w_R_p = lambda Z, logR: sp.Spectrum_Energy(Z, logR, gamma_p, logRcut)

	# load weights
	logEth, z_tab, weight_z, cum_weighted_R, cum_weights = sp.Compute_single_integrals(Tensor, E_times_k, A, Z, w_R, w_R_p)
	z_tab = np.concatenate(([0],z_tab))
	weight_z = np.concatenate(([weight_z[0]],weight_z))
	def fweight_d(d): return interpolate.interp1d(constant._fz_DL(z_tab), weight_z)(d)

	#return the selection and flux weights
	wflux = fweight_d(dist[sel])* lum[sel]/dist[sel]**2
	weights_galaxies.append([sel, wflux, cum_weighted_R, cum_weights])

	# Load anisotropic map
	Rmean, map_arbitrary_units = load_Map_from_Catalog(galCoord, nside, l, b, weights_galaxies)

	return Rmean, map_arbitrary_units


def map_arbitrary_units_with_all_cuts(galaxy_parameters, tensor_parameters, k_transient, galCoord, nside):

	name, dist, l, b, lum = galaxy_parameters
	Tensor, E_times_k, A, Z, logRcut, gamma_nucl, gamma_p = tensor_parameters
	weights_galaxies = []

	# select the galaxies behind Virgo
	dist0, l0, b0, R500, logM500 = utilities.load_virgo_properties(galCoord)
	sel_Virgo = sel_gal_behind(galCoord, dist0, l0, b0, dist, l, b, 3*R500)
	sel_NonShadowed = np.invert(sel_Virgo)

	# bin galaxies by maximum rigidity
	if k_transient is None: logRcut_galaxies = Tensor[0].logRi[-1]*np.ones_like(dist)#maximum possible rigidity
	else: logRcut_galaxies = utilities.logRcut_transient(k_transient, dist, lum)
	list_logRmax, sel_logRmax = Tensor[0].logR2bin(logRcut_galaxies)

	# load spectral model for all galaxies
	ws_R, ws_R_p = [], []
	for i, sel_lR in enumerate(sel_logRmax):
		ws_R.append(lambda Z, logR: sp.Spectrum_Energy(Z, logR, gamma_nucl, logRcut)*(logR<list_logRmax[i]))
		ws_R_p.append(lambda Z, logR: sp.Spectrum_Energy(Z, logR, gamma_p, logRcut)*(logR<list_logRmax[i]))

	# load weights for non-shadowed galaxies
	for i, sel_lR in enumerate(sel_logRmax):
		# select
		sel = sel_NonShadowed*sel_lR
		if np.sum(sel)>0:
			# define the function returning the weights
			logEth, z_tab, weight_z, cum_weighted_R, cum_weights = sp.Compute_single_integrals(Tensor, E_times_k, A, Z, ws_R[i], ws_R_p[i])
			z_tab = np.concatenate(([0],z_tab))
			weight_z = np.concatenate(([weight_z[0]],weight_z))
			def fweight_d(d): return interpolate.interp1d(constant._fz_DL(z_tab), weight_z)(d)

			#return the selection and flux weights
			wflux = fweight_d(dist[sel])* lum[sel]/dist[sel]**2
			weights_galaxies.append([sel, wflux, cum_weighted_R, cum_weights])

	# load weights for galaxies behind Virgo
	for i, sel_lR in enumerate(sel_logRmax):
		# select
		sel = sel_Virgo*sel_lR
		if np.sum(sel)>0:
			# define the function returning the weights
			w_R = lambda Z, logR: ws_R[i](Z, logR)*utilities.transparency_cluster(logM500, isProton=False)(logR)
			w_R_p = lambda Z, logR: ws_R_p[i](Z, logR)*utilities.transparency_cluster(logM500, isProton=True)(logR)
			logEth, z_tab, weight_z, cum_weighted_R, cum_weights = sp.Compute_single_integrals(Tensor, E_times_k, A, Z, w_R, w_R_p)
			def fweight_d(d): return interpolate.interp1d(constant._fz_DL(z_tab), weight_z)(d)

			#return the selection and flux weights
			wflux = fweight_d(dist[sel])* lum[sel]/dist[sel]**2
			weights_galaxies.append([sel, wflux, cum_weighted_R, cum_weights])

	# Load anisotropic map
	Rmean, map_arbitrary_units = load_Map_from_Catalog(galCoord, nside, l, b, weights_galaxies)

	return Rmean, map_arbitrary_units
