import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from astropy.table import Table
from astropy.io import ascii

from combined_fit import constant, draw

def Write_Draw_evolution(evolution, local_dens, annotations=None, deep_field_data=None, title_y=None, ymin=None, ymax=None, dL_min = 0.5, dL_max = 8.E4, output_write =None):
	''' Plot the evolution

	Parameters
	----------
	evolution : `function`
		function of luminosity distance 
	local_dens : `tuple of list`
		luminosity distance and density
	annotations : `list of tuples`
		coordinates and text for Local Structures
	deep_field_data: `list`
		zmin, zmax, dens, edens
	ymin: `float`
		minimum of y-axis
	ymax: `float`
		maximum of y-axis
	output_write: `float`
		name used to write the local density when not None
			
	Returns
	-------
	None
	'''
	#Load local and cosmic density
	ld, ldens = local_dens
	ildmin = np.argmax(ld>1)#turns off everything below 1 Mpc

	cd = np.logspace(np.log10(dL_min), np.log10(dL_max),100)
	cdens = [evolution(d) for d in cd]
	icdmin = np.argmax(cd>ld[-1])
	
	#Write tables
	if output_write!=None:
		z, dens = [0], [0]
		 #local density
		for i in range(ildmin, len(ld)):
			z.append(constant._fDL_z(ld[i]))
			dens.append(ldens[i]) 
		 #cosmic density
		for i in range(icdmin, len(cd)):
			z.append(constant._fDL_z(cd[i]))
			dens.append(cdens[i]) 
			
		table_evol = Table([z, dens], names = ('z',output_write))
		ascii.write(table_evol, output_write+'_local.dat', overwrite=True)
 	
	#Plot setup
	fig = plt.subplots(figsize=(6, 4), nrows=1, ncols = 1)
	plt.subplots_adjust(bottom = 0.15, top = 0.92, left=0.15, right=0.96)
	
	ax0 = plt.subplot(111)
	ax0.tick_params(top=True, right=True)
	ax0.set_xlabel(r'Luminosity distance, $d_{\rm L}$  [Mpc]')
	ax0.set_ylabel(title_y)
	ax0.set_xlim(dL_min, dL_max)
	if ymin!=None or ymax!=None: ax0.set_ylim(ymin, ymax)
	ax0.set_xscale('log')
	ax0.set_yscale('log')
	ax0.text(0., 1.045, r'Redshift, $z$', transform=ax0.transAxes)	
	
	ax1 = ax0.twiny()
	ax1.set_xscale('log')
	ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1.set_xlim(ax0.get_xlim())
	ax1.tick_params(which='minor', length=0, color='k')
	z_ticks = [0.01, 0.1, 1, 10]
	ax1.set_xticks(constant._fz_DL(np.array(z_ticks)))
	ax1.set_xticklabels(list(z_ticks))

	#Local density
	ax0.plot(ld, ldens, color="tab:gray", ls="-.")	
	ax0.plot(ld[ildmin:], ldens[ildmin:], color="k", label="Full-sky (Biteau '21 - 2MASS/WISE/SCOS)")
	
	#Deep field
	sz, sbounds = deep_field_data
	Lplot = np.logspace(np.log10(constant._fz_DL(sz[0])),np.log10(constant._fz_DL(sz[1])))
	ax0.fill_between(Lplot,sbounds[0]*np.ones_like(Lplot), sbounds[1]*np.ones_like(Lplot), label = "Deep-field (Driver+ '18 - GAMA/COSMOS/HST)", color='tab:green', alpha = 0.5, linewidth=0)
			
	#Evolution
	icdtmp = np.argmax(cd>1)
	ax0.plot(cd[icdtmp:icdmin], cdens[icdtmp:icdmin], color='tab:gray', ls=":")
	ax0.plot(cd[icdmin:], cdens[icdmin:], color='k', ls="--", label="Cosmic evolution (Koushan+ '21, López+ '18)")

	#Annotations
	for annot in annotations:
		text, xy, xytext = annot
		ax0.annotate(text, fontsize=12, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-", color="tab:blue"), ha='left', color="tab:blue", verticalalignment='bottom')	

	ax0.legend()
	
	#Write
	if output_write!=None:
		draw.MySaveFig(fig[0],"evolution_"+output_write)
		
def SFRD(age, mSFR, mPeak, mPeriod, mSkew):
	''' SFRD parametrization from 2021MNRAS.503.2033K

	Parameters
	----------
	age : `list`
		look-back time
	others : `floats`
		see 2021MNRAS.503.2033K
		
	Returns
	-------
	Cosmic SFR density
	'''
	x = (age-mPeak)/mPeriod
	X = x*np.exp(mSkew*np.arcsinh(x))
	
	return mSFR*np.exp(-0.5*X**2)

def Dens(t, dmin, dmax, key, key_corr, Dmax, Dmin=50E-3):
	''' Smooth the density from a catalog

	Parameters
	----------
	t : `astropy table`
		catalog of galaxies
	dmin : `float`
		minimum distance for distance estimation
	dmax : `float`
		maximum distance for distance estimation		
	key: `string`
		tracer considered
	key_corr: `string`
		completeness correction factor to consider
	Dmax: `float`
		maximum distance of the catalog
	Dmin: `float`
		minimum distance of the catalog		
		
	Returns
	-------
	[d, dens] : `list pair`
		distance and density of the tracer
	'''
	if(dmax>Dmax): dmax=Dmax
	if(dmin<Dmin): dmin=Dmin

	#Volume	
	dcmin = dmin/(1.+constant._fDL_z(dmin))
	dcmax = dmax/(1.+constant._fDL_z(dmax))
	V = 4*np.pi*(dcmax**3-dcmin**3)/3#Mpc/3

	#Observed density
	sel = (t['d']>=dmin)*(t['d']<dmax)
	sfr = np.power(10,t[sel][key])/t[sel][key_corr]#M_\odot
	
	#Flux weighted distance
	w = sfr*t[sel]['d']**2
	dmean = np.sum(w*t[sel]['d'])/np.sum(w)
	
	return [dmean, np.sum(sfr)/V]

def smoothed_Dens(t, Dmax=350, f_Omega=1, key='logSFR', key_corr='cNs', npoints=100):
	''' Smooth the density from a catalog

	Parameters
	----------
	t : `astropy table`
		catalog of galaxies
	f_Omega : `float`
		fraction of solid angle of the sphere considered
	key: `string`
		tracer considered
	key_corr: `string`
		completeness correction factor to consider
	npoints: `int`
		number of points of the output
		
	Returns
	-------
	d, dens_out : `list tuple`
		distance and density of the tracer
	'''
	dAnd, dLMC = 0.780, 0.050#Andromeda and LMC distance in Mpc
	dex_dmin, dex_dmax = 0.05*np.log10(dAnd/dLMC), 0.05#variable smoothing width
	log10_dmin, log10_dmax = 0.5*np.log10(dLMC*dAnd), np.log10(350)#0.05dex (12% at 350 Mpc)#0.05dex (12% at 350 Mpc = photometric redshift resolution)
	d_centers = np.logspace(log10_dmin, np.log10(Dmax),npoints)
	d_dex = dex_dmin + (dex_dmax-dex_dmin)*(np.log10(d_centers)-log10_dmin)/(log10_dmax-log10_dmin)
	d_dex[d_centers>350] = dex_dmax
	d_low = d_centers*np.power(10.,-d_dex)
	d_up = d_centers*np.power(10.,+d_dex)

	dens = [Dens(t, d_low[i], d_up[i], key, key_corr, Dmax) for i in range(d_centers.size)]
	dens = np.array(dens).T
	selec = (dens[1]>0)*(dens[0]!=np.nan)
	d, dens_out = dens[0][selec], dens[1][selec]/(f_Omega)

	return d, dens_out

### Main ##########################################################
if __name__ == "__main__":

	################################# Inputs ##################################
	###########################################################################
	
	#Large-scale evolution
	#mSFR, mPeak, mPeriod, mSkew = 0.110, 9.594, 2.010, 0.334#LAT initial from Table 5 in arXiv:2102.12323
	mSFR, mPeak, mPeriod, mSkew = 0.066, 10.378, 2.286, 0.364#Lopez Fernandez '18 final from Table 5 in arXiv:2102.12323	
	fun_sfrd = lambda age: SFRD(age, mSFR, mPeak, mPeriod, mSkew)#M* yr-1 Mpc-3

	sfrd = lambda dL: fun_sfrd(constant._fDL_t(dL))#M* yr-1 Mpc-3
	title_sfrd = r'SFR density  [$M_\odot\,$Mpc$^{-3}\,$yr$^{-1}$]'

	R = 0.41#for a Chabrier IMF - 2014ARA&A..52..415M
	smd = lambda dL: (1-R)*1E9*integrate.quad(fun_sfrd, constant._fDL_t(dL), constant._fz_t(10))[0]#M* Mpc-3  (1E9 = Gyr yr-1)
	title_smd = r'$M_{\star}$ density  [$M_\odot\,$Mpc$^{-3}$]'	

	#GAMA local measurements - 2018MNRAS.475.2891D
	sz = [0.02, 0.08]
	
	r, er = 8.30, np.sqrt(2*0.01**2+0.08**2)#reference CSM
	sm_stat = [np.power(10,r-er), np.power(10,r+er)]
	deep_field_sm = [sz, sm_stat]

	s, es = -1.95, np.sqrt(0.03**2+0.07**2)#reference CSFR
	sf_stat = [np.power(10,s-es), np.power(10,s+es)]
	deep_field_sfr = [sz, sf_stat]	
	
	#Catalogue - 2021ApJS..256...15B
	filename = "light_sfr_cleaned_corrected_cloned_LVMHL.dat"
	table = Table.read(filename, format='ascii.basic', delimiter=" ", guess=False)
	local_sfr = smoothed_Dens(table, key="logSFR", key_corr='cNs')
	local_sm = smoothed_Dens(table, key="logM*", key_corr='cNm')	
	
	#Annotations
	entries= [	[0.75, "Andromeda"],
				[4, "Local Sheet"],
				[16, "Virgo cluster"],
				[65, "Laniakea supercluster"]]

	fact_x, fact_y = 2.3, 2
	ymin_sm = 0.3E7
	ymin_sfr = 0.1E-3	
	annot_sfr, annot_sm = [], []
	for i, e in enumerate(entries):
		d, text = e
		annot_sm.append([text, (d, ymin_sm*70), (d*fact_x, 1.2*ymin_sm*np.power(fact_y, i))])
		annot_sfr.append([text, (d, ymin_sfr*70), (d*fact_x, 1.2*ymin_sfr*np.power(fact_y, i))])		
		
	################################## Plot ###################################
	###########################################################################

	plt.rcParams.update({'font.size': 14,'legend.fontsize': 12})
	
	#Plot SFRD
#	Write_Draw_evolution(sfrd, title_y=title_sfrd, local_dens = local_sfr, annotations = annot_sfr, deep_field_data = deep_field_sfr, ymin = ymin_sfr, ymax = ymin_sfr*1E4, output_write="sfrd")

	#Plot SMD
	print(np.log10(smd(0)))
#	Write_Draw_evolution(smd, title_y=title_smd, local_dens = local_sm, annotations = annot_sm, deep_field_data = deep_field_sm, ymin = ymin_sm, ymax = ymin_sm*1E4, output_write="smd")
		
	plt.show()
