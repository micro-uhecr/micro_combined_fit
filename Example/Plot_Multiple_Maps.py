from combined_fit import map
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

hp.disable_warnings()

### Main ##########################################################
if __name__ == "__main__":

	############ Parameters ####################
	NMaps = 100
	galCoord = False
	#folder = "/Users/adminlocal/Documents/micro_combined_fit/Example/Results_CCA/test2_SFR_TALYS_LogEmin_19.6_k_1e-05_radius_12.6_B_LGMF_nG_10_B_IGMF_nG_0.0001_smoothing_fisher_GC_False_fluxmin_0.0/"
	folder = "../Simulations/SFR_TALYS_LogEmin_19.6_k_1e-05_radius_12.6_B_LGMF_nG_25_B_IGMF_nG_0.0001_smoothing_fisher_GC_False_fluxmin_0_Radius_Cluster_2_Mpc/"
	#folder = "../../Maps_Sullivan/Results_Corrected/SFR_TALYS_LogEmin_19.6_k_1e-05_radius_12.6_B_LGMF_nG_25_B_IGMF_nG_0.0001_smoothing_fisher_GC_False_fluxmin_0_Radius_Cluster_2_Mpc/"
	#folder = "/Users/adminlocal/Documents/Maps_Sullivan/Results_Corrected/SFRD_TALYS_LogEmin_19.6_k_1e-05_radius_12.6_B_LGMF_nG_10_B_IGMF_nG_0.0001_smoothing_fisher_GC_False_fluxmin_0_Radius_Cluster_2_Mpc/"
	file_Flux = folder + "Flux_map_"
	file_Mass = folder + "Mass_map_"
	nside = 64

	int_flux = 0.013036408672982306 # Normalization factor (int_flux = ts.Compute_integrated_Flux(Tensor, E_times_k, Z, w_zR_Background))

	############ Computation ####################
	maps_lnA = []
	maps_Flux = []
	for i in range(NMaps):
		maps_Flux.append(hp.fitsfunc.read_map(file_Flux + str(i) + ".fits"))
		maps_lnA.append(hp.fitsfunc.read_map(file_Mass + str(i) + ".fits"))

	selected_Flux_map = np.median(maps_Flux, axis=0)
	selected_Flux_map = selected_Flux_map / np.sum(selected_Flux_map/hp.nside2npix(nside)) * int_flux
	selected_Mass_map = np.median(maps_lnA, axis=0)


	Sorted_Flux_map = np.sort(maps_Flux, axis=0)

	Sigma_Flux_map = np.mean(Sorted_Flux_map[int(NMaps/100*16):int(NMaps/100*84)], axis=0)
	STD_map = (Sigma_Flux_map-selected_Flux_map)/selected_Flux_map*100

	nside = hp.pixelfunc.get_nside(maps_Flux[0])

	############ Plot ####################

	color_bar_title = "<lnA>"
	if galCoord:
		ax_title = "Galactic coordinates"
	else:
		ax_title = "Equatorial coordinates"
	ax_title = ax_title + " - Magnetic confinement in Virgo included"
	fig_name = "<lnA>"

	title = "Composition map"
	print("Min lnA : " , np.min(selected_Mass_map), "Max lnA : " , np.max(selected_Mass_map))
	map.PlotHPmap(selected_Mass_map, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, projection="hammer", cmap='afmhot', vmin=-1, vmax=-1)

	print("Min flux : " , np.min(selected_Flux_map), "Max flux : " , np.max(selected_Flux_map))
	title = r"Median Map, $\Phi(\log_{10} (E/{\rm eV}) >19.6)$ - Fisher smoothing with 12.6° radius"
	color_bar_title = r"Flux $[\rm km^{-2} \, yr^{-1} \, sr^{-1}]$"
	map.PlotHPmap(selected_Flux_map, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, cmap='afmhot', vmin=0.003, vmax=0.037)

	title = r"Mean deviation from the median for 68% of the realisation"
	color_bar_title = "Flux [%]"
	max_value = np.max(np.abs(STD_map))
	map.PlotHPmap(STD_map, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, cmap='twilight_shifted', vmin=-max_value, vmax=max_value)

	plt.show()
