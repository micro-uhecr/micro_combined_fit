from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS, Galactic
import numpy as np
import time
from combined_fit import tensor as ts
from combined_fit import constant
from scipy import interpolate
import healpy as hp
import matplotlib.pyplot as plt



def PlotHPmap(HPmap, nside, galCoord, title, color_bar_title, ax_title, fig_name, plot_zoa=False, write=False, projection="mollweide", cmap='afmhot', vmin=0, vmax=-1):

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
        #cluster_catalog = pd.read_csv("catalog.csv", delimiter = ",")
        #x = np.radians(180.-cluster_catalog['RA'])
        #y = np.radians(cluster_catalog['Dec'])
        #size = cluster_catalog['Mass']/(constant._fz_DL(cluster_catalog['redshift'])**2)
        #size = size / np.max(size)*1000
        #plt.scatter(x, y, marker='o', s=size, color='green', alpha=1)
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

    start_time = time.time()
    idx = ts.find_n_given_z(constant._fDL_z(dist), zmax=zmax)
    weights = np.array([dist-bin_dist[idx], bin_dist[idx+1]-dist])
    den = np.sum(weights, axis=0)

    res = (result[idx, :, :]*weights[1, :, None, None] + result[idx+1, :, :]*weights[0, :, None, None])/den[:, None, None]

    return res

def load_Catalog(galCoord=True, Dmin=1, Dmax=350, tracer="logSFR", fluxmin=0):

    t = Table.read('light_sm_sfr_baryon_od_Resc_Rscreen_merged.dat', format='ascii.basic', delimiter=" ", guess=False)

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
    Rmin = np.log10(tsel['Resc']*1e18) #From EeV to log10(/eV)

    tracer_Of_UHECR = np.power(10, tsel[tracer])
    return dist, l, b, Cn, tracer_Of_UHECR, Rmin

def alpha(Table, A, Z, frac, z_cut, w_zR):
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

        T = frac[i]*np.tensordot(target, weight, axes=([iR[0][0],iz[0][0]],[0,1]))
        iso_T = frac[i]*np.tensordot(iso_target, iso_weight, axes=([iR[0][0],iz[0][0]],[0,1]))

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


    return logE, alpha, np.transpose(alpha_Per_A_inj)

def alpha_factor(z_cut, logEmin, Table, A, Z, frac, w_zR_alpha, zmax):

    zcu = ts.find_n_given_z(z_cut, zmax=zmax)
    zc = zcu + 1
    over, under = Table[0].z[zc], Table[0].z[zcu]
    mini_w = [over-z_cut, z_cut-under]

    logE_alpha_array, alpha_array_over, useless = alpha(Table, A, Z, frac, zc, w_zR_alpha)
    logE_alpha_array, alpha_array_under, useless = alpha(Table, A, Z, frac, zcu, w_zR_alpha)
    alpha_GE_o = interpolate.interp1d(logE_alpha_array, alpha_array_over)
    alpha_GE_u = interpolate.interp1d(logE_alpha_array, alpha_array_under)
    fact_o = alpha_GE_o(logEmin)
    fact_u = alpha_GE_u(logEmin)
    fact = (fact_o*mini_w[1] + fact_u*mini_w[0])/np.sum(mini_w)
    return fact

def MapToHealpyCoord(galCoord, l, b):

    theta = np.pi/2-b
    phi  = l

    if(not galCoord): # If equatorial coordinates
        phi -= np.pi
    return phi, theta

def LoadShapedData(galCoord, Dist, Cn, M_star, l, b):
    '''If galCoord = True, load data in Galactic coordinates. Equatorial coordinates otherwise. Minimum energy in EeV'''
    #Coordinate conversion

    phiData, thetaData = MapToHealpyCoord(galCoord, np.radians(l), np.radians(b))

    dataset = np.asarray([thetaData, phiData, Dist, Cn, M_star])
    return dataset

def LoadlnAMap(dataset, nside, Mass_flux):

    # Pixel Not_Zero_exposureex for each events of coordinates (theta, phi)
    Galaxies = hp.ang2pix(nside, dataset[0], dataset[1])

    # Count map of parameter nside
    npix = hp.nside2npix(nside)# npix corresponds to the number of pixel associated to a NSIDE healpy map

    lnA_map = np.zeros((npix, np.size(Mass_flux[0])))

    for i in range(len(Galaxies)):
        lnA_map[Galaxies[i]] = lnA_map[Galaxies[i]] + Mass_flux[i]/(dataset[2][i]**2*dataset[3][i])*dataset[4][i]

    return lnA_map

def LoadIsolnAMap(nside, iso_result, iso_bin_z, lnA_map, alpha_fact, f_z):
    # Count map of parameter nside
    npix = hp.nside2npix(nside)# npix corresponds to the number of pixel associated to a NSIDE healpy map
    Sz = f_z(iso_bin_z)
    ax = np.where(np.array(iso_result[:, 1, :].shape)==Sz.size)
    var = np.tensordot(iso_result[:, 1, :], Sz, axes=(ax[0],[0]))


    iso_results = [var]*npix
    iso_lnA_map = np.transpose(iso_results)

    int = np.sum(lnA_map)
    iso_int = np.sum(iso_lnA_map)


    if iso_int == 0: #This is a simple patch to override division by zero at high logEmin
        rel = 0
    else:
        param = alpha_fact/(1-alpha_fact)
        coefficient = param*(int/iso_int)
        print(param, int, iso_int, coefficient)
        rel = coefficient*iso_lnA_map

    # rel = iso_lnA_map
    return rel

def top_hat_beam(radius, nside):

    b = np.linspace(0.0,np.pi,10000)
    bw = np.where(abs(b)<=radius, 1, 0)
    return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space

def fisher_beam(radius, nside):
        b = np.linspace(0.0,np.pi,10000)
        bw = np.exp(np.cos(b)/radius**2)
        return hp.sphtfunc.beam2bl(bw, b, lmax=nside*3)#beam in the spherical harmonics space

def LoadSmoothedMap(hp_map, radius_deg, nside, smoothing="fisher"):
    radius = np.radians(radius_deg)
    if smoothing == "fisher":
        beam_function = fisher_beam
    elif smoothing == "top-hat":
        beam_function = top_hat_beam

    if isinstance(hp_map[0],np.ndarray):
        res = [hp.smoothing(hp_map[i], beam_window=beam_function(radius, nside)) for i in range(len(hp_map))]
    else:
        res= hp.smoothing(hp_map, beam_window=beam_function(radius, nside))

    return res


def Compute_integrated_Flux(t, E_times_k, A, Z, w_zR):
    #Computation##################################
    models = []
    for i, a in enumerate(A):
        je = t[i].J_E(t[i].tensor_stacked, w_zR, Z[i])
        models.append([t[i].logE , E_times_k[i]*je])

    total_spectrum = []
    logE = []
    for i, m in enumerate(models):
        logE, je = m[0], m[1]
        total_spectrum.append(je)
    total_spectrum = np.sum(np.array(total_spectrum))

    return total_spectrum
