import numpy as np
from astropy.table import Table

def LoadTable(a, zmax, prefix):
	"Load data from intermediate table"
	filename = prefix+"_A_"+str(a)+"_zmax_"+str(zmax)+".dat"
	t = Table.read(filename, format='ascii.basic', delimiter=" ", guess=True)
	return t

def zbins(zmin = 0, zmax = 2.5, dz_min = 1E-4, dz_max = 0.5):
	dz = lambda z: dz_min + dz_max*(z-zmin)/(zmax-zmin)
	list_z = [zmin]
	while list_z[-1]<=zmax-dz_max: list_z.append(list_z[-1]+dz(list_z[-1]))
	list_z.append(zmax)
	return np.array(list_z)
	
def load_primary_info(t):
	u, indices = np.unique(t['id'], return_index=True)
	tunique = t[indices]	
	n_entries = tunique['wi'].sum()
	ZA = np.unique(np.array(t['Z','A']).T)
	
	return n_entries, ZA

def write_tensor(a, zmax, prefix_input, prefix_output, bins_z, bins_logE, bins_logRi, bin_width_logE, write, verbose=False):
	print("write_tensor: If the computer gets stuck, it means that you're lacking of RAM. you need to reduce \"precisionR\" in LoadInput.C")

	t = LoadTable(a, zmax,prefix_input)
	lRmin, lRmax = np.min(t['logRi']), np.max(t['logRi'])

	n_entries, ZA = load_primary_info(t)
	print("Number of Z/A bins: ", ZA.size)

	#Mean number of entry per bin
	print("Average entries per bin:", np.round(n_entries/(ZA.size*logRi.size*logE.size*z.size)))

	if(verbose):
		print(t.info)
		for k in t.colnames:
			if(k!='id'): print(k, np.min(t[k]), np.max(t[k]))
		print(n_entries)
		print(ZA)

	print("------------ A =",a,"------------")

	#Fill the tensor
	logRi_size, z_size, logE_size = bins_logRi.size-1, bins_z.size-1, bins_logE.size-1
	tensor = np.zeros((ZA.size, logRi_size, z_size, logE_size))#(Z,A), z, E
	for iz in range(z_size):
		print("Redshift: ",bins_z[iz], bins_z[iz+1])

		sel = ( t['z']>bins_z[iz] )*( t['z']<=bins_z[iz+1] )
		tz = t[sel]

		for iR in range(logRi_size):
			sel = ( tz['logRi']>bins_logRi[iR] )*( tz['logRi']<=bins_logRi[iR+1] )
			tR = tz[sel]

			u, indices = np.unique(tR['id'], return_index=True)
			tunique = tR[indices]
			norm = tunique['wi'].sum()

			for iE in range(logE_size):

				sel = ( tR['logE']>bins_logE[iE] )*( tR['logE']<=bins_logE[iE+1] )
				tE = tR[sel]

				for iZA in range(ZA.size):
					sel = ( tE['Z']== ZA[iZA][0])*( tE['A']==ZA[iZA][1] )
					tsel = tE[sel]
					if(len(tsel)>0):
						tensor[iZA][iR][iz][iE] = 1.0*np.sum(tsel['w'])/norm

	if(write):
		filename_out = prefix_output+"A_"+str(a)+"_z_" +str(zmax)
		np.savez(filename_out, logRi = logRi, ZA=ZA, z=z, logE=logE, tensor = tensor)

	return tensor

	
### Main ##########################################################
if __name__ == "__main__":
	prefix_input = "Intermediate_MergedInput/SimProp"#input file
	logRmin, logRmax = 17, 21#determined by input file - see Intermediate_MergedInput/*.pdf
	zmax = 2.5#determined by input file - see Intermediate_MergedInput/*.pdf
	prefix_output = "Output_Tensor/SimProp"

	#parameters for energy & rigidity binning
	bin_width_logR = 0.1#rigidity binning at source
	bin_width_logE = 0.1#energy binning on Earth
	
	#define redshift bins
	bins_z = zbins(0, zmax)
	z = 0.5*(bins_z[1:]+bins_z[:-1])
	print("Number of redshift bins: ", z.size)

	#define gen rigidity bins
	bins_logRi = np.arange(logRmin, logRmax, bin_width_logR)
	logRi = np.round(100*0.5*(bins_logRi[1:]+bins_logRi[:-1]))/100.
	print("Number of Ri bins: ", logRi.size)
	
	#define energy bins
	bins_logE = np.arange(logRmin+1.5, logRmax+0,bin_width_logE)#Note: log10(Z_Fe) ~ 1.4 hence the +1.5, log10(Z_H) = 0 hence the +0
	logE = np.round(100*0.5*(bins_logE[1:]+bins_logE[:-1]))/100.
	print("Number of energy bins: ", logE.size)

	print("Better to have different numbers of bins along all axes for axis identification!")

	#load the table
	#A = [1, 4, 12, 14, 16, 20, 24, 28, 32, 56]# for all species
	A = [12]# for a single species
	for a in A:	tensor = write_tensor(a, zmax, prefix_input, prefix_output, bins_z, bins_logE, bins_logRi, bin_width_logE, write=True)

