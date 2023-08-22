import os
import pathlib
import numba as nb

import numpy as np
import pandas as pd
from scipy import interpolate

# This gives the combined_fit/combined_fit directory
COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

A    = [    1,     4,    14,    28,    56]
Z    = [    1,     2,     7,    14,    26]

#@nb.jit
def upload_Tensor(logRmin=1, logEmin=1, is_PSB=False):
    '''Function which uploads the tensor

    Parameters
    ----------
    None

    Returns
    -------
    Tensor: 'Table'
        Table which contains all the information about the tensor
    '''
    
    if logEmin<logRmin:
        print("Minimum energy above which flux is computed is below logRmin")
        print("Setting Emin to Rmin:", logRmin)        
        logEmin=logRmin
          
    # zmax = 1 or 2.5 to load the associated tensor
    zmin, zmax = 0.0, 1
    Tensor =[]
    for i,a in enumerate(A):
        ext = 'TALYS_'
        if is_PSB: ext =''
        filename = os.path.join(COMBINED_FIT_BASE_DIR, '../Tensor/A_'+ext+str(A[i])+'_z_'+str(zmax)+'.npz')
        Tensor.append(CR_Table(filename, zmin, zmax, logEmin=logEmin, logRmin=logRmin))
    return Tensor


class CR_Table:
    ''' `CR_Table` is a class which allows to deal with the tensor
    '''
    def __init__(self, filename, zmin, zmax, logEmin=1, logRmin=1):
        ''' `CR_Table` constructor

        Parameters
        ----------
        filename : `string`
            path of the tensor
        zmin, zmax : `float`
            range in energy
        logEmin: `float`
            minimal (injected) energy below which we do not read the tensor

         Returns
         -------
         None
         '''
        t = np.load(filename)
        #Load variables from npz file
        ind = find_nearest_Ri_bin(t['logE'], [logEmin,0])[0]+1
        ind_Ri = np.max((find_nearest_Ri_bin(t['logRi'], [logRmin,0])[0]-1, 0))

        self.logRi = t['logRi'][ind_Ri:]
        self.z = t['z']
        self.logE = t['logE'][ind:]
        self.ZA = t['ZA']
        self.tensor = t['tensor'][:,ind_Ri:,:,ind:]


        #Define redshift bins for integration
        bins_z = []
        bins_z.append(zmin)
        for z in self.z:
            bins_z.append(2*z-bins_z[-1])
        self.delta_z = (np.array(bins_z[1:])-np.array(bins_z[:-1]))

        #Create Z and A variables
        za = np.array(self.ZA, dtype=np.dtype('int,int'))
        za.dtype.names = ['Z','A']

        #Stacked over all species
        self.tensor_stacked = np.sum(self.tensor, axis=0)

        #Stacked over all A
        uA, idx = np.unique(za['A'], return_index=True)
        A, stacked_A = [], []
        for a in uA[idx]:
            sel = np.where(za['A'] == a)
            stacked_A.append(np.sum(self.tensor[sel], axis=0))
            A.append(a)
        self.tensor_stacked_A = np.array(stacked_A)
        self.A = np.array((A))


        #Stacked over all Z
        uZ = np.unique(za['Z'])
        Z_list, stacked_Z = [], []
        for z in uZ:
            sel = np.where(za['Z'] == z)
            stacked_Z.append(np.sum(self.tensor[sel], axis=0))
            Z_list.append(z)
        self.tensor_stacked_Z = np.array(stacked_Z)
        self.Z = np.array(Z_list)

    def logR2bin(self, logR):
        ''' Group logRi entries by logR bin

        Parameters
        ----------
        logRi: `array`
            list of rigidities

        Returns
        -------
        logR : `array`
            center of the rigidity bins of the tensor
        list_sel  : `list`
            selected entries in the bin
        '''    
        dlR = 0.5*(self.logRi[1]- self.logRi[0])
        list_sel = [(logR>lRi-dlR)*(logR<=lRi+dlR) for lRi in self.logRi]
        
        return self.logRi, list_sel
    
    def sum_logR_z(self, a, weights=[]):
        ''' Contracting the tensor in z

        Parameters
        ----------
        self: `object`
            the `CR_Table` object
        a : `tensor`
            Associated tensor
        weights : `list`
            weights in znp.

        Returns
        -------
        res : `tensor`
            the numpy tensor with all the information
        '''
        iz = np.where(np.array(a.shape) == self.z.size)
        iR = np.where(np.array(a.shape) == self.logRi.size)
        if((len(iz[0])==1) and (len(iR[0])==1)):
            res = 0
            if len(weights)>0:
                res = np.tensordot(a, weights, axes=([iR[0][0],iz[0][0]],[0,1]))
            else: res = np.sum(a, axis=[iz[0],iR[0]])
            return res
        else:
            print("logR or z axis not found")
            return -1

    def sum_logR_Given_z(self, a, weights=[]):
        iz = np.where(np.array(a.shape) == self.z.size)
        iR = np.where(np.array(a.shape) == self.logRi.size)

        if((len(iz[0])==1) and (len(iR[0])==1)):
            res = 0
            if(len(weights)>0):
                res = np.tensordot(a, weights, axes=([iR[0][0]],[0]))
            else:
                res = np.sum(a, axis=[iR[0]])
            return res
        else:
            print("logR or z axis not found")


    def j_zE(self, a, w_R, ZA):
        ''' Create the expected spectrum given the parameters at a single source

        Parameters
        ----------
        a : `tensor`
            Associated tensor
        w_R : `list`
            weights in Rigidity
        ZA : `list`
            charge of the injected particle A

        Returns
        -------
        z : `array`
            redshifts
        stacked_logR : `tensor`
            flux for a single galaxy prop to km-2 sr-1 yr-1 eV-1
        '''

        stacked_logR = self.sum_logR_Given_z(a, w_R(ZA, self.logRi))

        return stacked_logR
        
    def J_E(self, a, w_zR, ZA):
        ''' Create the expected spectrum given the parameters at the source

        Parameters
        ----------
        a : `tensor`
            Associated tensor
        w_zR : `list`
            weights in Rigidity and redshift
        ZA : `list`
            charge of the injected particle A

        Returns
        -------
        stacked_logR_z : `tensor`
            flux in km-2 sr-1 yr-1 eV-1
        '''
        zz, rr = np.meshgrid(self.z, self.logRi)
        stacked_logR_z = self.sum_logR_z(a, w_zR(ZA, zz,rr)*self.delta_z)

        return stacked_logR_z

#TBD fix to make it look nicer
def Load_evol(file="sfrd_local.dat", zmin=2.33e-4, key ="sfrd"):#Minimum redshift of ~2E-4 corresponds to 1 Mpc
    ''' Load the chosen evolution of the source

    Parameters
    ----------
    file : `string`
        name of the file in the folder Catalog which gives the evolution of sources
    zmin : `float`
        minimum distance for the evolution of source
    key : `string`
        Name of the column inside the file

    Returns
    -------
    f_z : `function`
        a f
    '''

    file = os.path.join(COMBINED_FIT_BASE_DIR, "../Catalog/" + file)
    tz = pd.read_csv(file, delimiter = " ")

    if zmin>1e-10:
        select = np.where(tz['z']<zmin)[0]

        dist_up = tz["z"][select[-1]+1]-zmin
        dist_down = zmin-tz[key][select[-1]]

        value = (tz[key][select[-1]+1]* dist_down + tz[key][select[-1]]*dist_up)/(dist_up + dist_down)
        tz[key][select] = 0

        dist = np.append(np.array(tz['z']), zmin)
        values = np.append(np.array(tz[key]), value)

        dist = np.append(dist, 0)
        values = np.append(values, 0)
    else:
        dist = np.append(np.array(tz['z']), tz['z'][0]-1e-10)
        values = np.append(np.array(tz[key]), 0)
        dist = np.append(dist, 0)
        values = np.append(values, 0)


    f_z = interpolate.interp1d( dist, values)

    return f_z

#TBD check if really needed
def find_nearest_Ri_bin(array, value):
    ''' Give the nearest bin in array below the value for linear binning

    Parameters
    ----------
    array : `array`
        array considered
    value : `float`

    Returns
    -------
    idx : `int`
        The index of the array
    '''

    idx = ((value-array[0])/(array[1]-array[0]))//1
    wrg = np.where(idx<0)[0]
    idx[wrg] = 0
    wrg = np.where(idx>=len(array))[0]
    idx[wrg] = len(array)-2
    idx = idx.astype(int)
    return idx
