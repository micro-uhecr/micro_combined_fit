import os
import numpy as np
import numba as nb
import pandas as pd
import pathlib
from scipy import interpolate

from combined_fit import constant


# This gives the combined_fit/combined_fit directory
COMBINED_FIT_BASE_DIR = pathlib.Path(__file__).parent.resolve()

A    = [    1,     4,    14,    28,    56]
Z    = [    1,     2,     7,    14,    26]


#@nb.jit
def upload_Tensor():
    '''Function which uploads the tensor

    Parameters
    ----------
    None

    Returns
    -------
    Tensor: 'Table'
        Table which contains all the information about the tensor
    '''
    # zmax = 1 or 2.5 to load the associated tensor
    zmin, zmax = 0.0, 1
    Tensor =[]
    for i,a in enumerate(A):
        filename = os.path.join(COMBINED_FIT_BASE_DIR, '../Tensor/A_'+str(A[i])+'_z_'+str(zmax)+'.npz')
        Tensor.append(CR_Table(filename, zmin, zmax))
    return Tensor


class CR_Table:
    ''' `CR_Table` is a class which allows to deal with the tensor
    '''
    def __init__(self, filename, zmin, zmax, logEmin=18):
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
        self.logRi = t['logRi']
        self.z = t['z']
        self.logE = t['logE']
        self.ZA = t['ZA']
        weights =  (self.logE>=logEmin)
        self.tensor = np.zeros_like(t['tensor'])
        self.tensor[:,:,:,weights] = t['tensor'][:,:,:,weights]

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
        uA = np.unique(za['A'])
        A_list, stacked_A = [], []
        for a in uA:
            sel = np.where(za['A'] == a)
            stacked_A.append(np.sum(self.tensor[sel], axis=0))
            A_list.append(a)
        self.tensor_stacked_A = np.array(stacked_A)
        self.A = np.array(A_list)

		#Stacked over all Z
        uZ = np.unique(za['Z'])
        Z_list, stacked_Z = [], []
        for z in uZ:
            sel = np.where(za['Z'] == z)
            stacked_Z.append(np.sum(self.tensor[sel], axis=0))
            Z_list.append(z)
        self.tensor_stacked_Z = np.array(stacked_Z)
        self.Z = np.array(Z_list)

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

    def J_E(self, a, w_zR, ZA):
        ''' Create the expected spectrum given the parameters at the source

        Parameters
        ----------
        a : `tensor`
            Associated tensor
        w_zR : `list`
            weights in Rigidity
        ZA : `list`
            charge of the injected particle A

        Returns
        -------
        stacked_logR_z : `tensor`
            a stacked tensor?
        '''
        zz, rr = np.meshgrid(self.z, self.logRi)
        stacked_logR_z = self.sum_logR_z(a, w_zR(ZA, zz,rr)*self.delta_z)

        return stacked_logR_z


def Load_evol():
    ''' Load the chosen evolution of the source

    Parameters
    ----------
    None

    Returns
    -------
    f_z : `function`
        a function that describes the source evolution
    '''
    file_sfrd = os.path.join(COMBINED_FIT_BASE_DIR, "../Catalog/sfrd_local.dat")
    tz = pd.read_csv(file_sfrd, delimiter = " ")
    tz['sfrd'] *= (1/constant._Mpc_2_km)**2
    f_z = interpolate.interp1d( tz['z'], tz['sfrd'])

    return f_z
