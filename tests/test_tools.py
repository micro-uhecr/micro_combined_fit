""" Unit test for the tools module
"""
import numpy as np
import sys

from combined_fit import tensor as ts
from combined_fit import xmax_distr
from combined_fit import gumbel


def test_get_gumbel(A = 1, lgE = 19, model = "EPOS-LHC" ):
    """ Verify we get normalized gumbel
     """
    g = gumbel.Gumbel_function(A, lgE, model)
    assert  np.round(np.sum(g.y)*20) == 1  # summing the values of the gumbel function per 20 (number of bins)

def test_Tensor():
    '''
    check if the tensor is correctly filled
    '''
    t = ts.upload_Tensor()
    assert np.size(t) != 0 #

def test_Xmax_distr():
    '''
    check if the tensor is correctly filled
    '''
    t, arr = xmax_distr.set_Xmax_distr()
    assert np.size(t) != 0  #
