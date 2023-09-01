""" Unit test for the tools module
"""
import numpy as np
import sys

from combined_fit import tensor as ts

def test_Tensor():
    '''
    check if the tensor is correctly filled
    '''
    t = ts.upload_Tensor()
    assert np.size(t) != 0 #

