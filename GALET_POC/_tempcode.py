import numba as nbimport numpy as np
@nb.njit()
def func1(a):
    return a**4

@nb.njit()
def func2(a):
    return a*45

