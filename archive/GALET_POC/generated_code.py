
'''
This code has been automatically generated by GALET_MF:

Here is a small relaxing view before the struggle!

                                   /\
                              /\  //\\
                       /\    //\\///\\\        /\
                      //\\  ///\////\\\\  /\  //\\
         /\          /  ^ \/^ ^/^  ^  ^ \/^ \/  ^ \
        / ^\    /\  / ^   /  ^/ ^ ^ ^   ^\ ^/  ^^  \
       /^   \  / ^\/ ^ ^   ^ / ^  ^    ^  \/ ^   ^  \       *
      /  ^ ^ \/^  ^\ ^ ^ ^   ^  ^   ^   ____  ^   ^  \     /|\
     / ^ ^  ^ \ ^  _\___________________|  |_____^ ^  \   /||o\
    / ^^  ^ ^ ^\  /______________________________\ ^ ^ \ /|o|||\
   /  ^  ^^ ^ ^  /________________________________\  ^  /|||||o|\
  /^ ^  ^ ^^  ^    ||___|___||||||||||||___|__|||      /||o||||||\       |
 / ^   ^   ^    ^  ||___|___||||||||||||___|__|||          | |           |
/ ^ ^ ^  ^  ^  ^   ||||||||||||||||||||||||||||||oooooooooo| |ooooooo  |
ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo


'''

import numba as nb
import numpy as np
import math


@nb.njit()
def _INTERNAL_param_0_cellarea(a):
    return a


@nb.njit()
def _INTERNAL_param_1_meq(a):
    return a


def _INTERNAL_param_2_neq(a):
    return a


def _INTERNAL_param_3_Kr(i, a):
    return a[i]


@nb.njit()
def _INTERNAL_process_0_drainage_area(i, receivers, cellarea, Qw, Qw_split):
    Qw[i] += cellarea
    for r in range(receivers.shape[0]):
        Qw_split[i,r] += Qw[i]/receivers.shape[0]


@nb.njit()
def _INTERNAL_process_1_SPL(i, receivers, cellarea, Qw, Qw_split):
    Qw[i] += cellarea
    for r in range(receivers.shape[0]):
        Qw_split[i,r] += Qw[i]/receivers.shape[0]


# --------------------------------------------- 
# --------------------------------------------- 
# ------------ MAIN FUNCTIONS BELOW ----------- 
# --------------------------------------------- 
# --------------------------------------------- 




@nb.njit()
def _internal_run(n_elements, 
quantity_int0D,quantity_int1D,quantity_int2D,quantity_int3D,
quantity_float0D,quantity_float1D,quantity_float2D,quantity_float3D,
D8Srec,D8Sdist,D8Sdons,D8Sndons,D8Mrecs,D8Mnrecs,D8Mdist,D8Mdons,D8Mndons,D8Mdondist,SStack,MStack):

	for i in range(n_elements):

# Writing the process 0_drainage_area
		_INTERNAL_process_0_drainage_area(i,
			D8Mrecs[i,:D8Mnrecs[i]],
			_INTERNAL_param_0_cellarea(None,quantity_float0D[0])
,
			quantity_float1D[2],
			quantity_float2D[0])

# Writing the process 1_SPL
		_INTERNAL_process_1_SPL(i,
			D8Mrecs[i,:D8Mnrecs[i]],
			_INTERNAL_param_0_cellarea(None,quantity_float0D[0])
,
			quantity_float1D[2],
			quantity_float2D[0])

		for neight in range(D8Mnrecs[i]):
			rec = D8Mrecs[neight]
			quantity_float1D[2,rec] += quantity_float2D[0,i,neight]
@nb.njit()
def _finalise_step(n_elements, 
quantity_int0D,quantity_int1D,quantity_int2D,quantity_int3D,
quantity_float0D,quantity_float1D,quantity_float2D,quantity_float3D,
D8Srec,D8Sdist,D8Sdons,D8Sndons,D8Mrecs,D8Mnrecs,D8Mdist,D8Mdons,D8Mndons,D8Mdondist,SStack,MStack):

	for i in range(n_elements):
		quantity_float1D[0] += quantity_float1D[1]