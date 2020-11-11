import numba as nb
import numpy as np
import math as m

@nb.jit(nopython = True,cache = True)
def hillshading(arr,dx,dy,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor):
  """
  Slow and basic imnplementation of the hillshading algorithm present in LSDTT c++ algorithmath.
  This is basically just to train myself before optimization/parrallelization with numba
  BG, from DAV-SWDG-SMM codes
  """

  for i in range(nrows):
    for j in range(ncols):

      if arr[i, j] != -9999:

        dzdx = (((arr[i, j+1] + 2*arr[i+1, j] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i-1, j] + arr[i-1, j+1]))
                / (8 * dx))
        dzdy = (((arr[i-1, j+1] + 2*arr[i, j+1] + arr[i+1, j+1]) -
                (arr[i-1, j-1] + 2*arr[i, j-1] + arr[i+1, j-1]))
                / (8 * dy))

        slope_rad = np.arctan(z_factor * np.sqrt((dzdx * dzdx) + (dzdy * dzdy)))

        if (dzdx != 0):
          aspect_rad = np.arctan2(dzdy, (dzdx*-1))
          if (aspect_rad < 0):
            aspect_rad = 2 * np.pi + aspect_rad

        else:
          if (dzdy > 0):
            aspect_rad = np.pi / 2
          elif (dzdy < 0):
            aspect_rad = 2 * np.pi - np.pi / 2
          else:
            aspect_rad = aspect_rad

        HSarray[i, j] = 255.0 * ((np.cos(zenith_rad) * np.cos(slope_rad)) +
                        (np.sin(zenith_rad) * np.sin(slope_rad) *
                        np.cos(azimuth_rad - aspect_rad)))
        # Necessary?
        if (HSarray[i, j] < 0):
          HSarray[i, j] = 0