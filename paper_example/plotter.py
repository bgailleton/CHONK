import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import numpy as np
import xarray as xr
import zarr
import numba as nb
from xmovie import Movie


matplotlib.rc('font', serif='Helvetica Neue') 


@nb.jit(nopython = True,cache = True)
def hillshading(arr,dx,dy,ncols,nrows,HSarray,zenith_rad,azimuth_rad, z_factor):
	
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

				HSarray[i, j] = ((np.cos(zenith_rad) * np.cos(slope_rad)) +
												(np.sin(zenith_rad) * np.sin(slope_rad) *
												np.cos(azimuth_rad - aspect_rad)))
				# Necessary?
				if (HSarray[i, j] < 0):
					HSarray[i, j] = 0
	
# 	HSarray = (HSarray - HSarray.min()) / HSarray.max()



def anim_lake_cross_section(
	ds, # The input ds 
	fname = "outputgif",
	timedim = "otime", # the time dimension
	batch_dim = None, # if there is a batch dim to pick
	cross_section_dir = 'y', # is the cross section in x or y direction
	xy_cross_section = 0, # coordinate on the other axis
	color_bedrock = 'gray', # color of the bedrock
	color_sediments = 'orange', # color of the bedrock
	color_water = 'blue', # color of the water
	z_min = None, # minimum z on the cross_section, if left to none -> min of all
	z_max = None, # max z on the cross_section, if left to none -> min of all
	czmin = 0,
	czmax = 1000,
	# Map parameters
	cmap_elev = 'gist_earth', # cmap of the cross-section
	alpha_hillshade = 0.6, # transparency of the hillshade
	figsize = (8,3),
	custom_tickszz = None,
	invert_ticks = True
):
	
	perczsc = 0.05 *ds["Topography__topography"].sel({cross_section_dir: xy_cross_section}, method = "nearest").max()
	if z_min is None:
		z_min = ds["Topography__topography"].sel({cross_section_dir: xy_cross_section}, method = "nearest").min() - perczsc
	if z_max is None:
		z_max = ds["Topography__topography"].sel({cross_section_dir: xy_cross_section}, method = "nearest").max() + perczsc

		
	def _anim_lake_cross_section(ds, fig, tt, framedim):

		tds = ds.isel({timedim:tt})
		
		shape = tds["Topography__topography"].values.shape
		HS = np.zeros(shape)
		hillshading(tds["Topography__topography"].values,200,200,shape[1],shape[0],HS,np.deg2rad(60),np.deg2rad(125),1)
		
# 		tds["Topography__topography"] -= ds["Topography__sed_height"]
# 		fig.set_size_inches(*figsize)
		(ax1, ax2) = fig.subplots(ncols=2)
		
		topomsed = tds["Topography__topography"] - tds["Topography__sed_height"]
		
		# Filling between stuff
		ax1.fill_between(tds[cross_section_dir],z_min,(topomsed.sel({cross_section_dir:xy_cross_section}, method = "nearest")) , color = color_bedrock)
		
		ax1.fill_between(tds[cross_section_dir].values,(topomsed.sel({cross_section_dir:xy_cross_section}, method = "nearest")), (topomsed + tds["Topography__sed_height"] ).sel({cross_section_dir:xy_cross_section}, method = "nearest"), color = color_sediments, lw = 0 )
		ax1.fill_between(tds[cross_section_dir],(topomsed + tds["Topography__sed_height"] ).sel({cross_section_dir:xy_cross_section}, method = "nearest"),(topomsed + tds["Topography__sed_height"] + tds["Lake__lake_depth"] ).sel({cross_section_dir:xy_cross_section}, method = "nearest") , color = color_water, lw = 0 )
		
		# Plot liny thingy
		ax1.plot(tds[cross_section_dir],(topomsed.sel({cross_section_dir:xy_cross_section}, method = "nearest")) , color = 'k', lw = 1)
		ax1.plot(tds[cross_section_dir],(topomsed + tds["Topography__sed_height"] ).sel({cross_section_dir:xy_cross_section}, method = "nearest") , color = 'k', lw = 1)
		ax1.plot(tds[cross_section_dir],(topomsed + tds["Topography__sed_height"] + tds["Lake__lake_depth"] ).sel({cross_section_dir:xy_cross_section}, method = "nearest") , color = 'k', lw = 1)
		
		
		ax1.set_ylim(z_min,z_max)
		ax1.set_xlabel("Northing (m)")
		ax1.set_ylabel("Elevation (m)")
# 		print(custom_tickszz)
		
		if(custom_tickszz is not None):
			custom_ticks = np.array(custom_tickszz)
			ax1.set_xticks(custom_ticks)
			if(invert_ticks):
				ax1.set_xticklabels(custom_ticks[::-1])
		
		ax1.annotate("Time: %s"%(ds[timedim].values[tt]), (0.6,0.92) ,xycoords = 'axes fraction')
# 		tticks = ax1.get_xticks()
# 		ticks = []
# 		ticklab = []
# 		for t in tticks:
# 			if(t is None) == False:
				
# 				ticks.append(t)
# 				ticklab.append(str(round(t)))
# 		print(ticks)
# 		print(ticklab)
# 		ax1.set_xticks(ticks)
# 		ax1.set_xticklabels(ticklab.reverse())
		
		
# 		ax1.scatter(points_x, points_y, s=10, color="black")
		
		cb = ax2.imshow((tds["Topography__topography"]), cmap = cmap_elev, vmin = czmin, vmax = czmax,
						extent = [tds["x"].min(), tds.x.max(), tds.y.min(), tds.y.max()])
		ax2.imshow((HS), cmap = 'gray', extent = [tds["x"].min(), tds.x.max(), tds.y.min(), tds.y.max()], alpha = alpha_hillshade,vmin = 0, vmax = 1)
		lakes = np.copy(tds["Lake__lake_depth"].values)
		lakes[lakes<=0] = np.nan
		lakes[np.isfinite(lakes)] = 0.85
		ax2.imshow((lakes), cmap = 'Blues',vmin = 0, vmax = 1, extent = [tds["x"].min(), tds.x.max(), tds.y.min(), tds.y.max()])
		ax2.set_xlabel("Easting (m)")
		ax2.set_ylabel("Northing (m)")
		
		
# 		norm_coord = (xy_cross_section - tds['x'].min()) / (tds['x'].max() - tds['x'].min() ) if cross_section_dir == 'y' else (xy_cross_section - tds['y'].min()) / (tds['y'].max() - tds['x'].min() )
		
		ax2.axvline(xy_cross_section, lw = 2, color = 'k', ls = '--') if cross_section_dir == 'x' else ax2.axhline(xy_cross_section, lw = 2, color = 'k', ls = '--')
		
		divider = make_axes_locatable(ax2)
		cax = divider.append_axes('top', size='5%', pad=0.4)
		fig.colorbar(cb, cax=cax, orientation='horizontal' )
		cax.set_xlabel("Elevation (m)")
		cax.xaxis.set_label_position('top') 
		
	#End of figurer
	mov_custom = Movie(ds, _anim_lake_cross_section, input_check = False, framedim = timedim)
	mov_custom.width = figsize[0]
	mov_custom.height = figsize[1]
	mov_custom.save(fname + '.gif', progress=True, overwrite_existing=True, remove_movie=False, gif_resolution_factor=1)

	

						
# 		tds["Topography__sed_height"]
# mov_custom = Movie(ds, custom_plotfunc)
# mov_custom.save('movie_custom.gif')




