import numpy as np
import xsimlab as xs
import CHONK_XL as chxl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import numpy as np
import xarray as xr
import zarr
import numba as nb
from xmovie import Movie
import matplotlib
from scipy import stats
from IPython.display import display, Markdown, Latex

matplotlib.rc('font', serif='Helvetica Neue') 

@nb.njit()
def _gather_cross_section_info_EW(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props, row):
	elev = []
	cols = []
	volume = []

	# for row in range(ny):
	for col in range(nx):
		# getting the vectorised ID for the map view
		vid = row * nx + col
		# getting the elevation of the summit
		z0 = zs[vid,0]

		# gettring the max vectorised depth id
		j = ncells[vid,1] - 1
		# Getting the min one
		jmin = ncells[vid,0]
		#temp vars
		tprops = 0
		oj = False
		#Going through the depths
		while( j>= jmin):
			# if first, geeting the first depth ortherwise getting the regulrd dz

			tdz = dz if oj else f1[vid]
			oj = True

			for tlab in range(nlabels-1,-1,-1):
				if tlab == label:
					#print("adding", props[j] ,tdz ,dx ,dy, j)
					volume.append(props[j] * tdz * dx * dy)
					cols.append(col)
					elev.append(z0 - tdz/2)
				j -= 1

			nz = z0 - tdz
			z0 = nz
	return elev, cols, volume

@nb.njit()
def _gather_cross_section_info_EW_props(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props, row):
	elev = []
	cols = []
	volume = []

	# for row in range(ny):
	for col in range(nx):
		# getting the vectorised ID for the map view
		vid = row * nx + col
		# getting the elevation of the summit
		z0 = zs[vid,0]

		# gettring the max vectorised depth id
		j = ncells[vid,1] - 1
		# Getting the min one
		jmin = ncells[vid,0]
		#temp vars
		tprops = 0
		oj = False
		#Going through the depths
		while( j>= jmin):
			# if first, geeting the first depth ortherwise getting the regulrd dz

			tdz = dz if oj else f1[vid]
			oj = True

			for tlab in range(nlabels-1,-1,-1):
				if tlab == label:
					#print("adding", props[j] ,tdz ,dx ,dy, j)
					volume.append(props[j])
					cols.append(col)
					elev.append(z0 - tdz/2)
				j -= 1

			nz = z0 - tdz
			z0 = nz
	return elev, cols, volume


@nb.njit()
def _get_full_volume_sediment_label_map(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props):
	sumvol = np.zeros((ny,nx))

	for row in range(ny):
		for col in range(nx):
			# getting the vectorised ID for the map view
			vid = row * nx + col
			# getting the elevation of the summit
			z0 = zs[vid,0]

			# gettring the max vectorised depth id
			j = ncells[vid,1] - 1
			# Getting the min one
			jmin = ncells[vid,0]
			#temp vars
			tprops = 0
			oj = False
			#Going through the depths
			while( j>= jmin):
				# if first, geeting the first depth ortherwise getting the regulrd dz

				tdz = dz if oj else f1[vid]
				oj = True

				for tlab in range(nlabels-1,-1,-1):
					if tlab == label:
						#print("adding", props[j] ,tdz ,dx ,dy, j)
						sumvol[row,col] += props[j] * tdz * dx * dy
					j -= 1

				nz = z0 - tdz
				z0 = nz
	return sumvol.reshape(ny,nx)

@nb.njit()
def _get_first_m_volume_sediment_label_map(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props, depth):
	sumvol = np.zeros((ny,nx))

	for row in range(ny):
		for col in range(nx):
			# getting the vectorised ID for the map view
			vid = row * nx + col
			# getting the elevation of the summit
			z0 = zs[vid,0]

			# gettring the max vectorised depth id
			j = ncells[vid,1] - 1
			# Getting the min one
			jmin = ncells[vid,0]
			#temp vars
			tprops = 0
			oj = False
			totz = 0
			#Going through the depths
			while( j>= jmin):
				# if first, geeting the first depth ortherwise getting the regulrd dz

				tdz = dz if oj else f1[vid]
				oj = True

				for tlab in range(nlabels-1,-1,-1):
					if tlab == label:
						#print("adding", props[j] ,tdz ,dx ,dy, j)
						sumvol[row,col] += props[j] * tdz * dx * dy
					j -= 1

				nz = z0 - tdz
				totz += tdz
				if(totz > depth):
					break
				z0 = nz
	return sumvol.reshape(ny,nx)

def get_full_volume_sediment_label_map(ds,sel,label, ChonkBase = 'ChonkBase'):
	

	nlabels = ds.n_labels.values.shape[0]


	tds = ds
	for key in sel.keys():
		try:
			tds = tds.sel({key:sel[key]}, method = 'nearest')
		except:
			tds = tds.sel({key:sel[key]})
			

	path_strati = tds[ChonkBase + "__path_strati"].values.item(0)
	pref_strati = tds[ChonkBase + "__pref_strati"].values.item(0)
	name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
	while(len(name) < 8):
		name = '0' + name

	try:
		zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
		ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
		props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
		vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
		f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")
	except FileNotFoundError:
		print("Could not find the strati file...")
		link = '''[Sorry!](https://www.youtube.com/watch?v=Iouungm1aeQ&feature=emb_rel_end")'''
		display(Markdown(link))

	dz = tds[ChonkBase + "__depths_res_sed_proportions"].values.item(0)
	nx = tds[ChonkBase + "__nx"].values.item(0)
	ny = tds[ChonkBase + "__ny"].values.item(0)
	dy = tds[ChonkBase + "__dy"].values.item(0)
	dx = tds[ChonkBase + "__dx"].values.item(0)
		
	print("loading ", path_strati+ "/" + pref_strati + "_props_" + name + ".npy")

	# print(tds)
	return  _get_full_volume_sediment_label_map(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props)

def get_first_m_volume_sediment_label_map(ds,sel,label, depth, ChonkBase = 'ChonkBase'):
	

	nlabels = ds.n_labels.values.shape[0]


	tds = ds
	for key in sel.keys():
		try:
			tds = tds.sel({key:sel[key]}, method = 'nearest')
		except:
			tds = tds.sel({key:sel[key]})
			

	path_strati = tds[ChonkBase + "__path_strati"].values.item(0)
	pref_strati = tds[ChonkBase + "__pref_strati"].values.item(0)
	name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
	while(len(name) < 8):
		name = '0' + name

	try:
		zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
		ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
		props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
		vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
		f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")
	except FileNotFoundError:
		print("Could not find the strati file...")
		link = '''[Sorry!](https://www.youtube.com/watch?v=Iouungm1aeQ&feature=emb_rel_end")'''
		display(Markdown(link))

	dz = tds[ChonkBase + "__depths_res_sed_proportions"].values.item(0)
	nx = tds[ChonkBase + "__nx"].values.item(0)
	ny = tds[ChonkBase + "__ny"].values.item(0)
	dy = tds[ChonkBase + "__dy"].values.item(0)
	dx = tds[ChonkBase + "__dx"].values.item(0)
		
	print("loading ", path_strati+ "/" + pref_strati + "_props_" + name + ".npy")

	# print(tds)
	return  _get_first_m_volume_sediment_label_map(ny,nx, zs, ncells, f1, dz, dx, dy, label, nlabels, props, depth)
	

	




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
						aspect_rad = np.pi / 2

				HSarray[i, j] = ((np.cos(zenith_rad) * np.cos(slope_rad)) +
												(np.sin(zenith_rad) * np.sin(slope_rad) *
												np.cos(azimuth_rad - aspect_rad)))
				# Necessary?
				if (HSarray[i, j] < 0):
					HSarray[i, j] = 0





########### plotting functions helper

def get_well(
	ds, # The input ds 
	# fname = "well.png",
	timedim = "otime", # the time dimension
	time = 1e6,
	X = 5000,
	Y = 5000,
	ChonkBase = 'ChonkBase',
	label = 0,
	cmap = "magma",
	color_bedrock = 'gray',
	figsize = None,
	nlabels = 2,
	dpi = 300

):

	tds = ds.sel({timedim:time}, method = 'nearest')
	path_strati = tds[ChonkBase + "__path_strati"].item(0)
	pref_strati = tds[ChonkBase + "__pref_strati"].item(0)
	nx = tds[ChonkBase + "__nx"].values.item(0)
	col, row = np.argmin(np.abs(ds.x.values - X)), np.argmin(np.abs(ds.y.values - Y))
	print("row, col are ",row,col)
	vid = nx * row + col
	print("vid is",vid)

	# print("dz is",dz)


	name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
	while(len(name) < 8):
		name = '0' + name

	zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
	ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
	props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
	vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
	f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")
	
	cmap = matplotlib.cm.get_cmap(cmap)
	norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

	fig = plt.figure(figsize = figsize) if not (figsize is None) else plt.figure()
	gs = matplotlib.gridspec.GridSpec(100, 100)

	ax = fig.add_subplot(gs[:,7:80])
	# cb = ax.imshow([[0,1],[0,1]], cmap = cmap, zorder = -1, aspect = 'auto')
	# cb = ax.imshow([[0,0],[0,0]], cmap = cmap, zorder = -1, aspect = 'auto')

	z0 = zs[vid,0]
	j = ncells[vid,1] - 1
	jmin = ncells[vid,0]
	tprops = 0
	oj = False
	print(f1[vid])

	while(j>=jmin):
		dz = tds[ChonkBase + "__depths_res_sed_proportions"].item(0) if oj else f1[vid]
		oj = True

		for tlab in range(nlabels-1,-1,-1):
			if tlab == label:
				tprops = props[j]
			j -= 1

		nz = z0 - dz
		# print("tprops was ", tprops)
		color = cmap(norm(tprops))
		ax.fill_between([0,1],[z0,z0],[nz,nz], lw = 0, color = color)
		z0 = nz


	ax.fill_between([0,1], [z0,z0], [z0 - 5 * dz, z0 - 5* dz], lw = 0, color = color_bedrock)


	ax.set_xlim(0,1)
	ax.set_xticks([])
	ax.set_xlabel(r"X:%s, Y: %s"%(X,Y))
	ax.set_ylabel("Elevation (m)")
	# ax.invert_yaxis()

	ax2 = fig.add_subplot(gs[:,90:])

	cb = matplotlib.colorbar.ColorbarBase(ax2, orientation='vertical', 
                               cmap=plt.get_cmap(cmap))

	ax2.set_ylabel("Proportions of label %s"%(label))

	return fig, ax
	

def get_well_time_serie(
	ds, # The input ds 
	# fname = "well.png",
	timedim = "otime", # the time dimension
	X = 5000,
	Y = 5000,
	ChonkBase = 'ChonkBase',
	label = 0,
	cmap = "magma",
	color_bedrock = 'gray',
	figsize = None,
	nlabels = 2,
	dpi = 300

):
	
	fig = plt.figure(figsize = figsize) if not (figsize is None) else plt.figure()
	gs = matplotlib.gridspec.GridSpec(100, 100)
	# dt = (ds[timedim].values[2] - ds[timedim].values[1])/2
	ax = fig.add_subplot(gs[:,7:80])

	for i in range(ds[timedim].values.shape[0]):

		tt = ds[timedim].values[i]
		ttm1 = ds[timedim].values[i - 1] if i>0 else ds[timedim].values[0] - 100
		ttp1 = ds[timedim].values[i + 1] if i<ds[timedim].values.shape[0] - 1 else ds[timedim].values[-1] + 100
		# print(ttm1,tt,ttp1)

		tds = ds.sel({timedim:tt}, method = 'nearest')
		path_strati = tds[ChonkBase + "__path_strati"].values.item(0)
		pref_strati = tds[ChonkBase + "__pref_strati"].values.item(0)
		nx = tds[ChonkBase + "__nx"].values.item(0)
		col, row = np.argmin(np.abs(ds.x.values - X)), np.argmin(np.abs(ds.y.values - Y))
		# print("row, col are ",row,col)
		vid = nx * row + col
		# print("vid is",vid)

		dz = tds[ChonkBase + "__depths_res_sed_proportions"].item(0)
		# print("dz is",dz)

		try:
			name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
		except:
			continue

		while(len(name) < 8):
			name = '0' + name
		# print(name)
		zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
		ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
		props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
		vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
		f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")

		
		cmap = matplotlib.cm.get_cmap(cmap)
		norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

		

		# cb = ax.imshow([[0,1],[0,1]], cmap = cmap, zorder = -1, aspect = 'auto')
		# cb = ax.imshow([[0,0],[0,0]], cmap = cmap, zorder = -1, aspect = 'auto')

		z0 = zs[vid,0]
		j = ncells[vid,1] - 1
		jmin = ncells[vid,0]
		tprops = 0
		oj = False


		while(j>=jmin):
			dz = tds[ChonkBase + "__depths_res_sed_proportions"].item(0) if oj else f1[vid]
			oj = True
			for tlab in range(nlabels-1,-1,-1):
				if tlab == label:
					tprops = props[j]
				j -= 1

			nz = z0 - dz
			# print("tprops was ", tprops)
			color = cmap(norm(tprops))
			# print([(tt + ttm1) /2, (ttp1 + tt)/2])
			ax.fill_between([(tt + ttm1) /2, (ttp1 + tt)/2],[z0,z0],[nz,nz], lw = 0, color = color)
			z0 = nz


		ax.fill_between([(tt + ttm1) /2, (ttp1 + tt)/2], [z0,z0], [z0 - 5 * dz, z0 - 5* dz], lw = 0, color = color_bedrock)


	# ax.set_xlim(0,1)
	# ax.set_xticks()
	ax.set_xlabel("time (yrs)")
	ax.set_ylabel("Elevation (m)")
	# ax.invert_yaxis()

	ax2 = fig.add_subplot(gs[:,90:])

	cb = matplotlib.colorbar.ColorbarBase(ax2, orientation='vertical', 
                               cmap=plt.get_cmap(cmap))

	ax2.set_ylabel("Proportions of label %s"%(label))

	return fig, ax




def get_cross_section_EW(
	ds, # The input ds 
	# fname = "well.png",
	sel,
	ChonkBase = 'ChonkBase',
	label = 0,
	cmap = "magma",
	color_bedrock = 'gray',
	figsize = None,
	nlabels = 2,
	dpi = 300,
	minmax = None,

):

	tds = ds
	for key in sel.keys():
		try:
			tds = tds.sel({key:sel[key]}, method = 'nearest')
		except:
			tds = tds.sel({key:sel[key]})
	
	path_strati = tds[ChonkBase + "__path_strati"].values.item(0)
	pref_strati = tds[ChonkBase + "__pref_strati"].values.item(0)
	nx = tds[ChonkBase + "__nx"].values.item(0)
	ny = tds[ChonkBase + "__ny"].values.item(0)

	dx = tds[ChonkBase + "__dx"].values.item(0)
	dy = tds[ChonkBase + "__dy"].values.item(0)
	
	name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
	while(len(name) < 8):
		name = '0' + name

	zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
	ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
	props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
	vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
	f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")
	
	cmap = matplotlib.cm.get_cmap(cmap)
	norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
	
	fig = plt.figure(figsize = figsize) if not (figsize is None) else plt.figure()
	gs = matplotlib.gridspec.GridSpec(100, 100)

	ax = fig.add_subplot(gs[:,:])
	
	gdz = tds[ChonkBase + "__depths_res_sed_proportions"].item(0)
	row = np.argmin(np.abs(ds.y.values - sel["y"]))


	elev, cols, volume = _gather_cross_section_info_EW(ny,nx, zs, ncells, f1, gdz, dx, dy, label, nlabels, props, row)
	elev = np.asarray(elev)
	# cols = np.asarray(cols).astype(np.int32)
	volume = np.asarray(volume)
	Ys = tds.x.values[cols]
	zbins = np.arange( elev.min() - gdz, elev.max() + gdz, gdz)
	# print(np.unique(np.isfinite(volume)))
	# print(np.unique(np.isfinite(Ys)))
	# print(np.unique(np.isfinite(zbins)))

	ret = stats.binned_statistic_2d(Ys, elev, volume, 'median', bins=[tds.x.values, zbins])
	# ret.statistic[np.isfinite(ret.statistic ) == False] = 0

	if(minmax is None):
		cb = ax.imshow(np.rot90(ret.statistic), extent = [ret.x_edge.min(),ret.x_edge.max(), ret.y_edge.min(),ret.y_edge.max()], cmap = cmap, aspect = 'auto')
	else:
		cb = ax.imshow(np.rot90(ret.statistic), extent = [ret.x_edge.min(),ret.x_edge.max(), ret.y_edge.min(),ret.y_edge.max()], cmap = cmap, aspect = 'auto', vmin = minmax[0], vmax = minmax[1])


	ax.plot(tds.x.values,tds.Topography__topography.values, color = 'k', lw =2)
	ax.set_xlim(tds.x.values.min(),tds.x.values.max())
	# ax.set_xticks([])
	ax.set_xlabel(r"Northing (m)")
	ax.set_ylabel("Elevation (m)")
	# ax.invert_yaxis()

	plt.colorbar(cb, label = "Volume of label %s in $m^{3}$"%(label))

	# ax2 = fig.add_subplot(gs[:,90:])

	# cb = matplotlib.colorbar.ColorbarBase(ax2, orientation='vertical', 
 #                               cmap=plt.get_cmap(cmap))

	# ax2.set_ylabel("Proportions of label %s"%(label))

	return fig, ax


def get_cross_section_EW_props(
	ds, # The input ds 
	# fname = "well.png",
	sel,
	ChonkBase = 'ChonkBase',
	label = 0,
	cmap = "magma",
	color_bedrock = 'gray',
	figsize = None,
	nlabels = 2,
	dpi = 300,
	minmax = None,

):

	tds = ds
	for key in sel.keys():
		try:
			tds = tds.sel({key:sel[key]}, method = 'nearest')
		except:
			tds = tds.sel({key:sel[key]})
	
	path_strati = tds[ChonkBase + "__path_strati"].values.item(0)
	pref_strati = tds[ChonkBase + "__pref_strati"].values.item(0)
	nx = tds[ChonkBase + "__nx"].values.item(0)
	ny = tds[ChonkBase + "__ny"].values.item(0)

	dx = tds[ChonkBase + "__dx"].values.item(0)
	dy = tds[ChonkBase + "__dy"].values.item(0)
	
	name = str(int(tds[ChonkBase + "__strati"].values.item(0)))
	while(len(name) < 8):
		name = '0' + name

	zs = np.load(path_strati+ "/" + pref_strati + "_zs_" + name + ".npy")
	ncells = np.load(path_strati+ "/" + pref_strati + "_ncells_" + name + ".npy")
	props = np.load(path_strati+ "/" + pref_strati + "_props_" + name + ".npy")
	vol = np.load(path_strati+ "/" + pref_strati + "_vol_" + name + ".npy")
	f1 = np.load(path_strati+ "/" + pref_strati + "_firstbox_" + name + ".npy")
	
	cmap = matplotlib.cm.get_cmap(cmap)
	norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
	
	fig = plt.figure(figsize = figsize) if not (figsize is None) else plt.figure()
	gs = matplotlib.gridspec.GridSpec(100, 100)

	ax = fig.add_subplot(gs[:,:])
	
	gdz = tds[ChonkBase + "__depths_res_sed_proportions"].item(0)
	row = np.argmin(np.abs(ds.y.values - sel["y"]))


	elev, cols, props = _gather_cross_section_info_EW_props(ny,nx, zs, ncells, f1, gdz, dx, dy, label, nlabels, props, row)
	elev = np.asarray(elev)
	# cols = np.asarray(cols).astype(np.int32)
	props = np.asarray(props)
	Ys = tds.x.values[cols]
	zbins = np.arange( elev.min() - gdz, elev.max() + gdz, gdz)
	# print(np.unique(np.isfinite(props)))
	# print(np.unique(np.isfinite(Ys)))
	# print(np.unique(np.isfinite(zbins)))

	ret = stats.binned_statistic_2d(Ys, elev, props, 'median', bins=[tds.x.values, zbins])
	# ret.statistic[np.isfinite(ret.statistic ) == False] = 0

	if(minmax is None):
		cb = ax.imshow(np.rot90(ret.statistic), extent = [ret.x_edge.min(),ret.x_edge.max(), ret.y_edge.min(),ret.y_edge.max()], cmap = cmap, aspect = 'auto', vmin = 0, vmax = 1)
	else:
		cb = ax.imshow(np.rot90(ret.statistic), extent = [ret.x_edge.min(),ret.x_edge.max(), ret.y_edge.min(),ret.y_edge.max()], cmap = cmap, aspect = 'auto', vmin = minmax[0], vmax = minmax[1])


	ax.plot(tds.x.values,tds.Topography__topography.values, color = 'k', lw =2)
	ax.set_xlim(tds.x.values.min(),tds.x.values.max())
	# ax.set_xticks([])
	ax.set_xlabel(r"Northing (m)")
	ax.set_ylabel("Elevation (m)")
	# ax.invert_yaxis()

	plt.colorbar(cb, label = "Proportion of label %s"%(label))

	# ax2 = fig.add_subplot(gs[:,90:])

	# cb = matplotlib.colorbar.ColorbarBase(ax2, orientation='vertical', 
 #                               cmap=plt.get_cmap(cmap))

	# ax2.set_ylabel("Proportions of label %s"%(label))

	return fig, ax


	




def anim_lake_cross_section(
	ds, # The input ds 
	fname = "outputgif",
	timedim = "otime", # the time dimension
	batch_dim = None, # if there is a batch dim to pick
	batch_val = None, # if there is a batch dim to pick
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
		
		if(batch_dim is not None):
			tds = ds.isel({timedim:tt, batch_dim:batch_val})
		else:
			tds = ds.isel({timedim:tt})
		
		shape = tds["Topography__topography"].values.shape
		HS = np.zeros(shape)
		hillshading(tds["Topography__topography"].values,tds["ChonkBase__dx"].item(0),tds["ChonkBase__dx"].item(0),shape[1],shape[0],HS,np.deg2rad(60),np.deg2rad(125),1)
		
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




























#Old stuff
class Landscape(object):
	"""
		Landscape class: helps generating the parametric landscape for CHONK paper
	"""
	
	def __init__(self):
		
		pass
	
	def set_dimensions_from_res(self, nx = 100, ny = 100, dx = 200, dy = 200):
		"""
			Generate the landscapes dimensions function of number of cols, row and dx, dy
		"""

		self.nx = nx
		self.ny = ny
		self.dx = dx
		self.dy = dy
		self.lx = (nx) * dx
		self.ly = (ny) * dy

		self.indices = np.zeros(ny * nx, dtype = np.int)

	def set_dimensions_from_length(self, nx = 100, ny = 100, lx = 20000, ly = 20000):
		"""
			Generate the landscapes dimensions function of number of cols, row and total length in X and Y
		"""
		self.nx = nx
		self.ny = ny
		self.lx = lx
		self.ly = ly
		self.dx = (lx) / (nx - 1)
		self.dy = (ly) / (ny - 1)
		self.indices = np.zeros(ny * nx, dtype = np.int)

	def set_boundaries_elevation(self, N = 1000, S = 0):
		self.N_bound = N
		self.S_bound = S
		
	def set_rel_distances(self, mountain_front = 0.7, normal_fault = 0.5):
		self.mountain_front = mountain_front
		self.normal_fault = normal_fault
		self.mountain_front_idx = round(mountain_front * self.ly / self.dy)
		self.normal_fault_idx = round(normal_fault * self.ly / self.dy)

		
	def generate_uplift_4_StSt(self, U = 2e-4):
			
		self.uplift4StSt = np.zeros((self.ny,self.nx), dtype = np.float64)
		self.uplift4StSt[:self.mountain_front_idx ,:] = U
		self.uplift4StSt[[-1,0],:] = 0
		self.Uy = np.copy(self.uplift4StSt[:,2])
		
	def generate_uplift_Normal_fault(self, Upos = 1e-4, Uneg = 1e-4, alpha_pos = 5e3, alpha_neg = 5e3):
		
		X = np.arange(0, self.ny * self.dy + self.dy, self.dy)
		self.Uy[:self.normal_fault_idx] -= (Uneg * (np.exp(-X[:self.normal_fault_idx][::-1]/alpha_neg) * np.cos(X[:self.normal_fault_idx][::-1]/alpha_neg)) )
		tX = np.arange(0,(self.mountain_front_idx - self.normal_fault_idx) * self.dy,self.dy )
		self.Uy[self.normal_fault_idx:self.mountain_front_idx] += Upos * (np.exp(-tX/alpha_pos) * np.cos(tX/alpha_pos))
		minUy = self.Uy.min()
		if(minUy < 0):
			self.Uy += abs(minUy)
		
		self.Uy[self.mountain_front_idx:] = 0
		
		self.uplift_NF = np.zeros((self.ny,self.nx), dtype = np.float64)
		for i in range(self.nx):
			self.uplift_NF[:,i] = self.Uy
		
	def add_pluton(self, dimless_X = 0.6, dimless_Y = 0.3, half_width = 5000,  half_heigth = 3000):

		x0 = round(dimless_X * self.lx )
		a = half_width  # x center, half width									   
		y0 = round(dimless_Y * self.ly )
		b = half_heigth  # y center, half height   

		x = np.linspace(0, self.lx, self.nx)  # x values of interest
		y = np.linspace(0, self.ly, self.ny)[:,None]  # y values of interest, as a "column" array
		# print("x0:", x0, "y0:", y0)

		self.indices =  ( ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1) #).astype(np.int32) # True for points inside the ellipse














		


	def generate_uplift_field_NFMT(self, distance_normfault = 0.4, distance_1st_fault = 0.65, distance_second_fault = 0.8,
																	max_U = 1e-3, med_U = 0.4e-3, min_U = 0, bound_U = 0 , a = 1e-5):

		self.uplift_phase_1 = np.zeros((self.ny,self.nx), dtype = np.float64)
		self.uplift_phase_2 = np.zeros((self.ny,self.nx), dtype = np.float64)

		idN = round(distance_normfault * self.ly / self.dy)
		id1 = round(distance_1st_fault * self.ly / self.dy)
		id2 = round(distance_second_fault * self.ly / self.dy)

		# getting the log inversed right
		Y = np.arange(0,self.ly + self.dy/2, self.dy)
		self.Uy = np.zeros_like(Y) + min_U
		self.Uy[:idN] = - np.log( 1/Y[:idN][::-1] + a)
		self.Uy[:idN] /= self.Uy[:idN].max()
		self.Uy[:idN] *= max_U
		self.Uy[idN - 1] = self.Uy[idN - 2]

		self.Uy[idN:id1] = max_U

		self.uplift_phase_1[:] = self.Uy
		self.uplift_phase_1 = np.rot90(self.uplift_phase_1,3)

		self.Uy[id1:id2] = max_U
		self.Uy[idN:id1] = 0
		
		self.uplift_phase_2[:] = self.Uy
		self.uplift_phase_2 = np.rot90(self.uplift_phase_2,3)


































		# end of file

