#!/usr/bin/env python
# coding: utf-8

# In[1]:

import socket
socket.gethostbyname("")
import numpy as np
import CHONK_XL as chxl
import matplotlib.pyplot as plt
import xsimlab as xs
import CHONK_cpp as ch
import zarr
import helplotlib as hpl



# In[2]:


ny = 50
nx = 50
dt = 500
dx = 100
dy = 100

rd3 = round(ny/3)

Up = np.zeros((ny,nx)) + 1e-3
# Up[0:rd3,:] = 1e-4
# Up[0:rd3,:] = 1e-4
# Up[2*rd3:,:] = 1e-4
Up[[0,-1],:] = 0
Up = Up.ravel()

Z = np.load("./example/SStopo50_50.npy")

# Z[25:30,25:30] = -1 * Z[25:30,25:30] + Z[25:30,25:30].min() - 10
# Z[25:30,25:30] = 10
Z[3, 7:20] = 600

# Z = np.load("Qw_massbal_issue_1.npy")


Z = np.random.rand(ny,nx)
Z = np.zeros((ny,nx))
Z[20:30,20:30] = 5
Z[20:30,20] = 6
# Z[20:30,29] = 6
# Z[20,20:30] = 6


my_label_array = np.zeros((ny,nx), np.int)
my_label_array[rd3:2*rd3,:] = 1

@xs.process
class CustomLabelling(chxl.Labelling):

    def initialize(self):

        self.label_list = []
        self.label_list.append(ch.label(0))
        self.label_list[0].set_double_attribute("SPIL_m", 0.45);
        self.label_list[0].set_double_attribute("SPIL_n", 1);
        self.label_list[0].set_double_attribute("SPIL_K", 1e-5);
        self.label_list[0].set_double_attribute("CHARLIE_I_Kr", 1e-5);
        self.label_list[0].set_double_attribute("CHARLIE_I_Ks", 2e-5);
        self.label_list[0].set_double_attribute("CHARLIE_I_V", 2);
        self.label_list[0].set_double_attribute("CHARLIE_I_dimless_roughness", 1);
        self.label_list[0].set_double_attribute("CHARLIE_I_dstar", 1);
        self.label_list[0].set_double_attribute("CHARLIE_I_threshold_incision", 0);
        self.label_list[0].set_double_attribute("CHARLIE_I_threshold_entrainment", 0);
        self.label_list.append(ch.label(1))
        self.label_list[1].set_double_attribute("SPIL_m", 0.45);
        self.label_list[1].set_double_attribute("SPIL_n", 1);
        self.label_list[1].set_double_attribute("SPIL_K", 2e-5);
        self.label_list[1].set_double_attribute("CHARLIE_I_Kr", 0.7e-5);
        self.label_list[1].set_double_attribute("CHARLIE_I_Ks", 2e-5);
        self.label_list[1].set_double_attribute("CHARLIE_I_V", 0.1);
        self.label_list[1].set_double_attribute("CHARLIE_I_dimless_roughness", 0.5);
        self.label_list[1].set_double_attribute("CHARLIE_I_dstar", 1);
        self.label_list[1].set_double_attribute("CHARLIE_I_threshold_incision", 0);
        self.label_list[1].set_double_attribute("CHARLIE_I_threshold_entrainment", 0);


# In[3]:


mod_sed = xs.Model(
    {
        "grid" : chxl.GridSpec,
        "bound" : chxl.BoundaryConditions,
        "topo": chxl.CustomInitialSurface,
        "labelling": CustomLabelling,
        "flow": chxl.D8Flow,  #MF2D8Flow,
        "methods": chxl.OrderedMethods,
        "core": chxl.CoreModel,
        "uplift": chxl.BlockUplift
        
    }
)
# mod_sed.input_vars


# In[4]:


timing = np.arange(0,100000, dt)
# timing = np.arange(0,600, dt)
otiming = timing[::10]

mod1 = xs.create_setup(
    model = mod_sed,
    clocks={
        'time': timing,
        'otime': otiming
    },
    master_clock = 'time',
    
    input_vars = {
        'bound' : {
            'boundary_conditions' : "periodic_EW"
        },
        'grid': {
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny
        },
        'labelling' : {
            'label_array': my_label_array
        },
        'methods': {
            'methods_pre_move' : np.array(["drainage_area"]),
            'methods_post_move' : np.array(["CHARLIE_I"])
        },
        'uplift' : {
            'uplift' : Up
        },
        'core' : {
            'depths_res_sed_proportions' : 1.,
            'lake_solver' : True
        },
        'topo' : {"this_surface_elevation": Z.ravel()}
        
    },
    output_vars = {
        'core__topo' : 'otime',
        'core__HS' : 'otime',
        'core__Q_water' : 'otime',
        'core__lake_depth' : 'otime',
        'core__labprop_Qs': 'otime',
        'core__labprop_superficial_layer': 'otime',
        'core__sed_thickness': 'otime',
        'core__E_r': 'otime',
        'core__E_s': 'otime',
        'core__sed_div': 'otime',
        'core__lake_id_raw': 'otime',
        'core__mstack_checker': 'otime',
        'core__Qw_in' : 'otime',
        'core__Qw_out' : 'otime',
        'core__Ql_in' : 'otime',
        'core__Ql_out' : 'otime',
        'core__water_balance_checker' : 'otime',
        'core__flat_mask' : 'otime',
        
        
    }

        
        
)


# In[5]:


# zg = zarr.group("test200_200.zarr", overwrite=True)
with mod_sed,xs.monitoring.ProgressBar():
    out_ds = mod1.xsimlab.run()


# In[ ]:


# fig, ax = hpl.mkfig_grey_bold()

# # ax.plot(out_ds.otime,out_ds.core__water_balance_checker.values)
# # ax.plot(out_ds.otime,out_ds.core__Qw_in.values)
# # ax.plot(out_ds.otime,out_ds.core__Qw_out.values)

# # ax.plot(out_ds.otime, out_ds.core__Ql_in.values)
# # ax.fill_between(out_ds.otime,0, out_ds.core__Ql_in.values, color = 'b')
# # ax.set_ylabel("$Q_{l}^{in}$ (in $m^3.yrs^{-1}$)")


# # ax.plot(out_ds.core__Ql_out.values)
# # ax.set_yscale('log')

# ax.set_xlabel("Time (yrs)")


# # In[ ]:


# fig,ax = plt.subplots()
# ax.imshow(Z)#, vmin = -0.01, vmax = 0.01)


# # In[ ]:


# import hvplot.xarray
# import holoviews as hv
# import datashader
# import xarray as xr


# hv.extension('bokeh')
# wi, he = 600,500

# topo = out_ds.core__topo.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he,#clim = (450,550),
#     cmap=plt.cm.gist_earth, groupby='otime', dynamic = True
# )

# Qw = np.log10(out_ds.core__Q_water).hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (3,8),
#     cmap=plt.cm.Blues, groupby='otime', dynamic = True
# )


# HS = out_ds.core__HS.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (0,250),
#     cmap=plt.cm.gray, groupby='otime', alpha = 0.45, dynamic = False
# )

# SH = out_ds.core__sed_thickness.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he,# clim = (0,5),
#     cmap=plt.cm.viridis, groupby='otime', alpha = 0.6, dynamic = True
# )

# prop = out_ds.core__labprop_superficial_layer.sel(n_labels = 1).hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (0,1),
#     cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = False
# )

# prop_QS = out_ds.core__labprop_Qs.sel(n_labels = 0).hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (0,1),
#     cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
# )


# prop_delta = (out_ds.core__labprop_Qs.sel(n_labels = 0) - out_ds.core__labprop_superficial_layer.sel(n_labels = 0)).hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (0,0.1),
#     cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
# )

# E_s = out_ds.core__E_s.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (0,1e-6),
#     cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
# )

# E_r = out_ds.core__E_r.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he,# clim = (0,1e-3),
#     cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
# )
# # core__labprop_Qs
# LD = out_ds.core__lake_depth.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (-1,0),
#     cmap=plt.cm.Blues, groupby='otime', alpha = 0.6, dynamic = True
# )

# LiDr = out_ds.core__lake_id_raw.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, 
#     cmap=plt.cm.jet, groupby='otime', alpha = 0.6, dynamic = True
# )

# mstack = out_ds.core__mstack_checker.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, 
#     cmap=plt.cm.jet, groupby='otime', alpha = 0.3, dynamic = True
# )

# FM = out_ds.core__flat_mask.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, 
#     cmap=plt.cm.jet, groupby='otime', alpha = 0.3, dynamic = True
# )

# # LiDr
# # Qw *LD * HS
# # SH * HS
# # FM
# # Qw * HS


# In[ ]:


# np.save("Qw_massbal_issue_1.npy",out_ds.core__topo.sel({"otime": 11000}).values )


# In[ ]:


# from PIL import Image, ImageDraw
# fy = (prop * HS)
# cpt = 0
# for i in fy:
#     name = str(cpt)
#     while(len(name) < 4):
#         name = "0" + name
#     hv.save(i,"temp_output" +  name + ".png", fmt = "png")


# In[ ]:


# from PIL import Image, ImageDraw


# In[ ]:


# del fyd


# In[ ]:


previous = 0
for val,lake in zip(out_ds.core__Q_water.values,out_ds.core__lake_depth.values):
    this_lake_stuff = np.nansum(np.abs(lake)) * dx * dy/dt
    print(np.sum(val[0,:]) + np.sum(val[-1,:]) + this_lake_stuff - previous)
    previous = this_lake_stuff
    
print(dx*dy * ny * nx)






