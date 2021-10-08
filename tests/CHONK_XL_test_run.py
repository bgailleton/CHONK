#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import CHONK_XL as chxl
import matplotlib.pyplot as plt
import xsimlab as xs
import CHONK_cpp as ch
import zarr


# In[2]:


ny = 100
nx = 100

rd3 = round(ny/3)

Up = np.zeros((ny,nx)) + 1e-3
Up[0:rd3,:] = 1e-4
Up[0:rd3,:] = 1e-4
Up[2*rd3:,:] = 1e-4
Up[[0,-1],:] = 0
Up = Up.ravel()

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
        "topo": chxl.RandomInitialSurface,
        "labelling": CustomLabelling,
        "flow": chxl.D8Flow, #MF2D8Flow,
        "methods": chxl.OrderedMethods,
        "core": chxl.CoreModel,
        "uplift": chxl.BlockUplift
        
    }
)
mod_sed.input_vars


# In[4]:

timing = np.arange(0, 1000000, 1000)
otiming = timing[::100]

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
            "dx": 200,
            "dy": 200,
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
        }
        
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
        
    }

        
        
)



# In[ ]:


with mod_sed,xs.monitoring.ProgressBar():
    out_ds = mod1.xsimlab.run()


# In[ ]:


import hvplot.xarray
import holoviews as hv
import datashader
import xarray as xr


hv.extension('bokeh')
wi, he = 600,500

topo = out_ds.core__topo.hvplot.image(
    x='x', y='y',
    width = wi, height = he,
    cmap=plt.cm.gist_earth, groupby='otime', dynamic = True
)

Qw = np.log10(out_ds.core__Q_water).hvplot.image(
    x='x', y='y',
    width = wi, height = he,
    cmap=plt.cm.Blues, groupby='otime', dynamic = True
)


HS = out_ds.core__HS.hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,250),
    cmap=plt.cm.gray, groupby='otime', alpha = 0.45, dynamic = True
)

LD = out_ds.core__sed_thickness.hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,5),
    cmap=plt.cm.viridis, groupby='otime', alpha = 0.6, dynamic = True
)

prop = out_ds.core__labprop_superficial_layer.sel(n_labels = 1).hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,1),
    cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
)

prop_QS = out_ds.core__labprop_Qs.sel(n_labels = 0).hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,1),
    cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
)


prop_delta = (out_ds.core__labprop_Qs.sel(n_labels = 0) - out_ds.core__labprop_superficial_layer.sel(n_labels = 0)).hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,0.1),
    cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
)

E_s = out_ds.core__E_s.hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,1e-6),
    cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
)

E_r = out_ds.core__E_r.hvplot.image(
    x='x', y='y',
    width = wi, height = he, clim = (0,1e-3),
    cmap=plt.cm.magma, groupby='otime', alpha = 1, dynamic = True
)
# core__labprop_Qs
# LD = out_ds.core__lake_depth.hvplot.image(
#     x='x', y='y',
#     width = wi, height = he, clim = (-1,0),
#     cmap=plt.cm.Blues, groupby='otime', alpha = 0.6, dynamic = True
# )


(LD * HS )


# In[ ]:


out_ds.core__sed_thickness


# In[ ]:





# In[ ]:





# In[ ]:




