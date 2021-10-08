#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import CHONK_XL as chxl
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex
import xsimlab as xs
import CHONK_cpp as ch
import zarr
import helplotlib as hpl
import helper
# get_ipython().run_line_magic('matplotlib', 'widget')
# get_ipython().run_line_magic('load_ext', 'xsimlab.ipython')


# In[2]:




@xs.process
class CustomParameters:
    label_array = xs.variable(intent = 'out', dims = (('y','x'), ('node')))
    label_list = xs.any_object()
    CHONK = xs.foreign(chxl.ChonkBase, "CHONK")
    nx = xs.foreign(chxl.ChonkBase, "nx")
    ny = xs.foreign(chxl.ChonkBase, "ny")
    dx = xs.foreign(chxl.ChonkBase, "dx")
    dy = xs.foreign(chxl.ChonkBase, "dy")
    
    active_nodes = xs.foreign(chxl.ChonkBase, "active_nodes")
    landscape = xs.any_object()

    def initialize(self):
        # Instanciating the landscape
        self.landscape = helper.Landscape()

        # params for the landscapes dimensions

        # landscape.set_dimensions_from_res( nx = 100, ny = 100, dx = 200, dy = 200)
        self.landscape.set_dimensions_from_length(nx = self.nx, ny = self.ny, lx = self.nx * self.dx, ly = self.ny * self.dy)
#         self.landscape.set_boundaries_elevation(N = 1000, S = 0)
        self.landscape.set_boundaries_elevation(N = 0, S = 0)
        self.landscape.set_rel_distances(mountain_front = 0.65, normal_fault = 0.30)
        self.landscape.generate_uplift_4_StSt(U = 2e-4)
        self.landscape.generate_uplift_Normal_fault(Upos = 1e-4, Uneg = 2e-4, alpha_pos = 2e3, alpha_neg = 0.8e4)
#         self.landscape.add_pluton( dimless_X = 0.6, dimless_Y = 0.3, half_width = 5000,  half_heigth = 3000)
        self.label_list = []
    
        self.label_array = self.landscape.indices

        self.label_list.append(ch.label(0))
        self.label_list[-1].m = 0.45;
        self.label_list[-1].n = 1;
        self.label_list[-1].base_K = 3e-6;
        self.label_list[-1].Ks_modifyer = 1.2;
        self.label_list[-1].Kr_modifyer = 0.8;
        self.label_list[-1].dimless_roughness = 0.5;
        self.label_list[-1].V = 0.5;
        self.label_list[-1].dstar = 1;
        self.label_list[-1].threshold_incision = 0;
        self.label_list[-1].threshold_entrainment = 0;
        self.label_list[-1].kappa_base = 1e-4;
        self.label_list[-1].kappa_r_mod = 1;
        self.label_list[-1].kappa_s_mod = 1;
        self.label_list[-1].critical_slope = 0.57;
        self.label_list[-1].sensitivity_tool_effect = 1;

        self.label_list.append(ch.label(1))
        self.label_list[-1].m = 0.45;
        self.label_list[-1].n = 1;
        self.label_list[-1].base_K = 3e-6;
        self.label_list[-1].Ks_modifyer = 1;
        self.label_list[-1].Kr_modifyer = 0.3;
        self.label_list[-1].dimless_roughness = 0.5;
        self.label_list[-1].V = 0.1;
        self.label_list[-1].dstar = 1;
        self.label_list[-1].threshold_incision = 0;
        self.label_list[-1].threshold_entrainment = 0;
        self.label_list[-1].kappa_base = 1e-4;
        self.label_list[-1].kappa_r_mod = 1;
        self.label_list[-1].kappa_s_mod = 1;
        self.label_list[-1].critical_slope = 0.57;
        self.label_list[-1].sensitivity_tool_effect = 1;

        self.CHONK.initialise_label_list(self.label_list)
        self.CHONK.update_label_array(self.label_array.ravel())
        
        
@xs.process
class UpliftLandscape(chxl.Uplift):
	uplift_done = xs.variable(intent = "out")
	runner_done = xs.foreign(chxl.Runner, "runner_done")
	uplift = xs.variable(intent = 'out', dims = [('y','x'), ('node')])
	switch_time = xs.variable(intent = 'in')
	CHONK = xs.foreign(chxl.ChonkBase, "CHONK")
	active_nodes = xs.foreign(chxl.ChonkBase, "active_nodes")
	landscape = xs.foreign(CustomParameters, "landscape")

	def initialize(self):
		self.uplift = self.landscape.uplift_phase_1
		self.uplift[[-1,0],:] = 0
		self.done = False

	@xs.runtime(args=['step_delta','step_end'])
	def run_step(self, dt, timing):
		self.CHONK.add_external_to_surface_elevation_tp1(self.uplift.ravel() * dt)
		self.uplift_done = True
		
		if (timing > self.switch_time and self.done == False):
			self.uplift = self.landscape.uplift_phase_2
			self.uplift[[-1,0],:] = 0
			self.done = True
            
@xs.process
class UpliftLandscapeStSt(chxl.Uplift):
	uplift_done = xs.variable(intent = "out")
	runner_done = xs.foreign(chxl.Runner, "runner_done")
	uplift = xs.variable(intent = 'out', dims = [('y','x'), ('node')])
	switch_time = xs.variable(intent = 'in')
	CHONK = xs.foreign(chxl.ChonkBase, "CHONK")
	active_nodes = xs.foreign(chxl.ChonkBase, "active_nodes")
	landscape = xs.foreign(CustomParameters, "landscape")

	def initialize(self):
		self.uplift = self.landscape.uplift4StSt
		self.uplift[[-1,0],:] = 0
		self.done = False

	@xs.runtime(args=['step_delta','step_end'])
	def run_step(self, dt, timing):
		self.CHONK.add_external_to_surface_elevation_tp1(self.uplift.ravel() * dt)
		self.uplift_done = True
        
@xs.process
class UpliftLandscapeNF1(chxl.Uplift):
	uplift_done = xs.variable(intent = "out")
	runner_done = xs.foreign(chxl.Runner, "runner_done")
	uplift = xs.variable(intent = 'out', dims = [('y','x'), ('node')])
	switch_time = xs.variable(intent = 'in')
	CHONK = xs.foreign(chxl.ChonkBase, "CHONK")
	active_nodes = xs.foreign(chxl.ChonkBase, "active_nodes")
	landscape = xs.foreign(CustomParameters, "landscape")

	def initialize(self):
		self.uplift = self.landscape.uplift_NF
		self.uplift[[-1,0],:] = 0
		self.done = False

	@xs.runtime(args=['step_delta','step_end'])
	def run_step(self, dt, timing):
		self.CHONK.add_external_to_surface_elevation_tp1(self.uplift.ravel() * dt)
		self.uplift_done = True
            
# landscape.uplift4StSt


# In[3]:


model = xs.Model({"ChonkBase": chxl.ChonkBase,
                "Runner": chxl.Runner,
                "Topography": chxl.Topography,
#                 "Uplift": UpliftLandscapeStSt,
#                 "Uplift": UpliftLandscape,
#                 "Uplift": UpliftLandscapeNF1,
                "Uplift": chxl.Uplift,
                "Lake": chxl.Lake,
                "Precipitation": chxl.Precipitation,
#                 "DefaultParameters": chxl.DefaultParameters,
                "DefaultParameters": CustomParameters,
                "Flow": chxl.Flow,
                "Fluvial": chxl.Fluvial,
                "Hillslope": chxl.Hillslope
            })


# In[4]:


ny,nx = 100,100
dy,dx = 200,200


# In[5]:


# %create_setup model
import xsimlab as xs
time = np.arange(0,3e6,1000)
otime = time[::50]

init_z = np.load("./initial_topo_100_100.npy")
# init_z = np.random.rand(ny,nx)
# init_z[0,:] = 1000
U = np.zeros((ny,nx)) + 2e-4
Utime = 1e8
ds_in = xs.create_setup(
    model=model,
    clocks={
        'time': time,
        'otime': otime
    },
    master_clock='time',
    input_vars={
        'ChonkBase__dx': dx,
        'ChonkBase__dy': dy,
        'ChonkBase__nx': nx,
        'ChonkBase__ny': ny,
        'ChonkBase__depths_res_sed_proportions': 10,
        'ChonkBase__n_depth_sed_tracking': 50,
        'ChonkBase__boundary_conditions': "periodic_EW",
        'Topography__initial_elevation': init_z,
        'Uplift__uplift': U,
#         'Uplift__switch_time': Utime,
        'Lake__method': 'implicit',
        'Lake__evaporation': False,
        'Lake__evaporation_rate': 1,
        'Flow__threshold_single_flow': 1e7,
        'Precipitation__precipitation_rate': 0.0001
    },
    output_vars={
        'Topography__topography': 'otime',
        'Topography__sed_height': 'otime',
        'Flow__Qw': 'otime',
        'Flow__water_balance_checker': 'otime',
        'Lake__lake_depth': 'otime',
#         'Fluvial__Qs': 'otime',
        'Hillslope__Qs': 'otime',
    }
)


# In[6]:


zg = zarr.group("test_obj.zarr", overwrite=True)
with model,xs.monitoring.ProgressBar():
    out_ds = ds_in.xsimlab.run(store = zg)
#     out_ds = mod1.xsimlab.run()  
out_ds.x.values[0] = 0
out_ds.y.values[0] = 0


# In[ ]:


from ipyfastscape import TopoViz3d



app = TopoViz3d(out_ds, canvas_height=600, time_dim="otime", elevation_var = "Topography__topography" )

app.show()


# In[ ]:


#np.save("initial_topo_100_100.npy", out_ds.Topography__topography.values[-1])
np.save("sed4testHS_CHONK.npy", out_ds.Topography__sed_height.values[-1])


# In[ ]:


fig,ax = plt.subplots()
ax.plot(out_ds.otime.values, out_ds.Flow__water_balance_checker.values/(dx*dy*nx*ny*0.3))


# In[ ]:




