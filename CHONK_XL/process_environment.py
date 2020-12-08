"""
Module providing helping function to set up the model environment
The model environment manages everything related to the grid structure and the model functionment
I will also put here the processes related to 
"""
import numpy as np
import xsimlab as xs
import CHONK_cpp as ch
from .hillshading import hillshading
import numcodecs
import zarr

@xs.process
class GridSpec:
	"""
		This grid object host the geometry in planview of the model: number of rows, cols, the conversions between nodes and x,y or the spacing
	"""
	dx = xs.variable(description = "Spacing in X direction")
	dy = xs.variable(description = "Spacing in Y direction")
	nx = xs.variable(description = "Number of nodes in X direction")
	ny = xs.variable(description = "Number of nodes in Y direction")
	x = xs.index(dims='x')
	y = xs.index(dims='y')
	node = xs.variable(dims='node', intent='out')

	def initialize(self):
		# Initialising dimentions
		self.x = np.arange(0,self.nx * self.dx,self.dx)
		self.y = np.arange(0,self.ny * self.dy,self.dy)
		self.node = np.arange(self.nx * self.ny)


@xs.process
class BoundaryConditions:
	"""
		Manage the boundary conditions in a rather basin way.
	"""

	boundary_conditions = xs.variable(description = "Method for boundary conditions")
	active_nodes = xs.variable(intent = 'out', dims = ('node'), description = "Array telling the model how to handle boundary conditions")
	nx = xs.foreign(GridSpec, 'nx')
	ny = xs.foreign(GridSpec, 'ny')

	def initialize(self):
		# Setting the boundary conditions
		if(self.boundary_conditions == "periodic_EW"):
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,:] = 1
		elif(self.boundary_conditions == "periodic_NW"):
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,1:-1] = 1
		else:
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,1:-1] = 1
		self.active_nodes = self.active_nodes.ravel()


@xs.process
class Topography:
	"""
		Create an initial topography with associated sediment height array
	"""
	nx = xs.foreign(GridSpec,'nx')
	ny = xs.foreign(GridSpec,'ny')
	active_nodes = xs.foreign(BoundaryConditions,'active_nodes')
	surface_elevation = xs.variable(intent = 'out', dims = ('node'), description = "The surface topography initialised as random noise")
	sed_height = xs.variable(intent = 'out', dims = ('node'), description = "initial non-layer of sediments")


@xs.process
class CustomInitialSurface(Topography):
	this_surface_elevation = xs.variable(intent = 'inout', dims = ('node'), description = "The surface topography initialised as random noise")
	recast_boundaries = xs.variable(intent = 'in', default = True)
	def initialize(self):
		self.surface_elevation = self.this_surface_elevation
		if(self.recast_boundaries):
			self.surface_elevation[self.active_nodes == 0] = 0
		self.sed_height = np.zeros_like(self.surface_elevation)


@xs.process
class RandomInitialSurface(Topography):
	"""
		Create an initial topography with associated sediment height array
	"""

	def initialize(self):
		self.surface_elevation = np.random.rand(self.nx * self.ny)
		self.surface_elevation[self.active_nodes == 0] = 0
		self.sed_height = np.zeros_like(self.surface_elevation)

@xs.process
class Flow:
	move_method = xs.any_object( description = "String identifyer for the move methods (e.g. D8, MF_fastscapelib_threshold_SF)")
	threshold_single_flow = xs.variable(intent = "inout", default = 1e6)


@xs.process
class D8Flow(Flow):
	"""
		TODO
	"""

	def initialize(self):
		self.move_method = "D8"


@xs.process
class MF2D8Flow(Flow):
	"""
		TODO
	"""


	def initialize(self):
		self.move_method = "MF_fastscapelib_threshold_SF"

@xs.process
class OrderedMethods:
	"""
		ToDO
	"""
	methods_pre_move = xs.variable(dims = ('n_methods_pre'), description = "List of methods that HAVE to be calculated BEFORE splitting the water. e.g. precipitation")
	methods_post_move = xs.variable(dims = ('n_methods_post'), description = "List of methods that HAVE to be calculated AFTER splitting the water. e.g. CHARLIE_I")

	method_string = xs.any_object()

	def initialize(self):
		self.method_string = np.concatenate([self.methods_pre_move, ["move"], self.methods_post_move]).tolist()


@xs.process
class Labelling:
	label_array = xs.variable(dims = ('y','x'))
	label_list = xs.any_object()

@xs.process
class StaticLabelling(Labelling):
	"""
		To Do
	"""


	def initialize(self):

		self.label_list = []
		self.label_list.append(ch.label(0))
		self.label_list[0].set_double_attribute("SPIL_m", 0.45);
		self.label_list[0].set_double_attribute("SPIL_n", 1);
		self.label_list[0].set_double_attribute("SPIL_K", 1e-5);
		self.label_list[0].set_double_attribute("CHARLIE_I_Kr", 1e-5);
		self.label_list[0].set_double_attribute("CHARLIE_I_Ks", 2e-5);
		self.label_list[0].set_double_attribute("CHARLIE_I_V", 1);
		self.label_list[0].set_double_attribute("CHARLIE_I_dimless_roughness", 1);
		self.label_list[0].set_double_attribute("CHARLIE_I_dstar", 1);
		self.label_list[0].set_double_attribute("CHARLIE_I_threshold_incision", 0);
		self.label_list[0].set_double_attribute("CHARLIE_I_threshold_entrainment", 0);


@xs.process
class Uplift:

	uplift = xs.variable(intent = 'inout', dims = 'node', description = "simple block uplift process, in m/yrs")


@xs.process
class BlockUplift(Uplift):
	"""
		Simple block uplift to be applied to the topography
	"""

	surface_elevation = xs.foreign(Topography, 'surface_elevation')
	active_nodes = xs.foreign(BoundaryConditions, 'active_nodes')

	def initialize(self):

		self.uplift = self.uplift.ravel()
		self.uplift[self.active_nodes == 0] = 0

@xs.process
class BlockUpliftCustomBounds(Uplift):
	"""
		Simple block uplift to be applied to the topography, but the boundary conditions are not corrected (ie, there can be vertical motions at outlets)
	"""

	surface_elevation = xs.foreign(Topography, 'surface_elevation')
	active_nodes = xs.foreign(BoundaryConditions, 'active_nodes')

	def initialize(self):

		self.uplift = self.uplift.ravel()




@xs.process
class CoreModel:
	"""
		main model instance which controls all of the others
	"""

	
	surface_elevation = xs.foreign(Topography, 'surface_elevation')
	sed_height = xs.foreign(Topography, 'sed_height')
	active_nodes = xs.foreign(BoundaryConditions, 'active_nodes')
	topolake = xs.variable(dims = ('y','x'), intent = 'out')

	dx = xs.foreign(GridSpec,'dx')
	dy = xs.foreign(GridSpec,'dy')
	ny = xs.foreign(GridSpec,'ny')
	nx = xs.foreign(GridSpec,'nx')

	uplift = xs.foreign(Uplift, 'uplift')

	threshold_single_flow = xs.foreign(Flow, 'threshold_single_flow')

	method_string = xs.foreign(OrderedMethods, 'method_string')
	move_method = xs.foreign(Flow, 'move_method')

	label_array = xs.foreign(Labelling, 'label_array')
	label_list = xs.foreign(Labelling, 'label_list')

	# Variables in
	depths_res_sed_proportions = xs.variable(intent = 'inout', description = "Depth resolution for saving sediments proportions") #, default = 1.
	lake_solver = xs.variable(intent = 'inout', description = 'Switch on or off the lake management. True: the lake will be dynamically filled (or not) with water and sediment. False: water and sediment fluxes are rerouted from lake bottom to outlet.') #, default = True
	n_depth_sed_tracking = xs.variable(intent = 'in', default = 50)
	n_depths_recorded = xs.index(dims = 'n_depths_recorded')
	
	# what gets out
	model = xs.any_object( description = "The main model object, controls the c++ part (I/O, results, run function, process order of execution, ...)")

	# OUTPUTS
	topo = xs.on_demand(dims = ('y','x'))
	Q_water = xs.on_demand(dims = ('y','x'))
	Q_sed = xs.on_demand(dims = ('y','x'))
	HS = xs.on_demand(dims = ('y','x'))
	lake_depth = xs.on_demand(dims = ('y','x'))
	sed_thickness = xs.on_demand(dims = ('y','x'))
	labprop_Qs = xs.on_demand(dims = ('n_labels','y','x'))
	labprop_superficial_layer = xs.on_demand(dims = ('n_labels','y','x'))
	E_r = xs.on_demand(dims = ('y','x'))
	E_s = xs.on_demand(dims = ('y','x'))
	sed_div = xs.on_demand(dims = ('y','x'))
	lake_id_raw = xs.on_demand(dims = ('y','x'))
	mstack_checker = xs.on_demand(dims = ('y','x'))
	flat_mask = xs.on_demand(dims = ('y','x'))
	NodeID =  xs.on_demand(dims = ('y','x'))
	full_sed_pile_prop = xs.on_demand(dims = ('y','x','n_depths_recorded','n_labels'))
	
	Qw_in = xs.on_demand()
	Qw_out = xs.on_demand()
	Ql_in = xs.on_demand()
	Ql_out = xs.on_demand()
	water_balance_checker = xs.on_demand()
	

	def initialize(self):
		self.n_depths_recorded = np.arange(0,self.depths_res_sed_proportions * self.n_depth_sed_tracking,self.depths_res_sed_proportions)

		# Initialising the model itself with default processes
		self.model = ch.ModelRunner( 1, ["move"], "") 

		self.model.update_double_param("dx", self.dx)
		self.model.update_double_param("dy", self.dy)
		self.model.update_int_param("ny", self.ny)
		self.model.update_int_param("n_rows", self.ny)
		self.model.update_int_param("nx", self.nx)
		self.model.update_int_param("n_cols", self.nx)
		self.model.update_int_param("n_elements",self.ny*self.nx)

		self.model.update_double_param("threshold_single_flow", self.threshold_single_flow)
		self.model.update_array_int_param("active_nodes", self.active_nodes)

		self.topolake = np.copy(self.surface_elevation).reshape(self.ny,self.nx) 

		self.model.update_array_double_param("surface_elevation", self.surface_elevation)
		self.model.update_array_double_param("surface_elevation_tp1", np.copy( self.surface_elevation))
		self.model.update_array_double_param("sed_height" , np.copy(self.sed_height))
		self.model.update_array_double_param("sed_height_tp1" , np.copy(self.sed_height))
		self.model.update_array_double_param("lake_depth" , np.zeros_like(self.sed_height))

		self.model.update_move_method(self.move_method)
		self.model.update_flux_methods(self.method_string)

		# setting the lake usage
		self.model.set_lake_switch(self.lake_solver)

		# setting the depths res for sediment tracking.
		self.model.update_double_param("depths_res_sed_proportions", self.depths_res_sed_proportions)

		# Setting the labelling
		self.model.initialise_label_list(self.label_list)
		self.model.update_label_array(self.label_array.ravel())

	@xs.runtime(args='step_delta')
	def run_step(self, dt):
		# print("Running", dt)
		self.model.update_timestep(dt)
		tempolake = self.model.get_array_double_param("surface_elevation")
		# self.model.update_array_double_param("surface_elevation", np.copy(self.model.get_array_double_param("surface_elevation_tp1") + np.random.rand(self.nx * self.ny) * 1e-7 ) )
		self.model.update_array_double_param("surface_elevation", np.copy(self.model.get_array_double_param("surface_elevation_tp1")) )
		self.model.update_array_double_param("sed_height", np.copy(self.model.get_array_double_param("sed_height_tp1")) )
		self.model.initiate_nodegraph()
		self.model.run()
		tempolake += self.model.get_array_double_param("lake_depth")
		self.topolake = tempolake.reshape(self.ny,self.nx)
		self.model.add_external_to_double_array("surface_elevation_tp1",self.uplift * dt)

	@topo.compute
	def _topo(self):
		return self.model.get_array_double_param("surface_elevation_tp1").reshape(self.ny,self.nx)

	@Q_water.compute
	def _Q_water(self):
		return self.model.get_water_flux().reshape(self.ny,self.nx)

	@Q_sed.compute
	def _Q_sed(self):
		return self.model.get_sediment_flux().reshape(self.ny,self.nx)

	@sed_thickness.compute
	def _sed_thickness(self):
		return self.model.get_array_double_param("sed_height_tp1").reshape(self.ny,self.nx)

	@lake_depth.compute
	def _lake_depth(self):
		lake = self.model.get_array_double_param("lake_depth").reshape(self.ny,self.nx)
		lake = np.abs(lake)
		lake[lake<=0] = np.nan
		return lake
	@lake_id_raw.compute
	def _lake_id_raw(self):
		lake = self.model.get_lake_ID_array_raw().reshape(self.ny,self.nx)
		return lake


	@HS.compute
	def _HS(self):
		tester = np.copy(self.model.get_array_double_param("surface_elevation_tp1")).reshape(self.ny,self.nx)
		this_HS = np.zeros_like(tester)
		hillshading(tester.reshape(self.ny,self.nx),self.dx,self.dy,self.nx,self.ny,this_HS,np.deg2rad(60),np.deg2rad(125),1)
		return this_HS.reshape(self.ny,self.nx)

	@labprop_Qs.compute
	def _labprop_Qs(self):
		import time
		LAB = np.array(self.model.get_label_tracking_results())
		temp = []
		for sub in LAB:
			temp.append(sub.reshape(self.ny,self.nx))
		return np.array(temp)

	@labprop_superficial_layer.compute
	def _labprop_superficial_layer(self):
		LAB = self.model.get_superficial_layer_sediment_prop()
		temp = []
		for sub in LAB:
			temp.append(sub.reshape(self.ny,self.nx))
		return np.array(temp)

	@E_r.compute
	def _E_r(self):
		return self.model.get_erosion_bedrock_only_flux().reshape(self.ny,self.nx)

	@E_s.compute
	def _E_s(self):
		return self.model.get_erosion_sed_only_flux().reshape(self.ny,self.nx)

	@sed_div.compute
	def _sed_div(self):
		return self.model.get_sediment_creation_flux().reshape(self.ny,self.nx)

	@mstack_checker.compute
	def _mstack_checker(self):
		return self.model.get_mstack_checker().reshape(self.ny,self.nx)



	@Qw_in.compute
	def _Qw_in(self):
		return self.model.get_Qw_in()
	@Qw_out.compute
	def _Qw_out(self):
		return self.model.get_Qw_out()
	@Ql_in.compute
	def _Ql_in(self):
		return self.model.get_Ql_in()
	@Ql_out.compute
	def _Ql_out(self):
		return self.model.get_Ql_out()


	@water_balance_checker.compute
	def _water_balance_checker(self):
		return self.model.get_Qw_in() - self.model.get_Qw_out() + self.model.get_Ql_in() - self.model.get_Ql_out()

	@flat_mask.compute
	def _flat_mask(self):
		return self.model.get_flat_mask().reshape(self.ny,self.nx);

	@NodeID.compute
	def _NodeID(self):
		return np.arange(0,(self.ny * self.nx)).reshape(self.ny,self.nx);

	@full_sed_pile_prop.compute
	def _full_sed_pile_prop(self):
		return  self.model.get_sed_prop_by_label_matrice(self.n_depth_sed_tracking)
		# return out.reshape(self.ny,self.nx)
		# z = zarr.empty(1, dtype=object, object_codec=numcodecs.JSON())
		# z[0] = temp
		# return z




























#end of file








