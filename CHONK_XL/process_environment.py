"""
Module providing helping function to set up the model environment
The model environment manages everything related to the grid structure and the model functionment
I will also put here the processes related to 
"""
import numpy as np
import xsimlab as xs
import CHONK_cpp as ch

@xs.process
class GridSpec:
	"""
		This grid object host the geometry in planview of the model: number of rows, cols, the conversions between nodes and x,y or the spacing
	"""
	dx = xs.variable(description = "Spacing in X direction", groups = "initial_param_double")
	dy = xs.variable(description = "Spacing in Y direction", groups = "initial_param_double")
	nx = xs.variable(description = "Number of nodes in X direction", groups = "initial_param_int")
	ny = xs.variable(description = "Number of nodes in Y direction", groups = "initial_param_int")
	x = xs.variable(dims='x', intent='out')
	y = xs.variable(dims='y', intent='out')
	node = xs.variable(dims='node', intent='out')

	def initialize(self):
		# Initialising dimentions
		self.x = np.arange(0,self.nx,self.dx)
		self.y = np.arange(0,self.ny,self.dy)
		self.node = np.arange(self.nx * self.ny)


@xs.process
class BoundaryConditions:
	"""
		Manage the boundary conditions in a rather basin way.
	"""

	boundary_conditions = xs.variable(description = "Method for boundary conditions")
	active_nodes = xs.variable(intent = 'out', dims = ('node'), description = "Array telling the model how to handle boundary conditions", groups = "initial_param_array_double")
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
	surface_elevation = xs.variable(intent = 'out', dims = ('node'), description = "The surface topography initialised as random noise", groups = "initial_param_array_double")
	sed_height = xs.variable(intent = 'out', dims = ('node'), description = "initial non-layer of sediments", groups = "initial_param_array_double")

@xs.process
class RandomInitialSurface(Topography):
	"""
		Create an initial topography with associated sediment height array
	"""

	def initialize(self):
		self.surface_elevation = np.random.rand(self.nx * self.ny)
		self.surface_elevation[self.active_nodes == 0] = 0
		self.sed_height = np.zeros_like(surface_elevation)

@xs.process
class Flow:
	move_method = xs.any_object( description = "String identifyer for the move methods (e.g. D8, MF_fastscapelib_threshold_SF)")


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

	threshold_single_flow = xs.variable(intent = "inout" , groups = "initial_param_double")

	def initialize(self):
		self.move_method = "MF_fastscapelib_threshold_SF"

@xs.process
class OrderedMethods:
	"""
		ToDO
	"""
	n_methods_pre = xs.variable(dims = ('n_methods_pre'), intent = 'out')
	n_methods_post = xs.variable(dims = ('n_methods_post'), intent = 'out')
	methods_pre_move = xs.variable(dims = ('n_methods_pre'), description = "List of methods that HAVE to be calculated BEFORE splitting the water. e.g. precipitation")
	methods_post_move = xs.variable(dims = ('n_methods_post'), description = "List of methods that HAVE to be calculated AFTER splitting the water. e.g. CHARLIE_I")

	method_string = xs.any_object()

	def initialize(self):
		n_methods_pre = np.arange(n_methods_pre.shape[0])
		n_methods_post = np.arange(n_methods_post.shape[0])
		self.method_string = np.concatenate([self.methods_pre_move, ["move"], self.methods_post_move]).tolist()


@xs.process
class StaticLabelling:
	"""
		To Do
	"""
	n_labels = xs.variable(dims = ('n_labels'), intent = 'out')
	label_array = xs.variable(dims = ('y','x'))
	label_list = xs.variable(dims = ('n_labels'))

	def initialize(self):
		n_labels = np.arange(label_list.shape[0])



@xs.process
class CoreModel:
	"""
		main model instance which controls all of the others
	"""

	
	surface_elevation = xs.foreign(Topography, 'surface_elevation')

	initial_param_double = xs.group('initial_param_double')
	initial_param_array_double = xs.group('initial_param_array_double')
	# initial_param_array_int = xs.group('initial_param_array_int')
	initial_param_int = xs.group('initial_param_int')

	method_string = xs.foreign(OrderedMethods, 'method_string')
	move_method = xs.foreign(Flow, 'move_method')

	label_array = xs.foreign(StaticLabelling, 'label_array')
	label_list = xs.foreign(StaticLabelling, 'label_list')

	# Variables in
	depths_res_sed_proportions = xs.variable(intent = 'inout', description = "Depth resolution for saving sediments proportions") #, default = 1.
	lake_solver = xs.variable(intent = 'inout', description = 'Switch on or off the lake management. True: the lake will be dynamically filled (or not) with water and sediment. False: water and sediment fluxes are rerouted from lake bottom to outlet.') #, default = True

	# what gets out
	model = xs.any_object( description = "The main model object, controls the c++ part (I/O, results, run function, process order of execution, ...)")

	def initialize(self):

		# Initialising the model itself with default processes
		self.model = ch.ModelRunner( 1, ["move"], "") 

		for key,val in self.initial_param_int.items():
			self.model.update_int_param(key,val)
		for key,val in self.initial_param_double.items():
			self.model.update_double_param(key,val)
		for key,val in self.initial_param_array_double.items():
			self.model.update_array_double_param(key,val)
		# for key,val in self.initial_param_array_int.items():
			# self.model.update_array_int_param(key,val)

		self.model.update_double_param("n_cols" , self.initial_param_double["nx"])
		self.model.update_double_param("n_rows" , self.initial_param_double["ny"])
		self.model.update_array_double_param("surface_elevation_tp1",  np.copy(self.initial_param_array_double["surface_elevation"]))
		self.model.update_array_double_param("sed_height_tp1" , np.copy(self.initial_param_array_double["sed_height"]))

		self.model.update_move_method(self.move_method)
		self.model.update_flux_methods(self.method_string)

		# setting the lake usage
		self.model.set_lake_switch(self.lake_solver)

		self.model.update_int_param("n_elements", self.initial_param_int['nx'] * self.initial_param_int['ny'] )

		# setting the depths res for sediment tracking.
		self.model.update_double_param("depths_res_sed_proportions", self.depths_res_sed_proportions)

		# Setting the labelling
		self.model.initialise_label_list(self.label_list)
		self.model.update_label_array(self.label_array.ravel())

	@xs.runtime
	def run_step(self, dt):

		self.model.update_timestep(dt)
		self.model.update_array_double_param("surface_elevation", self.surface_elevation.ravel() )
		self.model.update_array_double_param("surface_elevation_tp1", np.copy(self.surface_elevation.ravel()) )
		self.model.initiate_nodegraph()
		self.model.run()
		self.surface_elevation = self.model.get_array_double_param("surface_elevation_tp1")


@xs.process
class BlockUplift:
	"""
		Simple block uplift to be applied to the topography
	"""

	surface_elevation = xs.foreign(Topography, 'surface_elevation')
	active_nodes = xs.foreign(BoundaryConditions, 'active_nodes')
	uplift = xs.variable(intent = 'inout', description = "simple block uplift process, in m/yrs")

	def initialize(self):

		uplift = uplift.ravel()
		uplift[active_nodes == 0] = 0

	@xs.runtime
	def run_step(self, dt):
		self.surface_elevation += self.uplift * dt









