"""
Module providing helping function to set up the model environment
The model environment manages everything related to the grid structure and the model functionment
I will also put here the processes related to 
"""
import numpy as np
import xsimlab as xs
import chonk as ch

description_depths_res_sed_proportions = """
Depth resolution for deposited sediment proportions.
When different labels are tracked, the model creates boxes for each pixels that gets sediment cover.
These boxes track the proportion of each label for all the sediments within this boxe. 
This parameter sets the depths size of these box, so that a nex boxe is created each time a box of this particular size is filled.
"""

@xs.process
class BaseModel2D:
	"""
		main model instance which controls all of the others
	"""

	# First, details about the model dimensions
	dx = xs.variable(description = "Spacing in X direction")
	dy = xs.variable(description = "Spacing in Y direction")
	nx = xs.variable(description = "Number of nodes in X direction")
	ny = xs.variable(description = "Number of nodes in Y direction")
	x = xs.variable(dims='x', intent='out')
	y = xs.variable(dims='y', intent='out')


	# Variables in
	topo = xs.variable(intent = 'inout', description = "Surface topography")
	H0 = xs.variable(intent = 'in', description = "Initial sediment thickness")
	depths_res_sed_proportions = xs.variable(intent = 'in', default = 1., description = description_depths_res_sed_proportions)
	lake_solver = xs.variable(intent = 'in', default = True, description = 'Switch on or off the lake management. True: the lake will be dynamically filled (or not) with water and sediment. False: water and sediment fluxes are rerouted from lake bottom to outlet.')
	boundary_conditions = xs.variable(intent = 'in', default = "periodic_EW", description = 'For the v1 of this model, only 3 options (WIP to add more flexibility): periodic_EW, periodic_NS and open')

	# what gets out
	model = xs.variable(intent = 'out', description = "The main model object, controls the c++ part (I/O, results, run function, process order of execution, ...)")

	methods_premove = xs.variable(intent = 'in', default = ["drainage_area"], decription = "All the 'passive' methods to apply before aplitting the water fluxes to the receivers, ordered")
	method_move = xs.variable(intent = 'in', default = "D8", description = "Method used to calculate how water is routed to the receivers")
	methods_post_move = xs.variable(intent = 'in', default = ["CHARLIE_I"], description = "All the 'active' methods to apply after splitting the water fluxes")

	def initialize(self):

		# Initialising dimentions
		self.x = np.arange(0,self.nx,self.dx)
		self.y = np.arange(0,self.ny,self.dy)

		# Initialising the model itself with default processes
		self.model = ch.ModelRunner( 1, ["move"], "") 
		# setting the lake usage
		self.model.set_lake_switch(self.lake_solver)
		# initialising the elevation
		self.model.update_array_double_param("surface_elevation", self.topo.ravel())
		self.model.update_array_double_param("surface_elevation_tp1", np.copy(self.topo.ravel()))
		# Initialising the sediment heigh
		self.model.update_array_double_param("sed_height", np.copy(self.H0.ravel()))
		self.model.update_array_double_param("sed_height_tp1", np.copy(self.H0.ravel()))
		# Conversing with the model
		self.model.update_double_param("dx", self.dx)
		self.model.update_double_param("dy", self.dy)
		self.model.update_int_param("n_rows", self.ny)
		self.model.update_int_param("n_cols", self.nx)
		self.model.update_int_param("n_elements", self.nx * self.ny)
		# setting the depths res for sediment tracking.
		self.model.update_double_param("depths_res_sed_proportions", self.depths_res_sed_proportions)

		if(self.boundary_conditions == "periodic_EW"):
			active = np.zeros((self.ny, self.nx), dtype = np.int)
			active[1:-1,:] = 1
			model.update_array_int_param("active_nodes", active.ravel())
		elif(self.boundary_conditions == "periodic_NW"):
			active = np.zeros((self.ny, self.nx), dtype = np.int)
			active[1:-1,1:-1] = 1
			model.update_array_int_param("active_nodes", active.ravel())
		else:
			active = np.zeros((self.ny, self.nx), dtype = np.int)
			active[1:-1,1:-1] = 1
			model.update_array_int_param("active_nodes", active.ravel())


	def run_step(self, dt):

		self.model.update_timestep(dt)
		self.model.update_array_double_param("surface_elevation", self.topo.ravel() )
		self.model.update_array_double_param("surface_elevation_tp1", np.copy(self.topo.ravel()) )
		self.model.initiate_nodegraph()
		self.model.run()
		self.topo = self.model.get_array_double_param("surface_elevation_tp1")








