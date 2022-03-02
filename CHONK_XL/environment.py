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
import os


@xs.process
class ChonkBase(object):
	"""
	this is the base process containing the model c++ object
	It is fed to to most of the others to interact with the data
	It hosts the model but does not run it!
	"""

	# The following should be self-explanatory
	dx = xs.variable(description = "Spacing in X direction")
	dy = xs.variable(description = "Spacing in Y direction")
	nx = xs.variable(description = "Number of nodes in X direction")
	ny = xs.variable(description = "Number of nodes in Y direction")
	x = xs.index(dims='x')
	y = xs.index(dims='y')
	node = xs.index(dims='node')
	boundary_conditions = xs.variable()

	# the actual model
	CHONK = xs.any_object( description = "The c++ model object")
	# The boundary conditions
	active_nodes = xs.variable(intent = 'out', dims = ('node'), description = "Array telling the model how to handle boundary conditions")

	# Resolution in depth for the recording of sediment proportions
	depths_res_sed_proportions = xs.variable(intent = 'inout', description = "Depth resolution for saving sediments proportions") #, default = 1.
	# Number of thingy tracked
	n_depth_sed_tracking = xs.variable(intent = 'in', default = 50)
	# Associated index
	n_depths_recorded = xs.index(dims = 'n_depths_recorded')


	E_r = xs.on_demand( dims = ('y','x'), description = "Bedrock erosion (all processes)")
	E_s = xs.on_demand( dims = ('y','x'), description = "Sediment entrainment (all processes)")
	D_s = xs.on_demand( dims = ('y','x'), description = "Sediment deposition (all processes)")

	Q_sout_lab_N = xs.on_demand(dims = ('x','n_labels'), description = "Volume of suspended sediments outletting the model at the Northern boundary in m^3/yrs")
	Q_sout_lab_S = xs.on_demand(dims = ('x','n_labels'), description = "Volume of suspended sediments outletting the model at the Southern boundary in m^3/yrs")
	Q_sout_lab_E = xs.on_demand(dims = ('y','n_labels'), description = "Volume of suspended sediments outletting the model at the Eastern boundary in m^3/yrs")
	Q_sout_lab_W = xs.on_demand(dims = ('y','n_labels'), description = "Volume of suspended sediments outletting the model at the Western boundary in m^3/yrs")




	strati_recorder = xs.variable(default = False, description = "Enables the stratigraphy recording (proportions)")
	path_strati = xs.variable(default = "test", description = "Path to the folder holding this strati info (cannot store it in the zarr)")
	pref_strati = xs.variable(default = "test",description = "Common prefix to all strati files")
	create_folder_strati = xs.variable(default = False, description = "If True, will create the folder/replace the folder of path_strati")

	strati = xs.on_demand(default = False, description = "While it does not return anything meaningfull, it is needed to save the strati file at the right place/time")


	def initialize(self):
	

		# Initialising dimentions
		self.x = np.arange(0,self.nx * self.dx,self.dx)
		self.y = np.arange(0,self.ny * self.dy,self.dy)
		self.node = np.arange(0,self.ny * self.nx)
		self.n_depths_recorded = np.arange(0,self.depths_res_sed_proportions * self.n_depth_sed_tracking,self.depths_res_sed_proportions)
		
		# The model object (see c++ for explanations)
		self.CHONK = ch.ModelRunner( 1, ["move"], "")

		# Feeding the model with basic info
		self.CHONK.update_double_param("dx", self.dx)
		self.CHONK.update_double_param("dy", self.dy)
		self.CHONK.update_int_param("ny", self.ny)
		self.CHONK.update_int_param("n_rows", self.ny)
		self.CHONK.update_int_param("nx", self.nx)
		self.CHONK.update_int_param("n_cols", self.nx)
		self.CHONK.update_int_param("n_elements",self.ny*self.nx)
		self.CHONK.depths_res_sed_proportions =  self.depths_res_sed_proportions

		# Automating the boundary conditions
		if(self.boundary_conditions == "periodic_EW"):
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,:] = 1
		elif(self.boundary_conditions == "periodic_NS"):
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,1:-1] = 1
		else:
			self.active_nodes = np.zeros((self.ny, self.nx), dtype = np.int)
			self.active_nodes[1:-1,1:-1] = 1
		self.active_nodes = self.active_nodes.ravel()
		# And feedin them to the model
		self.CHONK.set_active_nodes( self.active_nodes)
		# Defaulting to no lake
		self.CHONK.lake_depth =  np.zeros((self.nx * self.ny))
		self.CHONK.set_lake_switch(False)


		# Stting the strati recorder options:
		if(self.strati_recorder):
			if(self.create_folder_strati):
				os.makedirs(self.path_strati, exist_ok=True ) #,self.pref_strati))
	
	@xs.runtime(args = 'step')
	def run_step(self, step):
		'''
		This class is more an data/model holder than anything, so the run fonction is static and does not do much
		'''
		# Simply saving the step number for ease in some other processes
		self.step = step


	@strati.compute
	def _strati(self):
		'''
		This function habndles the saving of strati data
		This is a bit convoluted but xarray/xsimlab don't manage variable length and the strati is varyuing all the time
		Older version of CHONK used to have a fixed size 3D matrix instead, but that is very memory-unefficient adn slow to construct and harder to process as the top and bottom of the matrix constantly changed

		'''
		# Checks if stratifyer is enabled

		if(self.strati_recorder):
			# mAnages the name: it makes sure the name is consistent: 00001, 00002, ..., 00122, ..., 11235, ... and not 1, 2, 3, 112. MacOS and Windows browser get panicked otherwise
			name = str(self.step)
			while(len(name) < 8):
				name = '0' + name

			# Error management: if there is literally no strati, it crashes.
			try:
				stuff = self.CHONK.get_stratiprop()

			# print(stuff)
				np.save(self.path_strati+ "/" + self.pref_strati + "_zs_" + name + ".npy",stuff[0])
				np.save(self.path_strati+ "/" + self.pref_strati + "_ncells_" + name + ".npy",stuff[1])
				np.save(self.path_strati+ "/" + self.pref_strati + "_props_" + name + ".npy",stuff[2])
				np.save(self.path_strati+ "/" + self.pref_strati + "_vol_" + name + ".npy",stuff[3])
				np.save(self.path_strati+ "/" + self.pref_strati + "_firstbox_" + name + ".npy",stuff[5])
			except:
				print("Strati missed step %s, if it happens at every output step, there is a problem with the path, prefix or landscape.", self.step)

		# Returning dummy stuff YOLO
		return self.step

	@D_s.compute
	def _D_s(self):
		'''
		Query and reshapes the total deposition rate (all processes)
		'''
		return self.CHONK.get_deposition_flux().reshape(self.ny,self.nx)

	@E_r.compute
	def _E_r(self):
		'''
		Query and reshapes the total bedrock incision rate (all processes)
		'''
		return self.CHONK.get_erosion_bedrock_only_flux().reshape(self.ny,self.nx)

	@E_s.compute
	def _E_s(self):
		'''
		Query and reshapes the total sediment entrainment fluxes (all processes)
		'''
		return self.CHONK.get_erosion_sed_only_flux().reshape(self.ny,self.nx)

	
	# Group of function managing the NSEW extraction of labels at outletting sediment flux		
	@Q_sout_lab_N.compute
	def _Q_sout_lab_N(self):
		return self.CHONK.get_Qsprop_bound("N")
	@Q_sout_lab_S.compute
	def _Q_sout_lab_S(self):
		return self.CHONK.get_Qsprop_bound("S")
	@Q_sout_lab_E.compute
	def _Q_sout_lab_E(self):
		return self.CHONK.get_Qsprop_bound("E")
	@Q_sout_lab_W.compute
	def _Q_sout_lab_W(self):
		return self.CHONK.get_Qsprop_bound("W")


@xs.process
class Runner(object):
	"""
		Class called when the model is ready to be ran.
		All the parameters and the rest are set in the model holder ChonkBase
	"""

	CHONK = xs.foreign(ChonkBase, "CHONK")
	runner_done = xs.variable(intent = "out", description = "Internal and debugging variable set to True if the model has ran at least once")

	@xs.runtime(args='step_delta')
	def run_step(self, dt):
		# Setting up the current time step (variable timestep can be ingested that way)
		self.CHONK.update_timestep(dt)
		# Set up the graph 
		self.CHONK.initiate_nodegraph()
		# Run the surface processes
		self.CHONK.run()
		# Done
		self.runner_done = True


@xs.process
class Uplift(object):
	'''
		Base class to mange surface uplift
		This one is generic and simply gets an external array of uplift rate
	'''


	uplift_done = xs.variable(intent = "out", description = 'Internal debugging variable,s et to True once the uplift has been applied')
	# Need the runner done (hack to make sure a certain order of things is executed)
	runner_done = xs.foreign(Runner, "runner_done")
	# Input uplift rate
	uplift = xs.variable(intent = 'in', dims = [('y','x'), ('node')], description = 'Uplift rates in [L/T]')
	# Model holder
	CHONK = xs.foreign(ChonkBase, "CHONK")
	# Boundary conditions array to check where the uplift needs to be cancelled
	active_nodes = xs.foreign(ChonkBase, "active_nodes")

	def initialize(self):
		self.uplift.ravel()[self.active_nodes == False] = 0

	@xs.runtime(args='step_delta')
	def run_step(self, dt):
		self.CHONK.add_external_to_surface_elevation_tp1(self.uplift.ravel() * dt)
		self.uplift_done = True


@xs.process
class Topography(object):

	initial_elevation = xs.variable(intent = 'in', dims = [('y','x'), ('node')])

	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")
	CHONK = xs.foreign(ChonkBase, "CHONK")
	runner_done = xs.foreign(Runner, "runner_done")
	uplift_done = xs.foreign(Uplift, "uplift_done")
	topo_updated = xs.variable(intent = "out")
	initial_carving = xs.variable(intent = 'in', default = True)


	topography = xs.on_demand( dims = ('y','x'))
	sed_height = xs.on_demand( dims = ('y','x'))

	def initialize(self):

		self.CHONK.set_surface_elevation(self.initial_elevation.ravel())
		self.CHONK.set_surface_elevation_tp1(np.copy( self.initial_elevation.ravel()))
		self.CHONK.set_sed_height(np.zeros_like(self.initial_elevation.ravel()))
		self.CHONK.set_sed_height_tp1(np.zeros_like(self.initial_elevation.ravel()))
		self.CHONK.initial_carving = self.initial_carving

	def finalize_step(self):
		self.CHONK.set_surface_elevation(self.CHONK.get_surface_elevation_tp1() )
		self.CHONK.set_sed_height(np.copy(self.CHONK.get_sed_height_tp1() ))
		self.topo_updated = True

	@topography.compute
	def _topography(self):
		return self.CHONK.get_surface_elevation_tp1().reshape(self.ny,self.nx)

	@sed_height.compute
	def _sed_height(self):
		return self.CHONK.get_sed_height_tp1().reshape(self.ny,self.nx)



@xs.process
class Lake(object):
	method = xs.variable(default = "implicit")
	Aincision_threshold = xs.variable(default = False)
	CHONK = xs.foreign(ChonkBase, "CHONK")
	evaporation = xs.variable()
	evaporation_rate = xs.variable(intent = 'in', dims = [('y','x'), 'node', ()])
	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")
	lake_depth = xs.on_demand(dims = ('y','x'))
	topolake = xs.on_demand(dims = ('y','x'))


	def initialize(self):
		if(self.method == 'implicit'):
			self.CHONK.set_lake_switch(False)
		else:
			self.CHONK.set_lake_switch(True)

		if(self.Aincision_threshold):
			self.CHONK.Ath_incision = True
		else:
			self.CHONK.Ath_incision = False


		if(self.evaporation):
			self.CHONK.lake_evaporation = True
			if(isinstance(self.evaporation_rate, np.ndarray)):
				self.CHONK.lake_evaporation_rate_spatial = self.evaporation_rate.ravel()
			else:
				self.CHONK.lake_evaporation_rate_spatial = np.full((self.ny,self.nx),self.evaporation_rate).ravel()

	@topolake.compute
	def _topolake(self):
		return self.CHONK.get_topography().reshape(self.ny,self.nx)
	
	@lake_depth.compute
	def _lake_depth(self):
		return self.CHONK.lake_depth.reshape(self.ny,self.nx)

@xs.process
class Precipitation(object):
	precipitation_rate = xs.variable(intent = 'in', dims = [('y','x'), 'node', ()] )
	CHONK = xs.foreign(ChonkBase, "CHONK")
	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")
	def initialize(self):
		self.CHONK.precipitations_enabled = True
		if(isinstance(self.precipitation_rate, np.ndarray)):
			self.CHONK.precipitations = self.precipitation_rate.ravel()
		else:
			self.CHONK.precipitations = np.full((self.ny,self.nx),self.precipitation_rate).ravel()


@xs.process
class DefaultParameters:
	label_array = xs.variable(intent = 'out', dims = (('y','x'), ('node')))
	label_list = xs.any_object()
	CHONK = xs.foreign(ChonkBase, "CHONK")
	active_nodes = xs.foreign(ChonkBase, "active_nodes")

	def initialize(self):
		self.label_list = []
		self.label_array = np.zeros_like(self.active_nodes, dtype = np.int32)
		self.label_list.append(ch.label(0))
		self.CHONK.initialise_label_list(self.label_list)
		self.CHONK.update_label_array(self.label_array.ravel())



@xs.process
class Flow(object):
	CHONK = xs.foreign(ChonkBase, "CHONK")
	threshold_single_flow = xs.variable(intent = "in")
	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")
	Qw = xs.on_demand(dims = ('y','x'))
	water_balance_checker = xs.on_demand()


	def initialize(self):
		self.CHONK.thresholdMF2SF = self.threshold_single_flow

	@Qw.compute
	def _Qw(self):
		return self.CHONK.get_water_flux().reshape(self.ny ,self.nx )

	@water_balance_checker.compute
	def _water_balance_checker(self):
		return self.CHONK.get_Qw_in() - self.CHONK.get_Qw_out() + self.CHONK.get_Ql_in() - self.CHONK.get_Ql_out()


@xs.process
class Fluvial(object):
	
	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")	
	CHONK = xs.foreign(ChonkBase, "CHONK")
	Qs = xs.on_demand(dims = ('y','x'))
	post_K = xs.on_demand(dims = ('y','x'))
	
	relative_erosivity_bed = xs.variable(default = False,description = 'Enable the "tool" effect via playing with the relative erosivity on the bedrocks if True (Default is False)')
	relative_erosivity_sed = xs.variable(default = False,description = 'Enable the "tool" effect via playing with the relative erosivity on the sediment layer if True (Default is False)')

	def initialize(self):
		self.CHONK.CHARLIE_I = True
		self.CHONK.tool_effect_rock = self.relative_erosivity_bed
		self.CHONK.tool_effect_sed = self.relative_erosivity_sed


	@Qs.compute
	def _Qs(self):
		return self.CHONK.get_fluvial_Qs().reshape(self.ny ,self.nx )

	@post_K.compute
	def _post_K(self):
		'''
		Get the K calculated post- tool effect
		'''

		# The try-catch structure is just in case this ouput is activated but K-calc is not. It then return a all-0 array
		# Also happens at the first time-step for some reasons
		try:
			return self.CHONK.get_K_calc().reshape(self.ny ,self.nx ) 
		except:
			return np.zeros((self.ny ,self.nx) ) 



@xs.process
class Hillslope(object):
	
	CHONK = xs.foreign(ChonkBase, "CHONK")
	nx = xs.foreign(ChonkBase, "nx")
	ny = xs.foreign(ChonkBase, "ny")

	Qs = xs.on_demand(dims = ('y','x'))


	def initialize(self):
		self.CHONK.CIDRE_HS = True

	@Qs.compute
	def _Qs(self):
		return self.CHONK.get_hillslope_Qs().reshape(self.ny ,self.nx )



minimal_model = xs.Model({"ChonkBase": ChonkBase,
													"Runner": Runner,
													"Topography": Topography,
													"Uplift": Uplift,
													"Lake": Lake,
													"DefaultParameters": DefaultParameters,
													"Flow": Flow,
													"Fluvial": Fluvial
												})





































#end of file
