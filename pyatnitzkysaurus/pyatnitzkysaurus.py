"""
Early python binding to CHONK model
"""
import CHONK_cpp as ch
import numpy as np



class BaseModel(object):
	"""docstring for BaseModel"""
	def __init__(self, nrows):
		super(BaseModel, self).__init__()

		# Getting the base informations
		self.nrows = nrows
		self.ncols = ncols
		self.dx = dx
		self.dy = dy
		self.nx = ncols
		self.ny = nrows
		self.nn = nrows * nx
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmin + self.nx * self.dx
		self.ymax = ymin + self.ny * self.dy
		self.topo = topo.ravel()

		self.oredered_preprocesses = ["drainage_area"]
		self.oredered_processes = ["basic_SPIL"]
		self.move_method = "MF_fastscapelib_threshold_SF"


		# Creating the model 
		self.model = ch.ModelRunner( 1, 0, ["drainage_area","move","basic_SPIL"], "MF_fastscapelib_threshold_SF")
		self.model.set_lake_switch(True)

		# Feeding the model with stuff
		self.model.update_array_double_param("surface_elevation", self.topo)
		self.model.update_double_param("x_min", self.xmin)
		self.model.update_double_param("y_min", self.ymin)
		self.model.update_double_param("x_max", self.xmax)
		self.model.update_double_param("y_max", self.ymax)
		self.model.update_double_param("dx", self.dx)
		self.model.update_double_param("dy", self.dy)
		self.model.update_double_param("no_data", -9999)
		self.model.update_int_param("n_rows", self.nrows)
		self.model.update_int_param("n_cols", self.ncols)
		self.model.update_int_param("n_elements", self.nn)
		self.model.update_array_double_param("surface_elevation_tp1", np.copy(self.topo))
		self.model.update_array_double_param("sed_height", np.zeros_like(self.topo.ravel()))
		self.model.update_array_double_param("sed_height_tp1", np.zeros_like(self.topo.ravel()))

		# // Defaulting parameters for default laws
		self.model.update_double_param("SPIL_n", 1)
		self.model.update_double_param("SPIL_m", 0.45)
		self.model.update_double_param("threshold_single_flow", 1e6)
		self.model.update_array_double_param("erodibility_K", np.zeros_like(self.topo.ravel())+1e-4)

	def activate_lake(self):
		"""Activate the lake management"""
		self.model.set_lake_switch(True)

	def deactivate_lake(self):
		"""Deactivate the lake management"""
		self.model.set_lake_switch(False)

	def update_model_param(self,name, param):
		""""""
		if(isinstance(param, int)):
			self.model.update_int_param(name,param)
		elif(isinstance(param, float)):
			self.model.update_double_param(name,param)
		elif(isinstance(param,np.ndarray)):
			if(np.issubdtype(param.dtype, np.integer):
				if(len(param.shape) == 2):
					self.model.update_array2d_int_param(name,param)
				if(len(param.shape) == 1):
					self.model.update_array_int_param(name,param)

			if(np.issubdtype(param.dtype, np.floating):
				if(len(param.shape) == 2):
					self.model.update_array2d_double_param(name,param)
				if(len(param.shape) == 1):
					self.model.update_array_double_param(name,param)



















































			#end of file


		