import numpy as np
import xsimlab as xs
import CHONK_XL as chxl

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
        self.landscape.generate_uplift_Normal_fault(Upos = 1e-3, Uneg = 2e-3, alpha_pos = 2e3, alpha_neg = 0.8e4)
#         self.landscape.add_pluton( dimless_X = 0.6, dimless_Y = 0.3, half_width = 5000,  half_heigth = 3000)
        self.label_list = []
    
        self.label_array = self.landscape.indices

        self.label_list.append(ch.label(0))
        self.label_list[-1].m = 0.45;
        self.label_list[-1].n = 1;
        self.label_list[-1].base_K = 1e-4;
        self.label_list[-1].Ks_modifyer = 1.2;
        self.label_list[-1].Kr_modifyer = 0.8;
        self.label_list[-1].dimless_roughness = 0.5;
        self.label_list[-1].V = 0.5;
        self.label_list[-1].dstar = 1;
        self.label_list[-1].threshold_incision = 0;
        self.label_list[-1].threshold_entrainment = 0;
        self.label_list[-1].kappa_base = 1e-4;
        self.label_list[-1].kappa_r_mod = 0.8;
        self.label_list[-1].kappa_s_mod = 1.2;
        self.label_list[-1].critical_slope = 0.57835;
        self.label_list[-1].sensitivity_tool_effect = 1;

        self.label_list.append(ch.label(1))
        self.label_list[-1].m = 0.45;
        self.label_list[-1].n = 1;
        self.label_list[-1].base_K = 1e-4;
        self.label_list[-1].Ks_modifyer = 1;
        self.label_list[-1].Kr_modifyer = 0.3;
        self.label_list[-1].dimless_roughness = 0.5;
        self.label_list[-1].V = 0.1;
        self.label_list[-1].dstar = 1;
        self.label_list[-1].threshold_incision = 0;
        self.label_list[-1].threshold_entrainment = 0;
        self.label_list[-1].kappa_base = 1e-4;
        self.label_list[-1].kappa_r_mod = 0.8;
        self.label_list[-1].kappa_s_mod = 1.2;
        self.label_list[-1].critical_slope = 0.57835;
        self.label_list[-1].sensitivity_tool_effect = 1;

        self.CHONK.initialise_label_list(self.label_list)
        self.CHONK.update_label_array(self.label_array.ravel())


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

