import lsdnumbatools as lsdnb
import numpy as np
import numba as nb
import inspects

float_type = np.float32
int_type = np.int32

class GALET_POC(object):
  """docstring for GALET_POC"""
  def __init__(self, topography, dx, dy, node_type):

    super(GALET_POC, self).__init__()

    self.topography = topography
    self.dx = dx
    self.dy = dy

    self.graph = None

    self.meta_info = {}

    self._quantity_int0D = []
    self._quantity_int1D = []
    self._quantity_int2D = []
    self._quantity_int3D = []
    self._quantity_float0D = []
    self._quantity_float1D = []
    self._quantity_float2D = []
    self._quantity_float3D = []

    self._processes_jitted_function = []
    self._processes_jitted_args = []
    self.nfuncs = 0

    self._params_jitted_function = []
    self._params_jitted_args = []
    self.nparams = 0

    # set of valid values for the arguments
    self.built_in_arg_keys = {
    "i", # current node
    "receivers", # all the downslopes receivers of the current node
    "distances_to_receivers", # distances to receivers
    "donors", # all the upslopes donors
    "distances_to_donors", # distances to donors
    "steepest_receiver", # Steepest receiver
    "distance_to_steepest_receiver", # distances to steepest receiver
    }

    self.valid_arg_keys = built_in_arg_keys.copy()


  def run(self):

    self.graph = lsdnb.node_graph.graph(topography.shape[1],topography.shape[0],dx,dy, node_type = topography, topography = node_type)
    self.graph.compute_D8S_graph()
    self.graph.compute_D8M_graph()
    self.graph.correct_D8S_depressions()

  def _register_quantity(self, name, original_value, type_of_quantity = "quantity", force_dtype = "f1d", need_tp1 = False, accumulative = False, tracker = False):

    
    which_list = self.type2list(dtype)

    self.meta_info[name].update({"type": type_of_quantity, "dtype":force_dtype, "index_array": len(which_list), 
    "tp1": need_tp1, "accumulative": accumulative, "tracker": tracker})

    which_list.append(original_value)

    self.valid_arg_keys.add(name)

    if(need_tp1):
      self._register_quantity(name+"_tp1", force_dtype = force_dtype, np.copy(original_value))

    if(accumulative or tracker):
      if(force_dtype == "f1d"):
        tforce_dtype = "f2d"
        tdtype = float_type
      elif(force_dtype == "i1d"):
        tforce_dtype = "i2d"
        tdtype = int_type
      elif(force_dtype == "f2d"):
        tforce_dtype = "f3d"
        tdtype = float_type
      elif(force_dtype == "i2d"):
        tforce_dtype = "i3d"
        tdtype = int_type

      self._register_quantity(name+"_split", np.full((list(original_value.shape).append(8)), -1, dtype = tdtype ), force_dtype = tforce_type)

  def _register_param(self, param_name, value = None, dtype = "f0d"):

    this_function_name = "_INTERNAL_P_" + self.nparams + "_" + param_name
    self.nparams += 1
    self.meta_info[param_name]["function_name"] = this_function_name
    self.meta_info[param_name]["function_ID"] = len(self._params_jitted_function)

    which_list = type2list(dtype)
    if(which_list is not None):
      self._register_quantity( param_name, value,type_of_quantity = "param", force_type = dtype, need_tp1 = False, accumulative = False, tracker = False)

      if("0d" in dtype):

        @nb.njit()
        def f(a):
          return a

        self._params_jitted_function.append(inspect.getsource(f))
        self._params_jitted_args.append([this_function_name, (dtype),self.meta_info[param_name]["index_array"]])

      elif("1d" in dtype):
        @nb.njit()
        def f(i,a):
          return a[i]

        self._params_jitted_function.append(inspect.getsource(f))
        self._params_jitted_args.append([this_function_name, ("i",None),(dtype,self.meta_info[param_name]["index_array"])])
      else:
        raise ValueError(dtype + " is not supported yet")

    elif(callable(value)):
      name, arguments = funcarg_parser(value)

      these_args = []
      these_args.append(this_function_name)
      # checker:
      for ar in arguments:
        if(ar not in self.valid_arg_keys):
          raise ValueError("Argument not understood")

        if(ar in self.built_in_arg_keys):
          these_args.append((ar,None))
        elif(ar in self.valid_arg_keys):
          these_args.append((self.meta_info["dtype"], self.meta_info["index_array"]))
        else:
          raise ValueError("Something went wrong in the argument translation during the ingestion of parameter function")


      self._params_jitted_function.append(inspect.getsource(value))
      self._params_jitted_args.append(these_args)


              
# def f(i, proportions_in_Qsed):
#   return (proportions_in_Qsed[i,0] * 0.8 + proportions_in_Qsed[i,1] * 0.2) * 1e-5

# def SPIL(i, topography, receivers, distances_to_receivers, water_quantity, water_quantity_split, sediment_quantity, K, m, n, Erosion):
#   this_E = 0
#   for j in range(receivers.shape[0]):
#     this_E += np.power((topography[i] - topography[receivers[j]])/ distances_to_receivers[j], n) 
#     * np.power(water_quantity[i] * water_quantity_split[i,j],m)
#     * K
#   Erosion[i] += this_E

# @nb.njit()
# def model_runner(...):

#   for i in graph.topological_order:
#     ...

#   optional_finalising_function(...)
# model._register_param(self, param_name, value = None, dtype = "f0d")

# model._register_quantity(name, original_value, force_type = "f1d", need_tp1 = False, accumulative = False, tracker = False)




  def _register_process(self,func, function_name):
    # Getting the arguments
    arguments = inspect.getfullargspec(func).args

    self.meta_info[function_name] = {}
    self.meta_info[function_name]["original_arguments"] = arguments
    self.meta_info[function_name]["original_function"] = func


  def funcarg_parser(self, func):
    name = func.__name__
    arguments = inspect.getfullargspec(func).args
    return name, arguments

  def type2list(self, dtype):
    which_list = None

    if force_type == "f1d":
      which_list = self._quantity_float
    elif force_type == "f2d":
      which_list = self._quantity_float2D
    elif force_type == "i1d":
      which_list = self._quantity_int1D
    elif force_type == "i2d":
      which_list = self._quantity_int2D
    elif force_type == "f3d":
      which_list = self._quantity_float3D
    elif force_type == "i3d":
      which_list = self._quantity_int3D
    elif force_type == "i0d":
      which_list = self._quantity_int0D
    elif force_type == "f0d":
      which_list = self._quantity_float0D

    return which_list























    # to get outputs:

    # self.cppmodel.output_times:
    ## 1d array of time for each outputs

    # self.cppmodel.output_depthsFS
    ## 1D array (dim=time) of depths where the minimum factor of safety is for each timestep
    ## Might always be the surface that ones!!!

    # self.cppmodel.output_minFS
    ## 1D array (dim=time) giving the minimum FS of the whole depths column (its depths is given by self.cppmodel.output_depthsFS)
    
    # self.cppmodel.output_PsiFS
    ## 1darray (dim = ?) not sure how it gets PSI here ...

    # self.cppmodel.output_durationFS
    ## 1D array of dureation of each rain events (in seconds ?)

    # self.cppmodel.output_intensityFS
    ## !d array (dim = time) containing the corresponding intensities of rain events

    # self.cppmodel.output_failure_times
    ## 1darray (dim = time) erm the time corresponding to the output_failure_bool

    # self.cppmodel.output_failure_bool
    ## 1darray (dim = time) 1 if there is a failure at that timestep


    # self.cppmodel.output_failure_mindepths
    ## 1d array (dim = time): the minimum depth at which there is a factor of safety < 1 (WARNING, 9999 if no failure) 
    # self.cppmodel.output_failure_maxdepths
    ## 1d array (dim = time): the maximum depth at which there is a factor of safety < 1 (WARNING, 0 if no failure) 

    # self.cppmodel.output_Psi_timedepth
    ## 2D array (dims = time,depths) of Psi values
    # self.cppmodel.output_FS_timedepth
    ## 2D array (dims = time,depths) of factor of safety values

