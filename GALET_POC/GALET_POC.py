import lsdnumbatools as lsdnb
import numpy as np
import numba as nb
import inspects

float_type = np.float32
int_type = np.int32

def rename_function_string(func,newname):
  argas = inspect.getfullargspec(func).args
  funclines = inspect.getsourcelines(func)
  return ''.join(['def %s('%(newname) + ', '.join(map(str, argas)) + '):\n'] +funclines[0][1:])


class GALET_POC(object):
  """docstring for GALET_POC"""
  def __init__(self, dx, dy, node_type):

    super(GALET_POC, self).__init__()

    self.main_topography = None
    self.dx = dx
    self.dy = dy

    self.graph = None

    self.meta_info = {}

    self.code_string = None

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

    self.param_keys = {}

    self.valid_arg_keys = built_in_arg_keys.copy()


  def run(self):

    self.graph = lsdnb.node_graph.graph(topography.shape[1],topography.shape[0],dx,dy, node_type = topography, topography = node_type)
    self.graph.compute_D8S_graph()
    self.graph.compute_D8M_graph()
    self.graph.correct_D8S_depressions()

  def _register_quantity(self, name, original_value, type_of_quantity = "quantity", 
    force_dtype = "f1d", need_tp1 = False, accumulative = False, tracker = False,
    is_main_topography = False):

  if(is_main_topography):
    self.main_topography = name

    
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

    this_function_name = "_INTERNAL_param_" + self.nparams + "_" + param_name
    self.nparams += 1
    self.meta_info[param_name]["function_name"] = this_function_name
    self.meta_info[param_name]["function_index"] = len(self._params_jitted_function)
    self.param_keys.add(param_name)
    self.valid_arg_keys.add(param_name)

    which_list = type2list(dtype)
    if(which_list is not None):
      self._register_quantity( param_name, value,type_of_quantity = "param", force_type = dtype, need_tp1 = False, accumulative = False, tracker = False)

      if("0d" in dtype):

        # @nb.njit()
        def func(a):
          return a

        self._params_jitted_function.append(rename_function_string(func,this_function_name))
        self._params_jitted_args.append([this_function_name, param_name])

      elif("1d" in dtype):
        # @nb.njit()
        def func(i,a):
          return a[i]

        self._params_jitted_function.append(rename_function_string(func,this_function_name))
        self._params_jitted_args.append([this_function_name, "i",param_name])
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

        if(ar in self.valid_arg_keys):
          these_args.append(ar)
        else:
          raise ValueError("Something went wrong in the argument translation during the ingestion of parameter function")


      self._params_jitted_function.append(rename_function_string(value,this_function_name))
      self._params_jitted_args.append(these_args)

  def _register_process(self, name, function):

    self.meta_info[name]["type"] = "process"
    this_function_name = "_INTERNAL_process_" + len(self._processes_jitted_function) + "_" + name

    self.meta_info[name]["function_name"] = this_function_name
    self.meta_info[name]["function_index"] = len(self._processes_jitted_function)

    name, arguments = funcarg_parser(function)
    these_args = []
    these_args.append(this_function_name)
    # checker:
    for ar in arguments:
      if(ar not in self.valid_arg_keys):
        raise ValueError("Argument not understood")

      if(ar in self.built_in_arg_keys):
        these_args.append((ar,None))
      elif(ar in self.valid_arg_keys):
        these_args.append((name, self.meta_info["index_array"]))
      else:
        raise ValueError("Something went wrong in the argument translation during the ingestion of process function")

    self._processes_jitted_function.append(rename_function_string(function,this_function_name))
    self._processes_jitted_args.append(these_args)


  def build(self):
    # first step is to statically set the params
    self._numpyification_of_lists()

    # Adding the njit decorator to the functions
    self._add_decorators()

    # Starting the code generation with the imports and the param functions
    self._initiate_code()

    # writting the core running function now
    self._generate_running_code()


  def _numpyification_of_lists(self):
    """
    Compiles all the python lists of quantities into numpy array with static dimensions and dtype
    """
    self._quantity_int0D = np.array(self._quantity_int0D)
    self._quantity_int1D = np.array(self._quantity_int1D)
    self._quantity_int2D = np.array(self._quantity_int2D)
    self._quantity_int3D = np.array(self._quantity_int3D)
    self._quantity_float0D = np.array(self._quantity_float0D)
    self._quantity_float1D = np.array(self._quantity_float1D)
    self._quantity_float2D = np.array(self._quantity_float2D)
    self._quantity_float3D = np.array(self._quantity_float3D)

  def _add_decorators(self):

    for i in range(len(self._processes_jitted_function)):
      self._processes_jitted_function[i] = "@nb.njit()\n" + self._processes_jitted_function[i] 
    for i in range(len(self._processes_jitted_function)):
      self._params_jitted_function[i] = "@nb.njit()\n" + self._params_jitted_function[i] 

  def _initiate_code(self):
    """
    Starts a script with the imports and all the jitted functions
    """
    # Starting with the imports
    self.code_string = """
import numba as nb
import numpy as np
import math\n\n\n"""
    
    # Now writing all the param functions, which will be called later
    for func in self._params_jitted_function:
      for line in func:
        self.code_string += line
      self.code_string += "\n\n"

    # Now writing all the process functions, which will be called later
    for func in self._processes_jitted_function:
      for line in func:
        self.code_string += line
      self.code_string += "\n\n"

    self.code_string += "# --------------------------------------------- \n"
    self.code_string += "# --------------------------------------------- \n"
    self.code_string += "# ------------ MAIN FUNCTIONS BELOW ----------- \n"
    self.code_string += "# --------------------------------------------- \n"
    self.code_string += "# --------------------------------------------- \n"

  def _generate_running_code(self):

    self.code_string += "\n\n\n"
    # Writing the function def line
    self.code_string += """
@nb.njit()
def _internal_run(n_elements, 
quantity_int0D,quantity_int1D,quantity_int2D,quantity_int3D,
quantity_float0D,quantity_float1D,quantity_float2D,quantity_float3D,
D8Srec,D8Sdist,D8Sdons,D8Sndons,D8Mrecs,D8Mnrecs,D8Mdist,D8Mdons,D8Mndons,D8Mdondist,SStack,MStack):

for i in range(n_elements):
"""
    
    # writing the param and function calls
    for funcarg in _processes_jitted_args:
      funame = funcarg[0]
      fuargs = funcarg[1:]

      args2write = []

      for targ,tid in funcarg:
        args2write.append(self._arg2code_writer(targ))


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

  def _arg2code_writer(self, targ):
    """
    Function parts of the code generator toolchain
    DO NOT CALL OUTSIDE THE CODE GENERATOR TOOLCHAIN
    """

    if(targ in self.built_in_arg_keys):
      if(targ == "i"): # current node
        return "i"
      elif(targ == "receivers"):
        # all the downslopes receivers of the current node
        return "D8Mrecs[i,:D8Mnrecs[i]]"
      elif(targ == "distances_to_receivers"):
        # distances to receivers
        return "D8Mdist[i,:D8Mnrecs[i]]"
      elif(targ == "donors"):
        return "D8Mdons[i,:D8Mndons[i]]"
      elif(targ == "distances_to_donors"):
        return "D8Mdondist[i,:D8Mndons[i]]"
      elif(targ == "steepest_receiver"):
        return "D8Srec[i]"
      elif(targ == "distance_to_steepest_receiver"):
        return "D8Sdist[i]"
      else:
        raise ValueError("I cannot find the param in the _arg2code_writer")
    elif(targ in self.param_keys):






































# END OF FILE








