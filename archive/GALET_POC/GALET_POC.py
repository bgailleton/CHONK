import lsdnumbatools as lsdnb
import numpy as np
import numba as nb
import inspect

float_type = np.float32
int_type = np.int32

def rename_function_string(func,newname):
  """
    ingest a function and return a tstring of the source code with a new function name
  """
  argas = inspect.getfullargspec(func).args
  funclines = inspect.getsourcelines(func)
  return ''.join(['def %s('%(newname) + ', '.join(map(str, argas)) + '):\n'] +funclines[0][1:])

# sample functions for very simple cases
def funcsample1(a):
    return a

def funcsample2(i,a):
    return a[i]

##########################################


class GALET_POC(object):
  """GALET_POC would be the fastscape context doing the backend work"""
  def __init__(self):

    super(GALET_POC, self).__init__()

    # NAme of the main topo
    self.main_topography = None

    # will host the graph object from lsdnumbatools
    self.graph = None
    # graph speciations
    self.nx = None
    self.ny = None
    self.dx = None
    self.dy = None

    # Main dictionary containing a dictionary for each of the variable and processes
    self.meta_info = {}

    # The generated code
    self.code_string = None

    # List of quantities per type and dimensions
    self._quantity_int0D = []
    self._quantity_int1D = []
    self._quantity_int2D = []
    self._quantity_int3D = []
    self._quantity_float0D = []
    self._quantity_float1D = []
    self._quantity_float2D = []
    self._quantity_float3D = []

    # list of process functions to be added in the right order
    self._processes_jitted_function = []
    # Sma eindex than above containing the param
    self._processes_jitted_args = []
    self.nfuncs = 0

    # list of quantity string name
    self.all_quantity_names = []
    # All the quantities needing reinitialisation
    self.all_reinitialisable_names = []


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

    # Set orf param keys
    self.param_keys = set()

    # Set of all valid keywords
    self.valid_arg_keys = self.built_in_arg_keys.copy()


  def set_graph(self,nx,ny,dx,dy,node_type):
    """
    Initialise the graph
    """
    self.nx = nx
    self.ny = ny
    self.dx = dx
    self.dy = dy
    self.node_type = node_type
    # Finding the main topo
    which_list = self.type2list(self.meta_info[self.main_topography]["dtype"])
    tid = self.type2list(self.meta_info[self.main_topography]["index_array"])
    # running the graph routines
    self.graph = lsdnb.node_graph.graph(self.nx, self.ny, self.dx, self.dy, node_type = node_type, topography = which_list[tid])
    self.graph.compute_D8S_graph()
    self.graph.compute_D8M_graph()
    self.graph.correct_D8S_depressions()

  def run(self):
    """
    the running function UNFINISHED 
    """
    for name in self.all_reinitialisable_names:
      which_list = self.type2list(self.meta_info[name]["dtype"])
      tid = self.meta_info[name]["index_array"]
      which_list[tid] = np.zeros_like(which_list[tid])
    # WILL CALL THE RUN AND FINALISE FUNCTION HERE


  def _register_quantity(self, name, original_value, type_of_quantity = "quantity", 
    force_dtype = "f1d", need_delta = False, accumulative = False, tracker = False,
    is_main_topography = False, reinitialisable = False):
    """
    Register a new quantity depending on its type
    """

    if(is_main_topography):
      self.main_topography = name

    self.all_quantity_names.append(name)

    if(reinitialisable):
      self.all_reinitialisable_names.append(name)

    which_list = self.type2list(force_dtype)

    updater = {"type": type_of_quantity, "dtype":force_dtype, "index_array": len(which_list), 
    "tp1": need_delta, "accumulative": accumulative, "tracker": tracker}

    if(name not in self.meta_info):
      self.meta_info.update({name:updater})
    else:
      self.meta_info[name].update(updater)

    if(self.meta_info[name]["type"] in ["quantity_delta", "quantity_splitter"]):
      self.meta_info[name]["parent_index"] = self.meta_info[name[:-6]]["index_array"]
      self.meta_info[name]["parent_dtype"] = self.meta_info[name[:-6]]["dtype"]

    which_list.append(original_value)

    self.valid_arg_keys.add(name)

    if(need_delta):
      self._register_quantity(name+"_delta", np.copy(original_value), type_of_quantity = 'quantity_delta' ,force_dtype = force_dtype, reinitialisable = True)

    # Adding a dimension to the subarray
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

      self._register_quantity(name+"_split", np.full((list(original_value.shape).append(8)), -1, dtype = tdtype ), 
        type_of_quantity = 'quantity_splitter', force_dtype = tforce_dtype)

  def _register_param(self, param_name, value = None, dtype = "f0d"):
    # Similar than registering a quantity, but can be a function

    this_function_name = "_INTERNAL_param_" + str(self.nparams) + "_" + param_name
    self.nparams += 1
    self.meta_info[param_name] = {}
    self.meta_info[param_name]["function_name"] = this_function_name
    self.meta_info[param_name]["function_index"] = len(self._params_jitted_function)
    self.param_keys.add(param_name)
    self.valid_arg_keys.add(param_name)

    which_list = self.type2list(dtype)
    if(which_list is not None):
      self._register_quantity( param_name, value, type_of_quantity = "param", force_dtype = dtype, need_delta = False, accumulative = False, tracker = False)

      if("0d" in dtype):

        self._params_jitted_function.append(rename_function_string(funcsample1,this_function_name))
        self._params_jitted_args.append([this_function_name, param_name])

      elif("1d" in dtype):

        self._params_jitted_function.append(rename_function_string(funcsample2,this_function_name))
        self._params_jitted_args.append([this_function_name, "i",param_name])
      else:
        raise ValueError(dtype + " is not supported yet")

    elif(callable(value)):
      name, arguments = self.funcarg_parser(value)

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
    """
    ingest a function in the system
    """

    self.meta_info[name] = {}
    self.meta_info[name]["type"] = "process"
    this_function_name = "_INTERNAL_process_" + str(len(self._processes_jitted_function)) + "_" + name

    self.meta_info[name]["function_name"] = this_function_name
    self.meta_info[name]["function_index"] = len(self._processes_jitted_function)

    name, arguments = self.funcarg_parser(function)
    these_args = []
    these_args.append(this_function_name)
    # checker:
    for ar in arguments:
      if(ar not in self.valid_arg_keys):
        raise ValueError("Argument " + str(ar) + " not understood")

      if(ar in self.valid_arg_keys):
        these_args.append(ar)
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

    # Writing the finalisation function
    self._generate_finalising_code()


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
'''
This code has been automatically generated by GALET_MF:\n
Here is a small relaxing view before the struggle!

                                   /\\
                              /\\  //\\\\
                       /\\    //\\\\///\\\\\\        /\\
                      //\\\\  ///\\////\\\\\\\\  /\\  //\\\\
         /\\          /  ^ \\/^ ^/^  ^  ^ \\/^ \\/  ^ \\
        / ^\\    /\\  / ^   /  ^/ ^ ^ ^   ^\\ ^/  ^^  \\
       /^   \\  / ^\\/ ^ ^   ^ / ^  ^    ^  \\/ ^   ^  \\       *
      /  ^ ^ \\/^  ^\\ ^ ^ ^   ^  ^   ^   ____  ^   ^  \\     /|\\
     / ^ ^  ^ \\ ^  _\\___________________|  |_____^ ^  \\   /||o\\
    / ^^  ^ ^ ^\\  /______________________________\\ ^ ^ \\ /|o|||\\
   /  ^  ^^ ^ ^  /________________________________\\  ^  /|||||o|\\
  /^ ^  ^ ^^  ^    ||___|___||||||||||||___|__|||      /||o||||||\\       |
 / ^   ^   ^    ^  ||___|___||||||||||||___|__|||          | |           |
/ ^ ^ ^  ^  ^  ^   ||||||||||||||||||||||||||||||oooooooooo| |ooooooo  |
ooooooooooooooooooooooooooooooooooooooooooooooooooooooooo


'''

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

\tfor i in range(n_elements):
"""
    
    # writing the param and function calls
    for funcarg in self._processes_jitted_args:
      funame = funcarg[0]
      fuargs = funcarg[1:]

      self.code_string += "\n"
      self.code_string += "# Writing the process " + funame[18:]

      args2write = []

      for targ in fuargs:
        args2write.append(self._arg2code_writer(targ))
      self.code_string += "\n"
      self.code_string += "\t\t" + funame + "(" + ',\n\t\t\t'.join(map(str, args2write)) + ')\n'

    # Writting teh splitting props now
    self.code_string += "\n"
    self.code_string += "\t\tfor neight in range(D8Mnrecs[i]):\n"
    self.code_string += "\t\t\t"
    self.code_string += "rec = D8Mrecs[neight]\n"
    for qtt in self.all_quantity_names:
      if(self.meta_info[qtt]["type"] in ["quantity_splitter"]):
        self.code_string += "\t\t\t"
        datarr = self.type2list( self.meta_info[qtt]["dtype"], as_string_for_code_gen = True) 
        parentarr = self.type2list( self.meta_info[qtt]["parent_dtype"], as_string_for_code_gen = True) 
        self.code_string += parentarr
        self.code_string += "[" + str(int(self.meta_info[qtt]["parent_index"])) + ",rec] += " + datarr + "[" + str(int(self.meta_info[qtt]["index_array"])) + ",i,neight]"



  def _generate_finalising_code(self):

    self.code_string += """
@nb.njit()
def _finalise_step(n_elements, 
quantity_int0D,quantity_int1D,quantity_int2D,quantity_int3D,
quantity_float0D,quantity_float1D,quantity_float2D,quantity_float3D,
D8Srec,D8Sdist,D8Sdons,D8Sndons,D8Mrecs,D8Mnrecs,D8Mdist,D8Mdons,D8Mndons,D8Mdondist,SStack,MStack):

\tfor i in range(n_elements):
"""
    for qtt in self.all_quantity_names:
      if(self.meta_info[qtt]["type"] == "quantity_delta"):
        self.code_string += "\t\t"
        self.code_string += self.type2list( self.meta_info[qtt]["dtype"], as_string_for_code_gen = True)
        self.code_string += "[" + str(self.meta_info[qtt]["parent_index"]) + "]" + " += "
        self.code_string += self.type2list( self.meta_info[qtt]["dtype"], as_string_for_code_gen = True)
        self.code_string += "[" + str(self.meta_info[qtt]["index_array"]) + "]"




  def funcarg_parser(self, func):
    name = func.__name__
    arguments = inspect.getfullargspec(func).args
    return name, arguments

  def type2list(self, dtype, as_string_for_code_gen = False):
    which_list = None

    if dtype == "f1d":
      which_list = self._quantity_float1D
      if(as_string_for_code_gen):
        which_list = "quantity_float1D"
    elif dtype == "f2d":
      which_list = self._quantity_float2D
      if(as_string_for_code_gen):
        which_list = "quantity_float2D"
    elif dtype == "i1d":
      which_list = self._quantity_int1D
      if(as_string_for_code_gen):
        which_list = "quantity_int1D"
    elif dtype == "i2d":
      which_list = self._quantity_int2D
      if(as_string_for_code_gen):
        which_list = "quantity_int2D"
    elif dtype == "f3d":
      which_list = self._quantity_float3D
      if(as_string_for_code_gen):
        which_list = "quantity_float3D"
    elif dtype == "i3d":
      which_list = self._quantity_int3D
      if(as_string_for_code_gen):
        which_list = "quantity_int3D"
    elif dtype == "i0d":
      which_list = self._quantity_int0D
      if(as_string_for_code_gen):
        which_list = "quantity_int0D"
    elif dtype == "f0d":
      which_list = self._quantity_float0D
      if(as_string_for_code_gen):
        which_list = "quantity_float0D"

    return which_list

  def _arg2code_writer(self, targ, inside_param_call = False):
    """
    Function parts of the code generator toolchain
    DO NOT CALL OUTSIDE THE CODE GENERATOR TOOLCHAIN
    """

    if(targ in self.built_in_arg_keys):
      # IS BUILTIN
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


    elif(targ in self.param_keys and inside_param_call == False):
      # IS PARAM (not called from param)
      try:
        this_func_call = self.meta_info[targ]["function_name"] + "("
      except:
        raise ValueError(str(targ) + " does not have function_name")
      tid = self.meta_info[targ]["function_index"]
      internal_args = []
      for params in self._params_jitted_args[tid]:
        internal_args.append(self._arg2code_writer( params, inside_param_call = True))
      this_func_call += ','.join(map(str, internal_args)) + ')\n'
      return this_func_call

    elif(targ in self.valid_arg_keys):
      tid = self.meta_info[targ]["index_array"]
      arrayst = self.type2list( self.meta_info[targ]["dtype"], as_string_for_code_gen = True)
      return arrayst + "[" + str(int(tid)) + "]" 







































# END OF FILE








