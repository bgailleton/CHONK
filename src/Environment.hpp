//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef Environment_HPP
#define Environment_HPP

// STL imports
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include <queue>
#include <iostream>
#include <numeric>
#include <cmath>
#include "pybind11/pybind11.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same

#include "nodegraph.hpp" // same
#include "CHONK.hpp" // same



// The simple model runner run a model with spatially uniform set of law. It allows 2D heterogeneities in specific parameters though (e.g. differential erodibility)
class ModelRunner
{
  public:
    // Default constructor
    ModelRunner() { create(); }

    // Full constructor
    ModelRunner(double ttimestep, double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method) { create( ttimestep, tstart_time, tordered_flux_methods, tmove_method); }

    // update parameters
    void update_int_param(std::string name, int tparam_val){io_int[name] = tparam_val;};
    void update_double_param(std::string name,double tparam_val){io_double[name] = tparam_val;};
    void update_array_int_param(std::string name,xt::pytensor<int,1>& tparam_val){io_int_array[name] = tparam_val;};
    void update_array2d_int_param(std::string name,xt::pytensor<int,2>& tparam_val){io_int_array2d[name] = tparam_val;};
    void update_array_double_param(std::string name,xt::pytensor<double,1>& tparam_val){io_double_array[name] = tparam_val;};
    void update_array2d_double_param(std::string name,xt::pytensor<double,2>& tparam_val){io_double_array2d[name] = tparam_val;};
    // # timestep
    void update_timestep(double dt){timestep = dt;};

    // Get parameters
    int get_int_param(std::string name){ return io_int[name];};
    double get_double_param(std::string name){return io_double[name];};
    xt::pytensor<int,1> get_array_int_param(std::string name){ return io_int_array[name];};
    xt::pytensor<int,2> get_array2d_int_param(std::string name){ return io_int_array2d[name];};
    xt::pytensor<double,1> get_array_double_param(std::string name){return io_double_array[name];};
    xt::pytensor<double,2> get_array2d_double_param(std::string name){return io_double_array2d[name];}; 

    void initiate_nodegraph();

    void run();

    void manage_fluxes_before_moving_prep(chonk& this_chonk);

    void manage_move_prep(chonk& this_chonk);

    void manage_fluxes_after_moving_prep(chonk& this_chonk);

    int solve_depression(int node);

    void process_inherited_water();

    void finalise();






    // Accessing functions (so far only works when memory mode is normal)
    // # return the water flux at dt
    xt::pytensor<double,1> get_water_flux();
    // # retrun erosion flux in L/T with T
    xt::pytensor<double,1> get_erosion_flux();
    xt::pytensor<double,1> get_sediment_flux();
    // # return generic attribute
    xt::pytensor<double,1> get_other_attribute(std::string key);
    //# return nodes in depression
    xt::pytensor<int,1> get_all_nodes_in_depression(){return graph.get_all_nodes_in_depression();}



    // DEBUGGING FUNCTIONS
    // ~ These have weird functionality you probably do not need, or very hacky slow process to check something works right
    // ~ Just Ignore
    void DEBUG_modify_double_array_param_inplace(std::string name, int place, double new_val){io_double_array[name][place] = new_val;}
    xt::pytensor<int,1> DEBUG_get_preacc(){return graph.DEBUG_get_preacc();}
    void DEBUG_check_weird_val_stacks();
    xt::pytensor<int,1> DEBUG_get_basin_label(){return graph.DEBUG_get_basin_label();}



  protected:

    // timestep of the model
    double timestep;
    double start_time;
    double current_time;


    // All the methods affecting the fluxes in the right order you want to apply it 
    std::vector<std::string> ordered_flux_methods;
    
    // method related to move the water/sediment fluxes within each precipiton
    std::string move_method;

    // The nodegraph of the model
    NodeGraph graph;

    // Chonk network
    std::vector<chonk> chonk_network;

    // method checkers
    std::map<std::string,bool> is_method_passive;

    // parameters
    std::map<std::string, int> io_int;
    std::map<std::string, double> io_double;
    std::map<std::string, xt::pytensor<int,1> > io_int_array;
    std::map<std::string, xt::pytensor<int,2> > io_int_array2d;
    std::map<std::string, xt::pytensor<double,1> > io_double_array;
    std::map<std::string, xt::pytensor<double,2> > io_double_array2d;

  private:
    void create() {return;};
    void create(double ttimestep,double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method);
     

};// End of ModelRunner


#endif