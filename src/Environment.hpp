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





// #####################################################
// ############# Labelling class #######################
// #####################################################

// the label class is a way to group every pixels having the same info and save memory.
// this is also the core concept of the tracking engine, which track and uses the label info to track the prop of each provenances area into the lagrangian particules
class labelz
{
public:
    // Default, empty constructor
    labelz(){};
    // Initialise a label class with an id corresponding to its place in the labelz vector
    labelz(int id){this->label_id = label_id;};
    // ID and place in the labelz vector
    int label_id;
    // integers attributes
    std::unordered_map<std::string, int> int_attributes;
    // floating points attributes
    std::unordered_map<std::string, double> double_attributes;
    // arrays of integers attributes
    std::unordered_map<std::string, xt::pytensor<int,1> > int_array_attributes;
    // arrays of floating points attributes
    std::unordered_map<std::string, xt::pytensor<double,1> > double_array_attributes;

    // Sets an integer attribute (update an old one or add a new one)
    void set_int_attribute(std::string key, int val){this->int_attributes[key] = val;};
    // Sets an floating attribute (update an old one or add a new one)
    void set_double_attribute(std::string key, double val){this->double_attributes[key] = val;};
    // Sets an integer array attribute (update an old one or add a new one)
    void set_int_array_attribute(std::string key, xt::pytensor<int,1> val){this->int_array_attributes[key] = val;};
    // Sets an floating array attribute (update an old one or add a new one)
    void set_double_array_attribute(std::string key, xt::pytensor<double,1> val){this->double_array_attributes[key] = val;}
};





// #####################################################
// ############# Node structure ########################
// #####################################################

// the class nodium is just used for the priority queue struture when solving lakes.
// it is a very small class that combine a node index and its elevation when I insert it within the PQ
// The operators are defined in the cpp file.
class nodium
{
  public:
    // empty constructor
    nodium(){};
    // Constructor by default
    nodium(int node,double elevation){this->node = node; this->elevation = elevation;};
    // Elevation data
    double elevation;
    // Node index
    int node;
};



// #####################################################
// ############# Lake ##################################
// #####################################################

// the lake class manages dynamically the filling of actual lakes: i.e. the part of depression filled with water and sediments
class Lake
{
  
  public:
    // Empty constructor
    Lake() {};
    // Default initialiser
    Lake(int lake_id)
    {this->lake_id = lake_id; n_nodes = 0; surface = 0; volume = 0; water_elevation = 0; outlet_node = -9999; nodes = std::vector<int>(); has_been_ingeted = -9999; volume_of_sediment = 0.; }

    // This functions ingest a whole existing lake into the current one *slurp*
    void ingest_other_lake(
       Lake& other_lake,
       std::vector<int>& node_in_lake, 
       std::vector<bool>& is_in_queue,
       std::vector<Lake>& lake_network
    );

    void pour_sediment_into_lake(double sediment_volume);

    void pour_water_in_lake(
      double water_wolume,
      int originode,
      std::vector<int>& node_in_lake,
      std::vector<bool>& is_processed,
      xt::pytensor<int,1>& active_nodes,
      std::vector<Lake>& lake_network,
      xt::pytensor<double,1>& surface_elevation,
      NodeGraphV2& graph,
      double cellarea,
      double dt,
      std::vector<chonk>& chonk_network
    );

    int check_neighbors_for_outlet_or_existing_lakes(
      nodium& next_node, 
      NodeGraphV2& graph, 
      std::vector<int>& node_in_lake, 
      std::vector<Lake>& lake_network,
      xt::pytensor<double,1>& surface_elevation,
      std::vector<bool>& is_in_queue,
      xt::pytensor<int,1>& active_nodes
    );

    double get_lake_depth_at_node(int node, std::vector<int>& node_in_lake);
    double set_lake_depth_at_node(int node, double value){depths[node] = value;};

    double get_lake_volume(){return this->volume;}
    double set_lake_volume(double value){ this->volume = value;}

    double get_volume_of_sediment(){return volume_of_sediment;}

    std::vector<int>& get_lake_nodes(){return nodes;}
    std::vector<int>& get_lake_nodes_in_queue(){return node_in_queue;}
    std::unordered_map<int,double>& get_lake_depths(){return depths;}
    std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& get_lake_priority_queue(){return depressionfiller;}
    int get_n_nodes(){return n_nodes;};
    int get_lake_id(){return lake_id;};
    int get_parent_lake(){return has_been_ingeted;}
    int set_parent_lake(int value){has_been_ingeted = value;}
    int get_lake_outlet(){return this->outlet_node;}
    std::vector<int> get_ingested_lakes(){return ingested_lakes;}
    chonk& get_outletting_chonk(){return outlet_chonk;};


  protected:
    // Lake ID, i.e. the lake place in the parent environment vector of lakes
    int lake_id;
    // Number of nodes in the lake/underwater
    int n_nodes;
    // The surface area of the lake in L^2
    double surface;
    // the volume of the lake in L^3
    double volume;
    // the absolute elevation of the water surface
    double water_elevation;
    //Sediments
    double volume_of_sediment;
    // The node outletting the lake
    int outlet_node;

    // outlet fluxes
    chonk outlet_chonk;

    // the index of the lake which ate this one
    int has_been_ingeted;
    std::vector<int> ingested_lakes; 
    // Vector of node in the lake
    std::vector<int> nodes;
    // Vector of nodes that are or have been in the queue
    std::vector<int> node_in_queue;
    // Vector of Depths in the lake
    std::unordered_map<int,double> depths;
    // The priority queue containing the nodes not in the lake yet but bordering the lake
    std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;

};



// The modelrunner manages the whole model run, it brings together the node graph, te chonks and the lakes while processing I/O and running the timesteps
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
    void cancel_fluxes_before_moving_prep(chonk& this_chonk);

    void manage_move_prep(chonk& this_chonk);

    void manage_fluxes_after_moving_prep(chonk& this_chonk);

    // int solve_depression(int node);
    int solve_depressionv2(int node);


    void process_inherited_water();

    void finalise();


    void find_underfilled_lakes_already_processed_and_give_water(int SS_ID, std::vector<bool>& is_processed );
    void process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep);
void process_node_nolake_for_sure(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep , bool need_flux_before_move);
    void find_nodes_to_reprocess(int start, std::vector<bool>& is_processed, std::vector<int>& nodes_to_reprocess, std::vector<int>& nodes_to_relake, int lake_to_avoid);


    // Accessing functions (so far only works when memory mode is normal)
    // # return the water flux at dt
    xt::pytensor<double,1> get_water_flux();
    // # retrun erosion flux in L/T with T
    xt::pytensor<double,1> get_erosion_flux();
    xt::pytensor<double,1> get_sediment_flux();
    // # return generic attribute
    xt::pytensor<double,1> get_other_attribute(std::string key);


    // DEBUGGING FUNCTIONS
    // ~ These have weird functionality you probably do not need, or very hacky slow process to check something works right
    // ~ Just Ignore
    void DEBUG_modify_double_array_param_inplace(std::string name, int place, double new_val){io_double_array[name][place] = new_val;}
    std::vector<int> DEBUG_get_receivers_at_node(int node){return this->graph.get_MF_receivers_at_node(node);}
    int DEBUG_get_Sreceivers_at_node(int node){return this->graph.get_Srec(node);}

    std::vector<std::vector<int> > get_DEBUG_connbas(){return this->graph.get_DEBUG_connbas();};
    std::vector<std::vector<int> > get_DEBUG_connode(){return this->graph.get_DEBUG_connode();};
    std::vector<int> get_mstree(){return this->graph.get_mstree();}
    std::vector<std::vector<int> > get_mstree_translated(){return this->graph.get_mstree_translated();}


    void DEBUG_check_weird_val_stacks();
    
    std::vector<std::string>& get_ordered_flux_method(){return ordered_flux_methods;}; 

    void set_lake_switch(bool value){lake_solver = value;}
    std::vector<int> get_broken_nodes(){return graph.get_broken_nodes();}

    // This function reinitialise the list of label to empty
    void reinitialise_label_list(){labelz_list.clear();};
    // Initialises a label list to n empty labels
    // void initialise_label_list(int n_labels){this->labelz_list.clear();this->labelz_list.reserve(n_labels);for(int i=0; i<n_labels; i++){this->labelz_list.emplace_back(labelz(i));}};
    void initialise_label_list(std::vector<labelz>& these_labelz){this->labelz_list = std::move(these_labelz); n_labels = int(these_labelz.size());};
    // returns the number of labels in the label list
    int get_n_labels(){return n_labels;}
    // get a list of a given attribute for each labels, this aims to minimise the calls to maps which is lower than looking in a vector. Especially when it would ned to be done for each nodes
    std::vector<int> get_list_of_int_labels_attribute(std::string key){std::vector<int> output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].int_attributes[key]);} return output;}
    std::vector<double> get_list_of_double_labels_attribute(std::string key){std::vector<double> output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].double_attributes[key]);} return output;}
    std::vector<xt::pytensor<double,1> > get_list_of_double_array_labels_attribute(std::string key){std::vector<xt::pytensor<double,1> > output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].double_array_attributes[key]);} return output;}
    std::vector<xt::pytensor<int,1> > get_list_of_int_array_labels_attribute(std::string key){std::vector<xt::pytensor<int,1> > output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].int_array_attributes[key]);} return output;}


  protected:

    // timestep of the model
    double timestep;
    double start_time;
    double current_time;

    // lake switch
    bool lake_solver;


    // All the methods affecting the fluxes in the right order you want to apply it 
    std::vector<std::string> ordered_flux_methods;
    
    // method related to move the water/sediment fluxes within each precipiton
    std::string move_method;

    // The nodegraph of the model
    NodeGraphV2 graph;

    // Chonk network
    std::vector<chonk> chonk_network;

    // method checkers
    std::map<std::string,bool> is_method_passive;

    // Lake Network
    //# This increments the lake vector
    int lake_incrementor;
    //# The vector containing all the different lake entities. Dynamically resized to the number of lakes
    std::vector<Lake> lake_network;
    //# Vetor containing the lake ID for each nodes of the landscape. -1 -> NAL node: Not A Lake
    std::vector<int> node_in_lake;

    // parameters, stored un maps of thingies by type
    std::map<std::string, int> io_int;
    std::map<std::string, double> io_double;
    std::map<std::string, xt::pytensor<int,1> > io_int_array;
    std::map<std::string, xt::pytensor<int,2> > io_int_array2d;
    std::map<std::string, xt::pytensor<double,1> > io_double_array ;
    std::map<std::string, xt::pytensor<double,2> > io_double_array2d;

    //Labellisation:
    int n_labels;
    std::vector<labelz> labelz_list;

  private:
    void create() {return;};
    void create(double ttimestep,double tstart_time, std::vector<std::string> tordered_flux_methods, std::string tmove_method);
     

};// End of ModelRunner








xt::pytensor<double,1> pop_elevation_to_SS_SF_SPIL(xt::pytensor<int,1>& stack, xt::pytensor<int,1>& rec,xt::pytensor<double,1>& length , xt::pytensor<double,1>& erosion, 
      xt::pytensor<double,1>& K, double n, double m, double cellarea);


#endif