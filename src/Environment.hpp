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
#include <queue> // for priority queues
#include <numeric> //for initialising variable to their respective max or min
#include <cmath>

//pybind11 headers, manages the link with python
#include "pybind11/pybind11.h"
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

// Other modules of the model
#include "nodegraph.hpp" // everything related to the node structure and relationships between them
#include "CHONK.hpp" // The particles object


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
// ############# Internal Node objects #################
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

//This class is used to sort nodes by their stack ID when reprocessing a landscape
class node_to_reproc
{
  public:
    // empty constructor
    node_to_reproc(){};
    // Constructor by default
    node_to_reproc(int node,int id_in_mstack){this->node = node; this->id_in_mstack = id_in_mstack;};
    // Id in the stack (low = upstream)
    int id_in_mstack;
    // Node index
    int node;
};


// #####################################################
// ############# Lake ##################################
// #####################################################

class LakeLite
{
public:
    LakeLite(){
                water_elevation = 0;
                volume_water = 0;
                volume_sed = 0;
                is_now = -1;
                outlet = -1;
                id = -1;
                this->sum_outrate = 0;
                nodes = std::vector<int>();
            };
    LakeLite(int id){
                water_elevation = 0;
                volume_water = 0;
                volume_sed = 0;
                is_now = -1;
                outlet = -1;
                this->id = id;
                this->sum_outrate = 0;
                nodes = std::vector<int>();
            };


    double water_elevation;
    double volume_water;
    double volume_sed;
    double sum_outrate;
    std::vector<int> nodes;
    std::vector<double> label_prop;
    int is_now;
    int outlet;
    int id;
};

class EntryPoint
{

public:
    EntryPoint(){this->volume_water = 0;this->volume_sed = 0;this->node = 0;}
    EntryPoint(int node){this->volume_water = 0;this->volume_sed = 0;this->node = node;}
    EntryPoint(double volume_water, double volume_sed, int node, std::vector<double> label_prop)
        {this->volume_water = volume_water;this->volume_sed = volume_sed;this->node = node;this->label_prop = label_prop;}
    double volume_water;
    double volume_sed;
    int node;
    std::vector<double> label_prop;
    void ingestNkill(EntryPoint& other);
};


// The modelrunner manages the whole model run, it brings together the node graph, te chonks and the lakes while processing I/O and running the timesteps
class ModelRunner
{
  public:
    // Default constructor
    ModelRunner() { create(); }

    // Full constructor
    ModelRunner(double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method) { create( ttimestep, tordered_flux_methods, tmove_method); }

    // update parameters, each of thes function are used to provide external parameters of each types to the model
    // int -> integer number,e.g. 1,34,654,3333
    void update_int_param(std::string name, int tparam_val){io_int[name] = tparam_val;};
    // double -> floating point number, e.g. -2.4, 2.8e65, 45.32
    void update_double_param(std::string name,double tparam_val){io_double[name] = tparam_val;};
    // array int -> numpy array/pytensor of integer data
    void update_array_int_param(std::string name,xt::pytensor<int,1>& tparam_val){io_int_array[name] = tparam_val;};
    // array2D int -> numpy array /pytensor data with 2 dimensions
    void update_array2d_int_param(std::string name,xt::pytensor<int,2>& tparam_val){io_int_array2d[name] = tparam_val;};
    // array double -> numpy array / pytensor of floating point number
    void update_array_double_param(std::string name,xt::pytensor<double,1>& tparam_val){io_double_array[name] = tparam_val;};
    // array2D double data -> numpy array/pytensor of floating point in 2 dimensions
    void update_array2d_double_param(std::string name,xt::pytensor<double,2>& tparam_val){io_double_array2d[name] = tparam_val;};
    

    // Update the timestep
    void update_timestep(double dt){timestep = dt;};

    // Get parameters, same principle than the update functions
    int get_int_param(std::string name){ return io_int[name];};
    double get_double_param(std::string name){return io_double[name];};
    xt::pytensor<int,1> get_array_int_param(std::string name){ return io_int_array[name];};
    xt::pytensor<int,2> get_array2d_int_param(std::string name){ return io_int_array2d[name];};
    xt::pytensor<double,1> get_array_double_param(std::string name){return io_double_array[name];};
    xt::pytensor<double,2> get_array2d_double_param(std::string name){return io_double_array2d[name];}; 

    // this function utilises the given topography to initiate the node graph object.
    // It also reinitialise the vectors of chonk to empty chonk, and 
    // the vector of lake to empty lakes. It processes the water from existing lakes before too.
    void initiate_nodegraph(); 

    // triggers the run for one timestep
    // It applies all the method in a given order and manages the processing of things
    void run();

    // Run all the given fluxes before moving preparation of a given particle:
    // This encompasses precipitation, inflitration, drainage area, ...
    void manage_fluxes_before_moving_prep(chonk& this_chonk, int label_id);
    // The model needs to be able to cancel these fluxes in order to reprocess them when needed
    void cancel_fluxes_before_moving_prep(chonk& this_chonk, int label_id);

    // Run the calculation deciding how to split the fluxes in the receiving nodes
    // It calculates vectors of propagation given the method 
    void manage_move_prep(chonk& this_chonk);

    // Run the method i the given order for fluxes after move preparation.
    // this encompasses all the fluxes requirng knowledge of where water and sediments are routed
    // Most of the erosional and depositional laws will be there: SPIM, CHarlie_I, hillslope diffusion, ...
    void manage_fluxes_after_moving_prep(chonk& this_chonk, int label_id);

    // Internal function processing preexisting pots of water
    // mainly lakes. It first check if the new topography has an outlet.
    // If it has an outlet, it reroute all the outflowing water to it
    // The rest of the water is given to the current node, simulating its transfer to the next-iteration lake in a rather cheap-but-working way
    void process_inherited_water();

    // The finalisation of the timestep (internal to the run function) applies all changes to the topography
    // It adds the deposition, remove erosion, process the lake sedimentation (TODO) and so on
    void finalise();


    // Extremely annoying functions to code... they manage the particular case of underfilled lakes receiving water
    // from one of their supposedly "downstream" neightbour. This function finds all the nodes and lakes impacted
    void find_underfilled_lakes_already_processed_and_give_water(int SS_ID, std::vector<bool>& is_processed );
    void find_nodes_to_reprocess(int start, std::vector<bool>& is_processed, std::vector<int>& nodes_to_reprocess, 
        std::vector<int>& nodes_to_relake,std::vector<int>& nodes_to_recompute_neighbors_at_the_end, int lake_to_avoid);

    // Main function processing a given node. Call the fluxe managing functions and takes care of lake management
    void process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep);

    // Avoid recursion by having this relatively small function processing node that I am sure won't need lake management
    void process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep , bool need_flux_before_move);
    void process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move, std::vector<int>& ignore_some);

    // Accessing functions (so far only works when memory mode is normal)
    // # return the water flux at dt
    xt::pytensor<double,1> get_water_flux();
    // # retrun erosion flux in L/T with T
    xt::pytensor<double,1> get_erosion_flux();
    xt::pytensor<double,1> get_sediment_flux();
    xt::pytensor<double,1> get_erosion_bedrock_only_flux();
    xt::pytensor<double,1> get_erosion_sed_only_flux();
    xt::pytensor<double,1> get_sediment_creation_flux();
    // # return generic attribute
    xt::pytensor<double,1> get_other_attribute(std::string key);
    
    std::vector<xt::pytensor<double,1> > get_label_tracking_results();

    xt::pytensor<double,2> get_superficial_layer_sediment_prop();

    xt::pytensor<int,1> get_lake_ID_array_raw();
    xt::pytensor<int,1> get_lake_ID_array();
    xt::pytensor<int,1> get_mstack_checker();


    // DEBUGGING FUNCTIONS
    // ~ These have weird functionality you probably do not need, or very hacky slow process to check something works right
    // ~ Just Ignore them please
    void DEBUG_modify_double_array_param_inplace(std::string name, int place, double new_val){io_double_array[name][place] = new_val;}
    std::vector<int> DEBUG_get_receivers_at_node(int node){return this->graph.get_MF_receivers_at_node(node);}
    int DEBUG_get_Sreceivers_at_node(int node){return this->graph.get_Srec(node);}
    std::vector<std::vector<int> > get_DEBUG_connbas(){return this->graph.get_DEBUG_connbas();};
    std::vector<std::vector<int> > get_DEBUG_connode(){return this->graph.get_DEBUG_connode();};
    std::vector<int> get_mstree(){return this->graph.get_mstree();}
    std::vector<std::vector<int> > get_mstree_translated(){return this->graph.get_mstree_translated();}
    void DEBUG_check_weird_val_stacks();
    

    // returns the ordered string of method to compute the fluxes
    std::vector<std::string>& get_ordered_flux_method(){return ordered_flux_methods;}; 
    void update_flux_methods(std::vector<std::string> methods){ordered_flux_methods = methods;}
    void update_move_method(std::string mots){move_method = mots;}

    // Activate or deactivate lake management in the model
    // Lake management = dynamic lake filling with water, sediments and lake sedimentation
    // No lake filling = fluxes automatically rerouted to the lake "natural" outlet following Cordonnier et al., 2018
    void set_lake_switch(bool value){lake_solver = value;}
    std::vector<int> get_broken_nodes(){return graph.get_broken_nodes();}


    // Functions managing the interactions with the label array: extracting the parameters and other related characteristics
    // This function reinitialise the list of label to empty
    void reinitialise_label_list(){labelz_list.clear();};
    // Initialises a label list to n empty labels
    // void initialise_label_list(int n_labels){this->labelz_list.clear();this->labelz_list.reserve(n_labels);for(int i=0; i<n_labels; i++){this->labelz_list.emplace_back(labelz(i));}};
    void initialise_label_list(std::vector<labelz> these_labelz);
    // returns the number of labels in the label list
    int get_n_labels(){return n_labels;}
    // Get a list of a given attribute for each labels, this aims to minimise the calls to maps which is lower than looking in a vector. Especially when it would ned to be done for each nodes
    // For example, you are processing the SPL, so you want a list of K for each label. Then you don't have to retrieve it from the maps at each iterations (accessing map elements is much slower than accessing vector elements)
    std::vector<int> get_list_of_int_labels_attribute(std::string key){std::vector<int> output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].int_attributes[key]);} return output;}
    std::vector<double> get_list_of_double_labels_attribute(std::string key){std::vector<double> output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].double_attributes[key]);} return output;}
    std::vector<xt::pytensor<double,1> > get_list_of_double_array_labels_attribute(std::string key){std::vector<xt::pytensor<double,1> > output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].double_array_attributes[key]);} return output;}
    std::vector<xt::pytensor<int,1> > get_list_of_int_array_labels_attribute(std::string key){std::vector<xt::pytensor<int,1> > output;output.reserve(n_labels);for(int i=0;i<n_labels;i++){output.emplace_back(labelz_list[i].int_array_attributes[key]);} return output;}
    // Preprocess the lsit of elements from labels
    void prepare_label_to_list_for_processes();


    // initialise the map of correspondances between model processes and integers for the switch engines
    void initialise_intcorrespondance();

    //update the label array
    void update_label_array(xt::pytensor<int,1>& arr){label_array = arr;};
    
    void add_to_sediment_tracking(int index, double height, std::vector<double> label_prop, double sed_depth_here);


    void add_external_to_double_array(std::string key,xt::pytensor<double,1>& adder){this->io_double_array[key] += adder;}


    double get_Qw_in() {return Qw_in;};
    double get_Qw_out() {return Qw_out;};
    double get_Ql_in() {return Ql_in;};
    double get_Ql_out() {return Ql_out;};
    double get_Qs_mass_balance() {return this->Qs_mass_balance;};

    xt::pytensor<int,1> get_flat_mask(){return this->graph.get_flat_mask();};
    void print_chonk_info(int node);

    inline void increment_new_lake(int& lakeid);

    std::map<int, std::vector<std::vector<double> > > get_sed_prop_by_label() {return sed_prop_by_label;};
    xt::pytensor<float,4> get_sed_prop_by_label_matrice(int n_depths);

   
    // New lake solver
    // Go through all the flat neighbors when originating a lake in order to deal with flat surfaces
    void original_gathering_of_water_and_sed_from_pixel_or_flat_area(int starting_node, double& water_volume, double& sediment_volume,
     std::vector<double>& label_prop, std::vector<int>& these_nodes);
    void iterative_lake_solver();
    int fill_mah_lake(EntryPoint& entry_point, std::queue<int>& iteralake);
    void drink_lake(int id_eater, int id_edible,EntryPoint& entry_point, std::queue<int>& iteralake);
    int motherlake(int this_lake_id);
    void reprocess_nodes_from_lake_outlet(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, EntryPoint& entry_point);
    void drape_deposition_flux_to_chonks();
    void check_what_gives_to_lake(int entry_node, std::vector<int>& these_lakid , std::vector<double>& twat, std::vector<double>& tsed, 
        std::vector<std::vector<double> >& tlab,  std::vector<int>& these_ET, int lake_to_ignore);
    void reprocess_nodes_from_lake_outlet_v2(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, EntryPoint& entry_point);

    void gather_nodes_to_reproc(std::vector<int>& local_mstack, 
  std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> >& ORDEEEEEER,
   std::vector<char>& is_in_queue, int outlet);
    chonk preprocess_outletting_chonk(chonk tchonk, EntryPoint& entry_point, int current_lake, int outlet,
 std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, std::map<int,std::vector<double> >& SL_corrector,
 std::vector<double>& pre_sed, std::vector<double>& pre_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre);
    void check_what_give_to_existing_outlets(std::map<int,double>& WF_corrector,  std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >&  SL_corrector, std::vector<int>& local_mstack);

    bool is_this_node_in_this_lake(int node, int tlake){bool out = false; int lakid = this->node_in_lake[node]; if(lakid>=0)lakid = this->motherlake(lakid);if(lakid==tlake)out = true;return out;}
    
    void check_what_give_to_existing_lakes(std::vector<int>& local_mstack, int outlet, int current_lake, std::vector<double>& this_sed,
   std::vector<double>& this_water, std::vector<int>& this_entry_node, std::vector<std::vector<double> >& label_prop_of_this);
    void deprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue);
    void reprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, int outlet, 
        int current_lake, std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >& SL_corrector);
    void unpack_entry_points_from_delta_maps(std::queue<int>& iteralake, std::vector<std::vector<double> >& label_prop_of_delta,
std::vector<double>& delta_sed, std::vector<double>& delta_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre,
std::vector<double>& pre_sed, std::vector<double>& pre_water);

    void label_nodes_with_no_rec_in_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, std::vector<char>& has_recs_in_local_stack);

    bool has_valid_outlet(int lakeid);

    std::vector<int> lake_in_order;
    std::vector<int> lake_status;
    

    xt::pytensor<int,1> get_debugint();

    // New setters:
    void set_surface_elevation(xt::pytensor<double,1>&& tsurface_elevation){this->surface_elevation = tsurface_elevation;}
    void set_surface_elevation_tp1(xt::pytensor<double,1>&& tsurface_elevation_tp1){this->surface_elevation_tp1 = tsurface_elevation_tp1;}
    void set_topography(xt::pytensor<double,1>&& ttopography){this->topography = ttopography;}
    void set_active_nodes(xt::pytensor<double,1>&& tactive_nodes){this->active_nodes = tactive_nodes;}



  protected:

    // timestep of the model
    double timestep;
    // Starting time (so far unused, will probably be managed in python)
    double start_time;
    // Curret time (see above)
    double current_time;

    // lake switch, if True: dynamic lake modelling
    // if false: fluxes rerouted from flux bottom to outlet
    bool lake_solver;

    // All the methods affecting the fluxes in the right order you want to apply it 
    // Important::it requires the strng "move" at the place at which the move method will be called
    std::vector<std::string> ordered_flux_methods;
    
    // method related to move the water/sediment fluxes within each precipiton
    std::string move_method;

    // The nodegraph of the model, see dedicated file for use
    NodeGraphV2 graph;

    // Chonk network: one chonk by node
    std::vector<chonk> chonk_network;

    // method checkers (deprecated???)
    std::map<std::string,bool> is_method_passive;

    // Lake Network
    //# This increments the lake vector
    int lake_incrementor;
    std::vector<int> node_in_lake;
    std::vector<char> has_been_outlet;

    std::vector<double> gave_to_lake_water;
    std::vector<double> gave_to_lake_sed;

    // Surface elevation at time t
    xt::pytensor<double,1> surface_elevation;
    // Surface elevation at next time step
    xt::pytensor<double,1> surface_elevation_tp1;
    // Current Surface elevation + lake depth
    xt::pytensor<double,1> topography;
    // active-node array, needed for node graphing around
    xt::pytensor<bool,1> active_nodes;




    // ONGOING DEPRECATION
    // parameters, stored un maps of thingies by type
    // these parameters are "model-wide parameters" like elevation, lake_depth or precipitation
    std::map<std::string, int> io_int;
    std::map<std::string, double> io_double;
    std::map<std::string, xt::pytensor<int,1> > io_int_array;
    std::map<std::string, xt::pytensor<int,2> > io_int_array2d;
    std::map<std::string, xt::pytensor<double,1> > io_double_array ;
    std::map<std::string, xt::pytensor<double,2> > io_double_array2d;


    //Deprecated:
    // //# The vector containing all the different lake entities. Dynamically resized to the number of lakes
    // std::vector<Lake> lake_network;
    //# Vetor containing the lake ID for each nodes of the landscape. -1 -> NAL node: Not A Lake

    // Debugging local mass_balance to identify sed leaks
    double sed_added_by_entry;
    double sed_added_by_prod;
    double sed_already_outletted;
    double sed_added_by_donors;
    double sed_outletting_system;


    //Labellisation:
    // Number of labels 
    int n_labels;
    // Vector containing all the different labels
    std::vector<labelz> labelz_list;
    // Maps of labels per_law utilise (not the most logical thing but it avoid a looooot of map access)
    std::unordered_map<std::string, std::vector<int> > labelz_list_int; 
    std::unordered_map<std::string, std::vector<double> > labelz_list_double; 
    std::unordered_map<std::string, std::vector<std::vector<int> > > labelz_list_int_array; 
    std::unordered_map<std::string, std::vector<std::vector<double> > > labelz_list_double_array; 
    // std::unordered_map<std::string, std::vector<int> > labelz_list_int; 

    // Correspondance between int and laws for speeding up with switched
    std::map<std::string,int> intcorrespondance;

    // Label array: because it is a systematic requirements, I need this aray to always be there
    xt::pytensor<int,1> label_array;

    xt::pytensor<int,1> debugint;


    // Discretisation of sediment height
    std::vector<bool> is_there_sed_here;
    std::map<int, std::vector<std::vector<double> > > sed_prop_by_label;

    // Parameters dealing with mass balance checks
    double Qw_in, Qw_out, Ql_in, Ql_out;
    double Qs_mass_balance, Qs_outlake_modyfier;

    std::vector<bool> is_processed;

    std::vector<LakeLite> lakes;
    std::vector<char> lake_is_in_queue_for_reproc;
    std::vector<EntryPoint> queue_adder_for_lake;

    std::unordered_map<int, std::vector<int> > original_outlet_giver;
    std::unordered_map<int, std::vector<double> > original_outlet_giver_water;
    std::unordered_map<int, std::vector<double> > original_outlet_giver_sed;
    std::unordered_map<int, std::vector<double> > original_outlet_giver_sedlab;
    std::unordered_map<int, chonk > original_chonk;
    std::vector<double> local_Qs_production_for_lakes;

    double DEBUG_GLOBDELT;
    double GLOB_GABUL;
    int n_outlets_remodelled;

  private:
    // mirror of the constructors
    void create() {return;};
    void create(double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method);
     

};// End of ModelRunner

#pragma once
namespace chonk_utilities
{
    // Random utilities
    bool has_duplicates(std::vector<int>& datvec);
}

// Ignore that at the moment
xt::pytensor<double,1> pop_elevation_to_SS_SF_SPIL(xt::pytensor<int,1>& stack, xt::pytensor<int,1>& rec,xt::pytensor<double,1>& length , xt::pytensor<double,1>& erosion, 
      xt::pytensor<double,1>& K, double n, double m, double cellarea);



// #####################################################
// ##### DEPRECATED OLD OBJECTS KEPT FOR LEGACY ########
// #####################################################

//

// // the lake class manages dynamically the filling of actual lakes: i.e. the part of depression filled with water and sediments
// class Lake
// {
  
//   public:
//     // Empty constructor
//     Lake() {};
//     // Default initialiser
//     Lake(int lake_id)
//     {this->lake_id = lake_id;ngested_nodes = 0; n_nodes = 0; surface = 0; volume = 0; water_elevation = 0; outlet_node = -9999; nodes = std::vector<int>(); has_been_ingeted = -9999; volume_of_sediment = 0.; }

//     // This functions ingest a whole existing lake into the current one *slurp*
//     void ingest_other_lake(
//        Lake& other_lake,
//        std::vector<int>& node_in_lake, 
//        std::vector<bool>& is_in_queue,
//        std::vector<Lake>& lake_network,
//        xt::pytensor<double,1>& topography
//     );

//     // this function add sediment volume to the lake 
//     void pour_sediment_into_lake(double sediment_volume, std::vector<double> label_prop);

//     // Core function of the lake dynamic: it pour water in the lake, empty or already bearing water
//     // It also detects if the lake has an outlet, and save its value if it does
//     void pour_water_in_lake(
//       double water_wolume,
//       int originode,
//       std::vector<int>& node_in_lake,
//       std::vector<bool>& is_processed,
//       xt::pytensor<int,1>& active_nodes,
//       std::vector<Lake>& lake_network,
//       xt::pytensor<double,1>& surface_elevation,
//       xt::pytensor<double,1>& topography,
//       NodeGraphV2& graph,
//       double cellarea,
//       double dt,
//       std::vector<chonk>& chonk_network,
//       double& Ql_out
//     );

//     // Internal function checking the neighbors of a given node to ingest them in the lake queue
//     // If an outlet is find, returns its node ID, otherwise -9999
//     int check_neighbors_for_outlet_or_existing_lakes(
//       nodium& next_node, 
//       NodeGraphV2& graph, 
//       std::vector<int>& node_in_lake, 
//       std::vector<Lake>& lake_network,
//       xt::pytensor<double,1>& surface_elevation,
//       std::vector<bool>& is_in_queue,
//       xt::pytensor<int,1>& active_nodes,
//       std::vector<chonk>& chonk_network,
//       xt::pytensor<double,1>& topography

      
//     );

//     // Return the depth of this lake at a given node
//     double get_lake_depth_at_node(int node, std::vector<int>& node_in_lake);
//     // Set the depth of this lake at a given node
//     double set_lake_depth_at_node(int node, double value){depths[node] = value;};

//     // returns the lake volume
//     double get_lake_volume(){return this->volume;}
//     // set the lae voume
//     double set_lake_volume(double value){ this->volume = value;}

//     // returns the total volume of sediment in the lake
//     double get_volume_of_sediment(){return volume_of_sediment;}

//     // return a vector of all nodes currently in this lake
//     std::vector<int>& get_lake_nodes(){return nodes;}
//     // returns a vector of all nodes currently in the queue, waiting to be assessed in the lake solving routine
//     std::vector<int>& get_lake_nodes_in_queue(){return node_in_queue;}

//     // returns the lake depth by node object
//     std::unordered_map<int,double>& get_lake_depths(){return depths;}
    
//     // returns the priority queue of the current lake
//     std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& get_lake_priority_queue(){return depressionfiller;}
    
//     // returns the number of nodes in the lake
//     int get_n_nodes(){return n_nodes;};
//     // Returns the lake ID
//     int get_lake_id(){return lake_id;};
//     // return the ID of the lake having ingested this lake, or -9999
//     int get_parent_lake(){return has_been_ingeted;}
//     // When ingested (and reinitialised) call this function to set this lake as now ingested by X
//     int set_parent_lake(int value){has_been_ingeted = value;}
//     // returns the current lake outlet node ID or -9999 if no outlet is given
//     int get_lake_outlet(){return this->outlet_node;}
//     // returns a vector of lake ID ingested by dat lake
//     std::vector<int> get_ingested_lakes(){return ingested_lakes;}
//     // Return the representative chonk of the lake, bearing its water flux and its sediment flux
//     chonk& get_outletting_chonk(){return outlet_chonk;};
//     std::vector<double> get_lake_lab_prop(){return lake_label_prop;};

//     void drape_deposition_flux_to_chonks(std::vector<chonk>& chonk_network, xt::pytensor<double,1>& surface_elevation, double timestep);

//     double get_water_elevation(){return water_elevation;};

//   protected:
//     // Lake ID, i.e. the lake place in the parent environment vector of lakes
//     int lake_id;
//     // Number of nodes in the lake/underwater
//     int n_nodes;
//     // The surface area of the lake in L^2
//     double surface;
//     // the volume of the lake in L^3
//     double volume;
//     // the absolute elevation of the water surface
//     double water_elevation;
//     //Sediments
//     double volume_of_sediment;
//     // The node outletting the lake
//     int outlet_node;

//     // Temporary node counter
//     int ngested_nodes;

//     // outlet fluxes, representative particule holding lakes characteristic in order to propagate it downstream
//     chonk outlet_chonk;

//     std::vector<double> lake_label_prop;

//     // the index of the lake which ate this one
//     int has_been_ingeted;
//     // Vector of lake having been ingested by this one
//     std::vector<int> ingested_lakes; 
//     // Vector of node in the lake
//     std::vector<int> nodes;
//     // Vector of nodes that are or have been in the queue
//     std::vector<int> node_in_queue;
//     // Vector of Depths in the lake
//     std::unordered_map<int,double> depths;
//     // The priority queue containing the nodes not in the lake yet but bordering the lake
//     std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;

// };

#endif