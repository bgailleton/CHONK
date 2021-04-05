#ifndef Environment_CPP
#define Environment_CPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <ctime>
#include <fstream>
#include <functional>
#include <queue>
#include <limits>
#include <chrono>
#include <thread>

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/xadapt.hpp"


#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
// #include <pair>

#include <queue>

#include "Environment.hpp"
#include "CHONK.hpp"
#include "nodegraph.hpp"

#include <boost/timer/timer.hpp>

//#############################################################
//#############################################################
//################### Custom Operators ########################
//#############################################################
//#############################################################

// Managing the comparison operators the different small classes
// it is required by all priority queues
// lhs/rhs : left hand side, right hand side

// nodiums are sorted by elevations for the depression filler
bool operator>( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation > rhs.elevation;
}
bool operator<( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation < rhs.elevation;
}

// the nodes to reprocess are sorted by their index in the stack. Smaller = upstream
bool operator>( const node_to_reproc& lhs, const node_to_reproc& rhs )
{
  return lhs.id_in_mstack > rhs.id_in_mstack;
}
bool operator<( const node_to_reproc& lhs, const node_to_reproc& rhs )
{
  return lhs.id_in_mstack < rhs.id_in_mstack;
}

//#############################################################
//#############################################################
//################ Local Utility Functions ####################
//#############################################################
//#############################################################

// Small function utilised into the debugging
// return true if the vector has a duplicate
bool chonk_utilities::has_duplicates(std::vector<int>& datvec)
{
  std::set<int> countstuff;
  for( auto U : datvec)
  {
    if(countstuff.find(U) != countstuff.end())
      return true;

    countstuff.insert(U);
  }
  return false;
}

// Entry point ingesting functions (utilised in the iterative lake management)
void EntryPoint::ingestNkill(EntryPoint& other)
{
  int save_id = other.node;
  this->volume_water += other.volume_water;
  this->label_prop = mix_two_proportions(this->volume_sed, this->label_prop, other.volume_sed, other.label_prop);
  this->volume_sed += other.volume_sed;
  // this->node = other.node;
  other = EntryPoint(save_id);
}

// ######################################################################
// ######################################################################
// ######################################################################
// ###################### Model Runner ##################################
// ######################################################################
// ######################################################################

// The model runner is the main object controlling the code

// Initialises the model object, actually Does not do much but is required.
void ModelRunner::create(double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method)
{ 
  n_timers = 6;
  CHRONO_start = std::vector<std::chrono::high_resolution_clock::time_point>(n_timers);
  CHRONO_stop = std::vector<std::chrono::high_resolution_clock::time_point>(n_timers);
  CHRONO_mean = std::vector<double>(n_timers,0.);
  CHRONO_n_time = std::vector<int>(n_timers,0);
  CHRONO_name = {"init_graph","total_run", "passive", "active", "splitNmerge", "move_prep"};

  // std::cout << "THING1" << std::endl;
  // Saving all the attributes
  this->timestep = ttimestep;
  this->ordered_flux_methods = tordered_flux_methods;
  this->move_method = tmove_method;
  // By default the lake solver is activated
  this->lake_solver = true;
  // Initialising the labelling stuff
  this->initialise_intcorrespondance();
  this->prepare_label_to_list_for_processes(); 
  this->Qw_in = 0;
  this->Qw_out = 0;
  this->Ql_in = 0;
  this->Ql_out = 0;
  // std::cout << "THING2" << std::endl;
  // Flushub tuc
  // galg
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{
  CHRONO_start[0] = std::chrono::high_resolution_clock::now();

  // std::cout << "initiating nodegraph..." <<std::endl;
  // Creating the nodegraph and preprocessing the depression nodes
  this->topography = xt::pytensor<double,1>(this->surface_elevation);
  this->dx = this->io_double["dx"];
  this->dy = this->io_double["dy"];
  this->cellarea = this->dx * this->dy;


  // Dat is the real stuff:
  // Initialising the graph
  this->graph = NodeGraphV2(this->surface_elevation, this->active_nodes,this->dx, this->dy,
                            this->io_int["n_rows"], this->io_int["n_cols"], this->lake_solver);
  
  // Chonkification: initialising chonk network
  //# clearing the chonk network
  if(this->chonk_network.size()>0)
  {
    this->chonk_network.clear();
  }
  //# Creating noew empty chonks
  this->chonk_network = std::vector<chonk>();
  this->chonk_network.reserve(size_t(this->io_int["n_elements"]));

  // filling my network with empty chonks
  for(size_t i=0; i<size_t(this->io_int["n_elements"]); i++)
  {
    this->chonk_network.emplace_back(chonk(int(i), int(i), false));
    this->chonk_network[i].initialise_local_label_tracker_in_sediment_flux(this->n_labels);
  }

  // Initialising water balance
  this->Qw_in = 0;
  this->Qw_out = 0;
  this->Ql_in = 0;
  this->Ql_out = 0;
  // And the sediment one
  this->Qs_mass_balance = 0;

  // Processes preexisting water from previous time steps, so far: lakes
  this->process_inherited_water();

  // Also initialising the Lake graph
  //# incrementor reset to 0
  lake_incrementor = 0;
  this->lakes.clear();
  //# no nodes in lakes
  node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);

  // I need the topoogical order of my depressions: which depressions will i get first
  this->lake_in_order = this->graph.get_Cordonnier_order();
  // Initialising the lake status array
  this->lake_status = std::vector<int>(this->io_int["n_elements"],-1);
  // Initialising the depression pits to 0
  for(auto tn:lake_in_order)
  {
    this->lake_status[tn] = 0;
  }

  // ready to run this time step!
  // std::cout << "THING3" << std::endl;
  CHRONO_stop[0] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[0] ++;
  CHRONO_mean[0] += std::chrono::duration<double>(CHRONO_stop[0] - CHRONO_start[0]).count();

}

// this is the main running function
void ModelRunner::run()
{
  CHRONO_start[1] = std::chrono::high_resolution_clock::now();

  // Keeping track of which node is processed, for debugging and lake management
  is_processed = std::vector<bool>(io_int["n_elements"],false);
  this->local_Qs_production_for_lakes = std::vector<double>(this->io_int["n_elements"],0);

  double cellarea = this->dx * this->dy;

  // Debug checker
  int underfilled_lake = 0;

  // ########### Strating the calculation #############
  // ## Iterating though all the nodes ##
  for(int i=0; i<io_int["n_elements"]; i++)
  {
    // Getting the current node in the Us->DS stack order
    int node = this->graph.get_MF_stack_at_i(i);

    // Processing that node
    // ### Saving the local production of sediment, in order to cancel it later
    double templocalQS = this->chonk_network[node].get_sediment_flux(); 
    // std::cout << "Processing node " << i << std::endl;
    this->process_node(node, is_processed, lake_incrementor, underfilled_lake, this->active_nodes, cellarea, surface_elevation, true);   
    // ### Saving the local production of sediment, in order to cancel it later
    this->local_Qs_production_for_lakes[node] = this->chonk_network[node].get_sediment_flux() - templocalQS; 

    // Switching to the next node in line
  }

  // First pass is done, all my nodes have been processed once. The flux is done is lake solver is implicit and we can finalise.

  // If the lake solver is explicit though, I can start the iterative process
  if(this->lake_solver)
  {
    // Debug variable to ignore
    DEBUG_GLOBDELT = 0;
    // Running the iterative lake solver
    this->iterative_lake_solver();
  }

  // Calling the finalising function: it applies the changes in topography and I think will apply the lake sedimentation
  this->finalise();
  // Done

  CHRONO_stop[1] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[1] ++;
  CHRONO_mean[1] += std::chrono::duration<double>(CHRONO_stop[1] - CHRONO_start[1]).count();



  std::cout << std::endl << "--------------------- START OF TIME REPORT ---------------------" << std::endl;
  for(int i=0; i< this->n_timers; i++)
    std::cout << CHRONO_name[i] << " took " << double(CHRONO_mean[i])/CHRONO_n_time[i] << " seconds out of " << CHRONO_n_time[i] << " runs" << std::endl;
  std::cout << "--------------------- END OF TIME REPORT ---------------------" << std::endl << std::endl;
  // Old debug statement to let the model catch its breath when there is shit ton of cout statements 
  // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}


void ModelRunner::iterative_lake_solver()
{
  // Right, this is a complex function. Solving lakes explicitely while retaining the dynamic adaptation of parameter (ie not solving water first, then fluvial erosion, 
  // then deposion, then tracking,... but all at each pixel movement) ended up being a pain. But it works and is reasonnably quick so I am happy.


  // Initialising an empty queue of entry points, ie, points which will initiate a lake with a given sed and wat content
  // Basically, each time a reprocessing ends up in a lake, I add a entry point. This is a FIFO queue processing all of them until it is done.
  // POSSIBLE OPTIMISATION: swithc to a priority queue based on a the depression topological order. 
  // This is already the case for the first insertion and most of the next one are subsequently sorted but potentially not all
  std::queue<int> iteralake;

  // reinitialising the queue helpers
  // the queue helpers just store everything that needs to go in a specific lake,  so if multiple entry points are stored for 
  // a single lake and it comes to processing, it merges all of them rather than reprocessing multiple times
  lake_is_in_queue_for_reproc.clear(); 
  queue_adder_for_lake.clear();


  // Debugger to ignore
  GLOB_GABUL = 0;
  this->n_outlets_remodelled = 0;


  // Initialising the lake_status array: -1 = not a lake; 0 = to be processed and 1= processed at least once
  // setting all potential entry_points to 0
  for(auto starting_node : this->lake_in_order)
    this->lake_status[starting_node] = 0;

  // reinitialising the lakes
  this->lakes = std::vector<LakeLite>();
  // Traces which nodes of the landscape has been an outlet or not
  this->has_been_outlet = std::vector<char>(this->io_int["n_elements"],'n');

  // KEEPING FOR LEGACY BUT I THINK IS DEPRECATED NOW
  // reinitialising all the maps dealing with saving outlet original status
  original_outlet_giver.clear();
  original_outlet_giver_water.clear();
  original_outlet_giver_sed.clear();
  original_outlet_giver_sedlab.clear();
  original_chonk.clear();
  // Best,
  // Boris
  

  //############# First step: initialising the original lakes in the topological order

  // Initialising the queue with the first lakes. 
  // Also I am preprocessing the flat surfaces: if a lake is juxtaposed by nodes with the exact same elevation, I merge all of these in the same lake
  for(auto starting_node : this->lake_in_order)
  {

    // Check if already incorporated into another flat surface
    if(this->lake_status[starting_node] > 0)
      continue;

    // Getting the full water and sed volumes to add to the lake
    double water_volume,sediment_volume; std::vector<double> label_prop;
    // I also plan to get the nodes of this area
    std::vector<int> these_nodes;
    
    // This function checks if there are flats around it and process the whole lake as a single flat
    this->original_gathering_of_water_and_sed_from_pixel_or_flat_area(starting_node, water_volume, sediment_volume, label_prop, these_nodes);

    // Also create an empty lake here
    this->lakes.push_back(LakeLite(this->lake_incrementor));
    // and gives it its nodes
    this->lakes[lake_incrementor].nodes = these_nodes;
    // Registering my entry point with the water and sediment content in the queu helper
    this->queue_adder_for_lake.push_back(EntryPoint( water_volume,  sediment_volume,  starting_node, label_prop));
    // Emplacing the node in the queue
    iteralake.emplace(starting_node);

    // Labelling each node with their lake ID
    for(auto tnode : these_nodes)
      this->node_in_lake[tnode] = this->lake_incrementor;

    // Incrementing lake ID
    this->lake_incrementor++;
  }

  // Debugging tramp variable to ignore
  int n_neg = 0;
  double n_volwat_neg = 0;
  

  //############# Second step: add fluxes to the system while there are still some to add
  std::cout << "Starting the Dequeuing" << std::endl << std::endl << std::endl;

  // I am iterating while my queue of entry points is not emptied
  // Each time a new lake outflows into other lakes, it will add an entry point in the queue
  // This process is repeated untilall fluxes have reached their final state and escaped the system
    int has_outlet = 0;
    int no_has_outlet = 0;
  double negsumsed = 0, negsumwat = 0;
  while(iteralake.empty() == false)
  {


    // This is a FIFO queue, First in, first out
    // front gives me the next elemetn in line
    int entry_node = iteralake.front();
    // removing the thingy
    iteralake.pop();
    // Lol
    //        
    // POP!       
    //     * []
    //        *  *
    //   * '*' *'
    //      \*'/
    //       ||
    //      |* |
    //      |__|
    //      | *|
    //      |__|


    // Get the lake ID of the current node
    int current_lake = this->node_in_lake[entry_node];
    // # Checks if the lake has been drunk by another one
    if(current_lake >= 0 )
      current_lake = this->motherlake(current_lake);
    std::cout << std::endl << "LAKE " << current_lake << " at node " << entry_node << std::endl;


    // Getting the amount of sediment and water to add
    EntryPoint entry_point = this->queue_adder_for_lake[current_lake];
    // reinitialising the queue
    this->queue_adder_for_lake[current_lake] = EntryPoint(entry_node);

    // If my entry point is empty, it will happen if my lake has been processed by another entry point, I skip to the next node
    if(entry_point.volume_water == 0 && entry_point.volume_sed == 0)
    {
      continue;
    }

    //############# Third important task (even if still in step 2): 
    // if I have something to put in me lake, I add the content to it

    if( entry_point.volume_sed < 0)
    {
      this->lakes[current_lake].label_prop = mix_two_proportions(this->lakes[current_lake].volume_sed, 
            this->lakes[current_lake].label_prop,entry_point.volume_sed, entry_point.label_prop );

      this->lakes[current_lake].volume_sed += entry_point.volume_sed;

      double save_volume_sed = entry_point.volume_sed;
      
      entry_point.volume_sed = 0;
      
      // if(this->lakes[current_lake].volume_sed < 0)
        // throw std::runtime_error("CriticalLakeError:" + std::to_string(this->lakes[current_lake].volume_sed) + " is not a valid volume for lake sediments maybe " + std::to_string(save_volume_sed) + " is the problem");
    }


    double transfer = 0;
    if(entry_point.volume_water < 0)
    {
      transfer += entry_point.volume_water;
      entry_point.volume_water = 0;
    }

    if(entry_point.volume_water > 0 || entry_point.volume_sed > 0)
    {
      if(entry_point.volume_water < 0)
        negsumwat += entry_point.volume_water;
      if(entry_point.volume_sed < 0)
        negsumsed += entry_point.volume_sed;


      // Filling the lake
      current_lake = this->fill_mah_lake(entry_point, iteralake);
    }
    else
    {
      // DEBUG CHECEKRS
      negsumwat += entry_point.volume_water;
      negsumsed += entry_point.volume_sed;

      if(this->has_valid_outlet(current_lake))
        has_outlet++;
      else
        no_has_outlet++;

    }


    // At that point in the code I have an entry_point corrected for lakes:
    // It has (or not anymore) sediments and/or water to transmit to the outlet.
    // Anything that was supposed to go in the lake, is in the lake.

    // If my lake outlets, I need to reprocess the affected downstream nodes
    // this is by far the most complicated part of the code
    if(this->lakes[current_lake].outlet >= 0)
    {
      entry_point.volume_water += transfer;
      transfer = 0;
      std::cout << "Outlets at " << this->lakes[current_lake].outlet << " <--> new lake is " << current_lake << std::endl;
      this->reprocess_nodes_from_lake_outlet_v2(current_lake, this->lakes[current_lake].outlet, is_processed, iteralake, entry_point);
    }
    if(entry_point.volume_sed != 0)
      std::cout << "STILL SED HERE??? " << entry_point.volume_sed << std::endl;
    if(transfer != 0)
      std::cout << "LOST IN TRANSFER::" << transfer << std::endl;

    // skiping to the next entry node
  }
  // std::cout << "I had " << negsumsed << " Sediments and " << negsumwat/this->timestep << " Water uncared of with " << has_outlet << "/" << no_has_outlet << " with valid outlet" << std::endl;
  // std::cout << "N OUTLET ROMOB::" << this->n_outlets_remodelled << std::endl;

  if(no_has_outlet > 0)
    std::cout << " Caught one lake who could have lost water" << std::endl;

  // And I am done with the iterative solver!
  for (auto EP: this->lakes)
  {
    if(EP.is_now >= 0)
    {
      std::cout << "Lake " << EP.id << " has " << EP.volume_water << " Qw and " << EP.volume_sed << " Qs stored" << std::endl;
      if(EP.volume_sed < -1000)
        throw std::runtime_error("NegativeSedFinalLake");
    }

  }
}


void ModelRunner::reprocess_nodes_from_lake_outlet_v2(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, 
  EntryPoint& entry_point)
{

  // This function is a mastodon dealing with the reprocessing of nodes following a newly outletting lake
  // My current objective is to keep it (as) clear (as I can). Optimisation will come later. 
  // I am writing it as a big script to make it correct. Writing functions when I am sure I keep a feature, but so far I am still working on it a lot


  //----------------------------------------------------
  //----------------- INITIALISATION -------------------
  //----------------------------------------------------

  // std::cout << "OUTLET:" << std::endl;
  // this->chonk_network[this->lakes[current_lake].outlet].print_water_status();
  // std::cout << "Entry LAKE " << current_lake <<  " ::" << entry_point.volume_water / this->timestep << std::endl;


  // First, saivng some values for debugging and water balance purposes
  double debug_saverW = entry_point.volume_water / this->timestep;
  double outlet_water_saver = this->chonk_network[outlet].get_water_flux();



  // Local mass balance debugger
  sed_added_by_entry =  entry_point.volume_sed;
  sed_added_by_prod =  0;
  sed_already_outletted =  0;
  sed_added_by_donors = 0;
  sed_outletting_system = 0;

  // Debug tracker to check wether this specific iteration was adding water (needed it during the (very painful) mass balance checks)
  bool was_0 = false;
  if(entry_point.volume_water == 0)
    was_0 = true;

  // Initialising a priority queue sorting the nodes to reprocess by their id in the Mstack. the smallest Ids should come first
  // Because I am only processing nodes in between 2 discrete lakes, it should not be a problem
  std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> > ORDEEEEEER; // I am not politically aligned with John Bercow, this is jsut for the joke haha
  
  // This FIFO queue gathers nodes to reprocess to build up the local stack of node to work on
  std::queue<int> transec;
  
  // Keeping track of the delta sed/all contributing to potential lakes to add
  // These containers are important when reprocesseing nodes leads to a downstream depression
  std::vector<double> pre_sed(this->lakes.size(),0), pre_water(this->lakes.size(),0);
  std::vector<int> pre_entry_node(this->lakes.size(),-9999);
  std::vector<std::vector<double> > label_prop_of_pre(this->lakes.size(), std::vector<double>(this->n_labels,0.) );
  std::vector<double> delta_sed(this->lakes.size(),0), delta_water(this->lakes.size(),0);
  std::vector<std::vector<double> > label_prop_of_delta(this->lakes.size(), std::vector<double>(this->n_labels,0.) );

  

  //----------------------------------------------------
  //------- FORMING THE STACK OF NODE TO REPROC --------
  //----------------------------------------------------
  // This section aims to gather all nodes to reprocess. All node of the landscape have a type sort them by type: 
  // # 'n' if not concerned, 'l' if in lake, 'd' if donor, 'r' if potential reproc and 'y' if in stack
  std::vector<int> local_mstack;
  // keeping track of which nodes are already in teh queue (or more generally to avoid) and which one have receivers in the local stack
  std::vector<char> is_in_queue(this->io_int["n_elements"],'n');
  std::vector<char> has_recs_in_local_stack(this->io_int["n_elements"],'n');
  // Before starting to gather the nodes downstream of me outlet, let's gather mark the node in me lake as not available (as if already in the queue)
  // Avoinding nodes in the lakes
  for(auto tnode:this->lakes[current_lake].nodes)
  {
    is_in_queue[tnode] = 'l';
  }


  // This function gathers all the nodes to be reprocessed. Including their donor which will be partially reprocessed
  // This function is only geometrical, it does not assume any existing transfer of sed/water
  this->gather_nodes_to_reproc(local_mstack,  ORDEEEEEER,  is_in_queue,  outlet);

  // Final size OK
  // I have a stack of nodes to reprocess. 

  //----------------------------------------------------
  //------- STARTING THE OUTLET PREPROCESSING ----------
  //----------------------------------------------------



  // /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
      // // DEBUG VARIABLE TO CHECK IF WATER IS CREATED WHEN REPROCESSING A LAKE WITH 0 WATER
      // double local_sum = 0;
      // std::map<int,double> deltas;
      // std::vector<int> nodes;
      // std::vector<int> local_stack_checker = std::vector<int>(local_mstack);
      // local_stack_checker.push_back(outlet);
      // // this->label_nodes_with_no_rec_in_local_stack(local_stack_checker,is_in_queue, has_recs_in_local_stack);
      // for(auto node:local_stack_checker)
      // {
      //   if(is_in_queue[node] == 'd')
      //     continue;

      //   bool is_done = false;
        
        
      //   std::vector<int> recs = this->chonk_network[node].get_chonk_receivers_copy();
      //   std::vector<double> WW = this->chonk_network[node].get_chonk_water_weight_copy();
      //   double chonk_water = this->chonk_network[node].get_water_flux();
      //   int i = 0;
      //   for(auto rec : recs)
      //   { 
      //     if(is_in_queue[rec] == 'n' || is_in_queue[rec] == 'r' || is_in_queue[rec] == 'd')
      //     {
      //       double this_water = WW[i] * chonk_water ;
      //       local_sum -=  this_water;

      //       if(is_done == false)
      //       {
      //         deltas[node] = (-1 * this_water);
      //         nodes.push_back(node);
      //       }
      //       else
      //       {
      //         deltas[node] -= this_water;
      //       }
      //     }
      //     i++;
      //   }

      //   if(active_nodes[node] == 0)
      //   {
      //     deltas[node] -= chonk_water;

      //     local_sum -= chonk_water;
      //   }
        
      // }
      // // std::cout << "LOCAL SUM IS " << local_sum << std::endl;
      // // std::cout << outlet << "|||" << debug_saverW << "||||" << outlet_water_saver << "||||" << this->chonk_network[this->lakes[current_lake].outlet].get_water_flux() << std::endl;
  // end of DEBUG
  // /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

  // Getting the outletting chonk particule
  chonk tchonk = this->chonk_network[this->lakes[current_lake].outlet];
  
  // Now initialising the map correcting the fluxes
  // These corrector are used with nodes that have already been outlet and need reprocessing. These nodes are not reset and need to keep some corrections in mind
  std::map<int,double> WF_corrector; std::map<int,double> SF_corrector; std::map<int,std::vector<double> > SL_corrector;
  
  // Calling teh function preparing the outletting chonk processing
  // std::cout << "PRESEDOUT::" << this->chonk_network[this->lakes[current_lake].outlet].get_sediment_flux() << std::endl;
  this->chonk_network[this->lakes[current_lake].outlet] = chonk(this->preprocess_outletting_chonk(tchonk, entry_point, current_lake, 
    this->lakes[current_lake].outlet, WF_corrector,  SF_corrector,  SL_corrector, 
    pre_sed, pre_water, pre_entry_node, label_prop_of_pre));

  // std::cout << "POSTSEDOUT::" << this->chonk_network[this->lakes[current_lake].outlet].get_sediment_flux() << std::endl;

  // this->chonk_network[this->lakes[current_lake].outlet] = tchonk;
  //   _      _      _
  // >(.)__ <(.)__ =(.)__
  //  (___/  (___/  (___/  quack

  //----------------------------------------------------
  //---------- DEPROCESSING THE LOCAL STACK ------------
  //----------------------------------------------------
  
  // preprocessing the nodes on the path that are outlets
  this->check_what_give_to_existing_outlets(WF_corrector,  SF_corrector,  SL_corrector, local_mstack);
  // preprocessing the quantity given to existing lakes (to later calculate the delta)
  this->check_what_give_to_existing_lakes(local_mstack, outlet, current_lake, pre_sed,
    pre_water, pre_entry_node, label_prop_of_pre);
  
  double delta_sedsed = 0;
  std::cout << "nodes:";
  for(auto tnode:local_mstack)
  {
    if(current_lake == 18)
      std::cout << tnode << "|";
    if(is_in_queue[tnode] == 'd')
      delta_sedsed -= this->chonk_network[tnode].get_erosion_flux_only_bedrock() + this->chonk_network[tnode].get_erosion_flux_only_sediments() - this->chonk_network[tnode].get_deposition_flux();
  }
  std::cout << std::endl;

  // and finally deprocess the stack
  this->deprocess_local_stack(local_mstack,is_in_queue);
  // std::cout << "PREWATER 20::" << pre_water[20] << std::endl;
  // std::cout << "BITE::" << std::endl;
  // for(auto tnode:local_mstack)
  //   std::cout << "["<< is_in_queue[tnode] <<"]" << this->chonk_network[tnode].get_deposition_flux() << "|";

  //----------------------------------------------------
  //---------------- OUTLET PROCESSING -----------------
  //----------------------------------------------------
  // Process the outlet, whithout preparing the move (Already done) and readding the precipitation-like fluxes (already taken into account).
  // local_Qs_production_for_lakes[this->lakes[current_lake].outlet] = -1 * this->chonk_network[this->lakes[current_lake].outlet].get_sediment_flux();
  this->process_node_nolake_for_sure(this->lakes[current_lake].outlet, is_processed, this->active_nodes, 
      cellarea,topography, false, false);
  // local_Qs_production_for_lakes[this->lakes[current_lake].outlet] += this->chonk_network[this->lakes[current_lake].outlet].get_sediment_flux();
  this->sed_added_by_prod += this->chonk_network[this->lakes[current_lake].outlet].get_erosion_flux_only_bedrock()\
   * this->timestep * this->dx * this->dy;
  this->sed_added_by_prod += this->chonk_network[this->lakes[current_lake].outlet].get_erosion_flux_only_sediments()\
   * this->timestep * this->dx * this->dy;
  this->sed_added_by_prod -= this->chonk_network[this->lakes[current_lake].outlet].get_deposition_flux()\
   * this->timestep * this->dx * this->dy;


  std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
    this->chonk_network[this->lakes[current_lake].outlet].copy_moving_prep(rec,wwf, wws, strec);
    for(size_t u =0; u< rec.size(); ++u)
    {
      int ttnode = rec[u]; 
      if(this->has_been_outlet[ttnode] == 'y')
        std::cout << "@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!" << std::endl;
      if( (is_in_queue[ttnode] == 'y' || ttnode == this->lakes[current_lake].outlet )  && this->active_nodes[ttnode] && this->node_in_lake[ttnode] < 0 )
        continue;
      if(this->is_this_node_in_this_lake(ttnode, current_lake))
        continue;

      sed_outletting_system += wws[u] *  this->chonk_network[this->lakes[current_lake].outlet].get_sediment_flux();
    }
   



  // this->chonk_network[this->lakes[current_lake].outlet].print_status();

  //----------------------------------------------------
  //------------ LOCAL STACK REPROCESSING --------------
  //----------------------------------------------------
  // this section reprocess all nodes affected by the routletting of the lake nodes from upstream to donwstreamå
  this->reprocess_local_stack(local_mstack, is_in_queue, outlet, current_lake, WF_corrector, SF_corrector, SL_corrector);

  for(auto tnode:local_mstack)
  {
    if(is_in_queue[tnode] == 'd')
     delta_sedsed += this->chonk_network[tnode].get_erosion_flux_only_bedrock() + this->chonk_network[tnode].get_erosion_flux_only_sediments() - this->chonk_network[tnode].get_deposition_flux();;
  }

  // std::cout << "real delta is " << delta_sedsed * cellarea * this->timestep << std::endl;

  // DEBUG FOR WATER BALANCE
  // double sum_out = 0;
  // // std::cout << "Is considered:";
  // for(auto node:local_stack_checker)
  // {
  //   if(is_in_queue[node] == 'd')
  //     continue;
  //   bool is_done = false;
    
    
  //   std::vector<int> recs = this->chonk_network[node].get_chonk_receivers_copy();
  //   std::vector<double> WW = this->chonk_network[node].get_chonk_water_weight_copy();
  //   double chonk_water = this->chonk_network[node].get_water_flux();
  //   int i = 0;
  //   for(auto rec : recs)
  //   { 
  //     if(is_in_queue[rec] == 'n' || is_in_queue[rec] == 'r' || is_in_queue[rec] == 'd')
  //     {
  //       // std::cout << is_in_queue[rec] <<"(" << node << "->" << rec << ")" << "|";
  //       double this_water = WW[i] * chonk_water ;
  //       local_sum +=  this_water;
  //       deltas[node] += this_water;
        
  //     }
  //     i++;
  //   }

  //   if(active_nodes[node] == 0)
  //   {
  //     // std::cout << "z";
  //     deltas[node] += chonk_water;
  //     local_sum += chonk_water;
  //   }
    
  // }
  // // std::cout << std::endl;
  // for(auto v:deltas)
  // {
  //   // std::cout << v.first << "-->" << v.second << std::endl;
  //   if(active_nodes[v.first] == 0)
  //   {
  //     sum_out += v.second;
  //   }
  // }
  // // std::cout << "sum_out IS " << sum_out << " out of " << debug_saverW << std::endl;



  // if(double_equals(local_sum,debug_saverW,1) == false)
  // {
  //   std::cout << "WARNING:: " << debug_saverW << " got added to this local system but there is a delta of " << local_sum - debug_saverW << std::endl;
  //   // throw std::runtime_error("WaterDeltaWhileReprocError"); 
  // }


  //----------------------------------------------------
  //------------ PROCESSING ENTRY POINTS ---------------
  //----------------------------------------------------
  // I need now to calculate what my reprocessing nodes are giving to the different lakes, and calculate the delta
  // preprocessing the quantity given to existing lakes (to later calculate the delta)
  this->check_what_give_to_existing_lakes(local_mstack, outlet, current_lake, delta_sed,
    delta_water, pre_entry_node, label_prop_of_delta);
  std::vector<int> toutletstack = {outlet};
  this->check_what_give_to_existing_lakes(toutletstack, outlet, current_lake, delta_sed,
    delta_water, pre_entry_node, label_prop_of_delta);

  this->unpack_entry_points_from_delta_maps(iteralake, label_prop_of_delta, delta_sed,delta_water, pre_entry_node, 
     label_prop_of_pre, pre_sed,pre_water);

  // labelling the outlet
  this->has_been_outlet[outlet] = 'y';

  // Done
  //   _      _      _
  // >(.)__ <(.)__ =(.)__
  //  (___/  (___/  (___/  quack

    // DEBUGGING HARVERSTER PROCESS TO IGNORE
  debugint = xt::zeros<int>({this->io_int["n_elements"]});
  for(int i = 0; i < this->io_int["n_elements"]; i++)
  {
    int val = -1;

    if(is_in_queue[i] == 'd')
      val = 0;
    if(is_in_queue[i] == 'y')
      val = 1;
    if(has_been_outlet[i] == 'y')
      val = 2;

    debugint[i] = val;
  }



  // DEBUG LOCAL BALANCE
  std::cout << "Local balance: EP:" << sed_added_by_entry << " | prod:" << sed_added_by_prod \
 << " | sed_already_outletted: " << sed_already_outletted << " | don:" << sed_added_by_donors << " | away:" << sed_outletting_system << std::endl;

  std::cout << "Final balance = " << sed_added_by_entry + sed_added_by_prod + sed_already_outletted + sed_added_by_donors - sed_outletting_system << std::endl;

}

void ModelRunner::unpack_entry_points_from_delta_maps(std::queue<int>& iteralake, std::vector<std::vector<double> >& label_prop_of_delta,
std::vector<double>& delta_sed, std::vector<double>& delta_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre,
std::vector<double>& pre_sed, std::vector<double>& pre_water)
{
  double sum_dwats = 0;
  // Now I can go through my delta and pre vector to calculate and apply my new entry points
  for(size_t i=0; i < this->lakes.size(); i++ )
  {
    // Calculating potential delta for this lake
    double dwat = delta_water[i] - pre_water[i];
    double dsed = delta_sed[i] - pre_sed[i];

    // this lake has not been encountered if everything is 0
    if(double_equals(dwat,0,1e-3) && double_equals(dsed,0,1e-3))
      continue;

    sum_dwats += dwat;

    int target_lake = this->node_in_lake[pre_entry_node[i]];
    target_lake = motherlake(target_lake);
    EntryPoint other(dwat * this->timestep, dsed, pre_entry_node[i], label_prop_of_delta[i]);
    queue_adder_for_lake[target_lake].ingestNkill( other);
    
    std::cout << "ENTRY POINT LAKE " << target_lake << " IS NOW " << queue_adder_for_lake[target_lake].volume_sed << " ADDED " << dsed << " from " << pre_entry_node[i] << " was outlet befiore? " << this->has_been_outlet[pre_entry_node[i]] << std::endl;

    // Emplacing the next lake entry in the queue
    iteralake.emplace(pre_entry_node[i]);
  }
  // std::cout << "Sum Dwat IS::" << sum_dwats << std::endl;
}

void ModelRunner::label_nodes_with_no_rec_in_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, std::vector<char>& has_recs_in_local_stack)
{
  for (auto node:local_mstack)
    has_recs_in_local_stack[node] = 'p';
  for (auto node:local_mstack)
  {
    auto recs = this->chonk_network[node].get_chonk_receivers_copy();
    for(auto rec:recs)
    {
      if(has_recs_in_local_stack[rec] == 'p' || has_recs_in_local_stack[rec] == 'o')
        has_recs_in_local_stack[node] = 'o';
    }
    if(this->active_nodes[node] == false)
      has_recs_in_local_stack[node] = 'p';
  }
}

void ModelRunner::reprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, int outlet, int current_lake,
  std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >& SL_corrector)
{
  
  // double full_delta = 0;

  // Iterating through the local stack
  for(auto tnode:local_mstack)
  {
    // std::cout << tnode << "{" << is_in_queue[tnode] << "," << this->has_been_outlet[tnode] << "} was " << this->chonk_network[tnode].get_sediment_flux();
    // Discriminating between the donors and the receivers
    if(is_in_queue[tnode] == 'd')
    {
      // # I am a donor, I just need to regive the sed/water without other reproc
      // # Let me just just check which of my receivers I need to give fluxes (I shall not regive to the nodes not in my local stack)
      std::vector<int> ignore_some; 
      for(auto ttnode: this->chonk_network[tnode].get_chonk_receivers_copy())
      {
        // if the node is the outlet, or not a 'y', I ignore it
        if(is_in_queue[ttnode] != 'y' || ttnode == this->lakes[current_lake].outlet || (this->is_this_node_in_this_lake(ttnode, current_lake)))
        {          
          ignore_some.push_back(ttnode);
          continue;
        }
        // I also ignore the nodes in the current lake (they have a 'y' signature)
        else if (this->node_in_lake[ttnode] >= 0)
        {
          ignore_some.push_back(ttnode);
          continue;
        }
      }

      // # Ignore_some has the node i so not want
      // # So I transmit my fluxes to the nodes I do not ignore
      this->chonk_network[tnode].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);

      std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
      this->chonk_network[tnode].copy_moving_prep(rec,wwf, wws, strec);
      for(size_t u =0; u< rec.size(); ++u)
      {
        int ttnode = rec[u]; 
        if(ttnode == 1752)
          std::cout << "node d giving " << wws[u] *  this->chonk_network[tnode].get_sediment_flux() << " to 1752" << std::endl;

        if(std::find(ignore_some.begin(), ignore_some.end(), ttnode) != ignore_some.end())
          continue;
        if(this->is_this_node_in_this_lake(ttnode, current_lake))
          continue;

        if(this->active_nodes[ttnode])
          this->sed_added_by_donors += wws[u] *  this->chonk_network[tnode].get_sediment_flux();
        // else
        //   this->sed_added_by_donors -= wws[u] *  this->chonk_network[tnode].get_sediment_flux();

      }
      

    }
    else
    {
      // # I am not a nodor:
      if ( this->has_been_outlet[tnode] == 'y' )
      {
        std::cout << "*^*^*^*^*^*^*^*^*^*^*^**^*^*^*^*^*^*^*^*^*^*^**^*^*^*^*^*^*^*^*^*^*^*IT HAPPENS!!" << std::endl;
        this->n_outlets_remodelled ++;
        this->chonk_network[tnode].add_to_water_flux( WF_corrector[tnode]);
        this->chonk_network[tnode].add_to_sediment_flux( -1 * local_Qs_production_for_lakes[tnode], this->chonk_network[tnode].get_fluvialprop_sedflux());
        this->chonk_network[tnode].add_to_sediment_flux( SF_corrector[tnode], SL_corrector[tnode], this->chonk_network[tnode].get_fluvialprop_sedflux());
        std::cout << "After SF_corrector of " << SF_corrector[tnode] << " shit is " << this->chonk_network[tnode].get_sediment_flux() << std::endl;
        this->sed_added_by_donors -= SF_corrector[tnode];
      }
      // # So I need full reproc yaaay
      // full_delta -= this->chonk_network[tnode].get_sediment_flux();
      this->process_node_nolake_for_sure(tnode, is_processed, this->active_nodes, 
        cellarea,topography, true, true);
      this->chonk_network[tnode].check_sums();
      // full_delta += this->chonk_network[tnode].get_sediment_flux();
      // local_Qs_production_for_lakes[tnode] = full_delta;
      this->sed_added_by_prod += this->chonk_network[tnode].get_erosion_flux_only_bedrock()\
   * this->timestep * this->dx * this->dy;
  this->sed_added_by_prod += this->chonk_network[tnode].get_erosion_flux_only_sediments()\
   * this->timestep * this->dx * this->dy;
  this->sed_added_by_prod -= this->chonk_network[tnode].get_deposition_flux()\
   * this->timestep * this->dx * this->dy;

      std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
      this->chonk_network[tnode].copy_moving_prep(rec,wwf, wws, strec);
      for(size_t u =0; u< rec.size(); ++u)
      {
        int ttnode = rec[u]; 
        if(ttnode == 1752)
          std::cout << "node y giving " << wws[u] *  this->chonk_network[tnode].get_sediment_flux() << " to 1752" << std::endl;
        if((is_in_queue[ttnode] == 'y' || ttnode == this->lakes[current_lake].outlet) && this->active_nodes[ttnode])
          continue;
        if(this->is_this_node_in_this_lake(ttnode, current_lake))
          continue;

        sed_outletting_system += wws[u] *  this->chonk_network[tnode].get_sediment_flux();
      }
   

    }

    // std::cout << " Now is " << this->chonk_network[tnode].get_sediment_flux() << std::endl;;
    // this->chonk_network[tnode].print_status();

  }

    // std::cout << " Full delta is " << full_delta << std::endl;;

}

void ModelRunner::deprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue)
{
  // Now deprocessing the receivers in potential lake while saving their contribution to lakes in order to calculate the delta
  for(auto tnode : local_mstack)
  {
    // # If my node is just a donor, I am not resetting it
    if (is_in_queue[tnode] != 'y')
      continue;

    // # Cancelling the fluxes before moving prep (i.e. the precipitation, infiltrations, ...)
    // # This is not for water purposes as it gets reproc anyway, but for mass balance calculation
    this->cancel_fluxes_before_moving_prep(this->chonk_network[tnode], tnode);
    // # resetting the node (All fluxes and modifyers to 0)
    if ( this->has_been_outlet[tnode] != 'y' )
    {
      this->chonk_network[tnode].reset();
      this->chonk_network[tnode].set_other_attribute_array("label_tracker", std::vector<double>(this->n_labels,0));
    }
    else
    {
      std::cout << "This remob outlet is " << this->chonk_network[tnode].get_sediment_flux() << std::endl;
      this->sed_added_by_donors = this->chonk_network[tnode].get_sediment_flux();
      std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
      this->chonk_network[tnode].copy_moving_prep(rec,wwf, wws, strec);
      for ( auto trec:rec)
      {
        if(this->has_been_outlet[trec] == 'y')
          std::cout << "Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!Double outlet!" << std::endl;
      }
      this->chonk_network[tnode].reinitialise_moving_prep();
    }
  }
}

void ModelRunner::check_what_give_to_existing_outlets(std::map<int,double>& WF_corrector,  std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >&  SL_corrector, std::vector<int>& local_mstack)
{
  // going through the nodes of the local stack
  for( auto node : local_mstack)
  {
    chonk& tchonk = this->chonk_network[node];

    //getting the weights
    // # Initialising a bunch of intermediate containers and variable
    std::vector<int> tchonk_recs;
    std::vector<double> tchonk_slope_recs;
    std::vector<double> tchonk_weight_water_recs, tchonk_weight_sed_recs;

    // copying the weights from the current 
    tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);

    // checking all neighboiyrs
    for(size_t i =0; i < tchonk_recs.size(); i++)
    {
      // node indice of the receiver
      int tnode = tchonk_recs[i];


      // Now checking if the rec is an outlet:
      if(this->has_been_outlet[tnode] == 'y')
      {
        if(WF_corrector.count(tnode) == 0)
        {
          WF_corrector[tnode] = 0;
          SF_corrector[tnode] = 0;
          SL_corrector[tnode] = {};
        }
        WF_corrector[tnode] -= tchonk_weight_water_recs[i] * tchonk.get_water_flux();
        SL_corrector[tnode] = mix_two_proportions(SF_corrector[tnode],SL_corrector[tnode], -1 * tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux(), tchonk.get_other_attribute_array("label_tracker"));
        SF_corrector[tnode] -= tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux();
        std::cout << "CORRECTOR ON " << tnode << " IS " << SF_corrector[tnode] << std::endl;
        
      }
    }
  }
}

void ModelRunner::check_what_give_to_existing_lakes(std::vector<int>& local_mstack, int outlet, int current_lake, std::vector<double>& this_sed,
   std::vector<double>& this_water, std::vector<int>& this_entry_node, std::vector<std::vector<double> >& label_prop_of_this)
{
  for( auto node : local_mstack)
  {
    chonk& tchonk = this->chonk_network[node];

    //getting the weights
    // # Initialising a bunch of intermediate containers and variable
    std::vector<int> tchonk_recs;
    std::vector<double> tchonk_slope_recs;
    std::vector<double> tchonk_weight_water_recs, tchonk_weight_sed_recs;

    // copying the weights from the current 
    tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);

    for(size_t i =0; i < tchonk_recs.size(); i++)
    {
      // node indice of the receiver
      int tnode = tchonk_recs[i];
      // And checking if the rec is a lake
      int lakid = this->node_in_lake[tnode];
      if(lakid >= 0)
      {
        lakid = this->motherlake(lakid);
        if(lakid != current_lake)
        {
          // Old debug statement
          // std::cout << "Adding " << tchonk_weight_water_recs[i] * tchonk.get_water_flux() << " to " << lakid << " from " << node << "->" << tnode << " Breakdown: " << tchonk_weight_water_recs[i] << " * " <<  tchonk.get_water_flux() << std::endl; ;
          label_prop_of_this[lakid] = mix_two_proportions(this_sed[lakid],label_prop_of_this[lakid], tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux(), tchonk.get_other_attribute_array("label_tracker"));
          this_sed[lakid] += tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux();
          this_water[lakid] += tchonk_weight_water_recs[i] * tchonk.get_water_flux();
          this_entry_node[lakid] = tnode;
        }
      }
    }
  }
}


chonk ModelRunner::preprocess_outletting_chonk(chonk tchonk, EntryPoint& entry_point, int current_lake, int outlet,
 std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, std::map<int,std::vector<double> >& SL_corrector,
 std::vector<double>& pre_sed, std::vector<double>& pre_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre)
{
  // Getting the additioned water rate
  // std::cout << "I WATER RATE IS " << tchonk.get_water_flux() << std::endl;
  double water_rate = entry_point.volume_water / this->timestep;
  // std::cout << "II WATER RATE IS " << water_rate << std::endl;
  // Summing it to the previous one
  water_rate += tchonk.get_water_flux();
  // std::cout << "III WATER RATE IS " << water_rate << std::endl;

  // Dealing with sediments
  std::vector<double> label_prop = entry_point.label_prop;//mix_two_proportions(entry_point.volume_sed,entry_point.label_prop, tchonk.get_sediment_flux(), tchonk.get_other_attribute_array("label_tracker"));
  
  double sedrate = entry_point.volume_sed + tchonk.get_sediment_flux() - local_Qs_production_for_lakes[outlet];
  std::cout << "Outlet = " << entry_point.volume_sed << " + " <<  \
  tchonk.get_sediment_flux() << " - " << local_Qs_production_for_lakes[outlet] << " = " << sedrate << std::endl;
  entry_point.volume_sed = 0;

  this->sed_already_outletted += tchonk.get_sediment_flux() - local_Qs_production_for_lakes[outlet];

  //getting the weights
  // # Initialising a bunch of intermediate containers and variable
  std::vector<int> ID_recs, tchonk_recs;
  std::vector<double> slope_recs,tchonk_slope_recs;
  std::vector<double> weight_water_recs, weight_sed_recs, tchonk_weight_water_recs, tchonk_weight_sed_recs;
  double sumW = 0;
  double sumS = 0;
  int nrecs = 0;
  // copying the weights from the current 
  tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);
  
  // Iterating through the receivers of the outlet and removing the water given to the current lake by this outlet prior reprocessing
  // this water is already in the lake outletting thingy!
  for(size_t i = 0; i < tchonk_recs.size(); i++)
  {
    // Node
    int tnode = tchonk_recs[i];
    // lake?
    int tlak = this->node_in_lake[tnode];
    if(tlak >= 0)
      tlak = this->motherlake(tlak);
    else
      continue;

    // Is current lake?
    if(tlak == current_lake)
    {
      // yes, removing sed and water rate
      water_rate -= tchonk_weight_water_recs[i] * tchonk.get_water_flux();
      // label_prop = mix_two_proportions(sedrate,  label_prop, -1 * tchonk_weight_water_recs[i] * tchonk.get_sediment_flux(),  tchonk.get_other_attribute_array("label_tracker"));
      // sedrate -= tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux();
      // std::cout << "Subtracting II " << tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux() << std::endl;; 
    }
  }

  // Now iterating thorugh the neighbours
  std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(outlet, this->active_nodes, neightbors, dummy);

  // calculating the slope too 
  double SS = -9999;
  int SS_ID = -9999;
  for(size_t i = 0; i< neightbors.size(); i++)
  {
    // node indice of the receiver
    int tnode = neightbors[i];

    // J is the indice of this specific node in the tchonk referential
    int j = -1;
    auto itj = std::find(tchonk_recs.begin(), tchonk_recs.end(), tnode);
    if(itj != tchonk_recs.end() )
      j = std::distance(tchonk_recs.begin(), itj);
    // Note that if j is still < 0, tnode is not in the chonk

    if(topography[tnode] >= topography[outlet])
    {
      // double tsedfromdon = this->chonk_network[tnode].sed_flux_given_to_node(outlet);
      // label_prop = mix_two_proportions(tsedfromdon, this->chonk_network[tnode].get_other_attribute_array("label_tracker"), sedrate, label_prop);
      // sedrate += tsedfromdon;
      continue;
    }

    // Checking wether it is giving to the original lake or not
    if(this->is_this_node_in_this_lake(tnode, current_lake) ==  false)
    {
      // get the node
      ID_recs.push_back(tnode);
      // calculate the slope
      double tS = topography[outlet] - topography[tnode];
      tS = tS / dummy[i];
      // If j exists push the weights
      if(j >= 0)
      {
        weight_water_recs.push_back(tchonk_weight_water_recs[j]);
        weight_sed_recs.push_back(tchonk_weight_sed_recs[j]);
        sumW += tchonk_weight_water_recs[j];
        sumS += tchonk_weight_sed_recs[j];
      }
      else
      {
        // else 0
        weight_water_recs.push_back(0);
        weight_sed_recs.push_back(0);
      }
      
      // Slope      
      slope_recs.push_back(tS);
      nrecs++;

    }
    // else
    // {
    //   if(j >= 0)
    //   {
    //     this->Qs_mass_balance -= tchonk_weight_sed_recs[j] * tchonk.get_sediment_flux();
    //   }
    // }

    // Now checking if the rec is an outlet and pushing the correctors:
    if(this->has_been_outlet[tnode] == 'y')
    {
      if(WF_corrector.count(tnode) == 0)
      {
        WF_corrector[tnode] = 0;
        SF_corrector[tnode] = 0;
        SL_corrector[tnode] = {};
      }

      if(j>=0)
      {
        WF_corrector[tnode] -= tchonk_weight_water_recs[j] * tchonk.get_water_flux();
        SL_corrector[tnode] = mix_two_proportions(SF_corrector[tnode],SL_corrector[tnode], -1 * tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux(), tchonk.get_other_attribute_array("label_tracker"));
        SF_corrector[tnode] -= tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux();
      }
    }

    // And finally checking if the rec is a lake
    int lakid = this->node_in_lake[tnode];
    if(lakid >= 0 && j >= 0)
    {
      lakid = this->motherlake(lakid);
      if(lakid != current_lake)
      {
        label_prop_of_pre[lakid] = mix_two_proportions(pre_sed[lakid],label_prop_of_pre[lakid], tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux(), tchonk.get_other_attribute_array("label_tracker"));
        pre_sed[lakid] += tchonk_weight_sed_recs[j] * tchonk.get_sediment_flux() ;
        pre_water[lakid] += tchonk_weight_water_recs[j] * tchonk.get_water_flux();
        pre_entry_node[lakid] = tnode;
      }
    }
  }

  // Normalising the weights to their new states
  if(sumW > 0)
  {
    for(size_t i =0; i < weight_water_recs.size(); i++)
    {
      weight_water_recs[i] = weight_water_recs[i]/sumW;
    }
  }
  else
  {
    weight_water_recs = std::vector<double>(weight_water_recs.size(), 1./int(weight_water_recs.size()));
  }

  // if(sumS > 0)
  // {
  //   for(size_t i =0; i < weight_sed_recs.size(); i++)
  //   {
  //     weight_sed_recs[i] = weight_sed_recs[i]/sumS;
  //   }

  // }
  // else
  // {
  //   weight_sed_recs = std::vector<double>(weight_sed_recs.size(), 1./int(weight_sed_recs.size()));

  // }
  weight_sed_recs = std::vector<double>(ID_recs.size(),0);

  std::cout << "outlet was " << tchonk.get_sediment_flux() << " and is now " << sedrate << std::endl;;
  if(sedrate < 0)
  {
    std::cout << "Warning::Sedrate recasted to 0" << std::endl;;
    sedrate = 0;
  }


  // Resetting the CHONK
  tchonk.reset();
  tchonk.external_moving_prep(ID_recs,weight_water_recs,weight_sed_recs,slope_recs);
  tchonk.set_water_flux(water_rate);
  tchonk.set_sediment_flux(sedrate,label_prop, 1.);
  tchonk.I_solemnly_swear_all_my_sediments_are_fluvial();

  // for(auto idf:ID_recs)
  //   std::cout << idf << "||";
  // std::cout << std::endl;
 // Old debug statement
  // std::cout << "Set to outlet::" << water_rate << " wat and sed: " << sedrate << std::endl; 
  // Ready to go ??!!
  return tchonk;
}

void ModelRunner::gather_nodes_to_reproc(std::vector<int>& local_mstack, 
  std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> >& ORDEEEEEER,
   std::vector<char>& is_in_queue, int outlet)
{
  // Initiating the transec with the outlet, but not putting it the 
  std::queue<int> transec;
  transec.emplace(outlet);
  is_in_queue[outlet] = 'y';

  // this is the loop gathering downstream nodes
  while(transec.empty() == false)
  {
    // Getting the next node in the FIFO structure (first = outlet)
    int next_node = transec.front();
    // Getting rid of it
    transec.pop();

    // If this is not the outlet, which will be treated separatedly I add it in the PQ to reproc
    if(next_node != outlet )
    {
      ORDEEEEEER.emplace(node_to_reproc(next_node,this->graph.get_index_MF_stack_at_i(next_node)));
      is_in_queue[next_node] = 'y';
    }

    // Otherwise going through neightbors
    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node, this->active_nodes, neightbors, dummy);

    // checking the state of the neightbor
    for (auto tnode : neightbors)
    {
      // If is already processed, in queue, or in the original lake, ingore it
      if(is_in_queue[tnode] == 'y' || is_in_queue[tnode] == 'l')
        continue;
      
      // if it is below my current node
      if(topography[tnode] < topography[next_node] )
      {
        // If already in an existing lake, saving the node as lake potential
        if(this->node_in_lake[tnode] >= 0)
        {
          is_in_queue[tnode] = 'r';
          continue;
        }
        else
        {
          //Else, this is a node to reproc, well done
          transec.emplace(tnode);
          is_in_queue[tnode] = 'y';
          continue;
        } 
      }
      else if (topography[tnode] > topography[next_node])
      {
        if(is_in_queue[tnode] == 'n')// && this->node_in_lake[tnode] < 0)
        {
          is_in_queue[tnode] = 'd';
          continue;
        }
      }
    }
  }

  // Adding the donors to the PQ, not before because they couldvebeen added as rec later on
  for(int i =0; i<this->io_int["n_elements"]; i++)
  {
    if (is_in_queue[i] == 'd')
    {
      ORDEEEEEER.emplace(node_to_reproc(i,this->graph.get_index_MF_stack_at_i(i)));
    }

  }

  // Formatting the iteratorer, whatever the name this thing is. not iterator. is something else. I guess. I think. why would you care anyway as I'd be surprised anyone ever reads this code.
  local_mstack = std::vector<int>(); local_mstack.reserve(ORDEEEEEER.size());
  // emptying the queue and reinitialising the chonks
  while(ORDEEEEEER.empty() == false)
  {
    // Getting the firsrt node
    node_to_reproc n2r = ORDEEEEEER.top();
    int tnode = n2r.node;
    // POP!
    ORDEEEEEER.pop();
    // Geeting in teh local stack  
    local_mstack.emplace_back(tnode);

  }
}

bool ModelRunner::has_valid_outlet(int lakeid)
{
  int outlet = this->lakes[lakeid].outlet;
  if(outlet<0)
    return false;

  std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(outlet, this->active_nodes, neightbors, dummy);
  for(auto node:neightbors)
  {
    if(this->topography[node] < this->topography[outlet])
    {
      int tlid = this->node_in_lake[node];
      if(tlid >= 0) tlid = this->motherlake(tlid);
      if(tlid != lakeid) return true;
    }
  }
  return false;

}


void ModelRunner::reprocess_nodes_from_lake_outlet(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, EntryPoint& entry_point)
{
  std::cout << "ONGOING DEPRECATION" << std::endl;
}



void ModelRunner::check_what_gives_to_lake(int entry_node, std::vector<int>& these_lakid , std::vector<double>& twat, std::vector<double>& tsed,
 std::vector<std::vector<double> >& tlab, std::vector<int>& these_ET, int lake_to_ignore)
{
  these_lakid = std::vector<int>();
  twat = std::vector<double>();
  tsed = std::vector<double>();
  tlab = std::vector<std::vector<double> >();
  these_ET = std::vector<int>();
  // int idx_rec = -1;
  auto WWC = this->chonk_network[entry_node].get_chonk_water_weight_copy();
  auto WWS = this->chonk_network[entry_node].get_chonk_sediment_weight_copy();
  std::vector<int> recs = this->chonk_network[entry_node].get_chonk_receivers_copy();

  for(int idx_rec = 0; idx_rec < int(recs.size()); idx_rec++)
  {

    int rec = recs[idx_rec];
    int tapo = this->node_in_lake[rec];
    if(tapo < 0)
      continue;

    int this_lakid = this->motherlake(tapo);
    if(this_lakid == lake_to_ignore)
      continue;

    const bool is_not_here = std::find(these_lakid.begin(), these_lakid.end(), this_lakid) == these_lakid.end();

    int index;
    // std::cout<< "GAFAAAAAAAAT::" << entry_node << " gave " << this->chonk_network[entry_node].get_water_flux() * WWC[idx_rec] << " to laek " << this_lakid << std::endl;
    
    if(is_not_here)
    {
    
      index = int(these_lakid.size());
      these_lakid.push_back(this_lakid);
      twat.push_back(this->chonk_network[entry_node].get_water_flux() * WWC[idx_rec]);
      tsed.push_back(this->chonk_network[entry_node].get_sediment_flux() * WWS[idx_rec]);
      tlab.push_back(this->chonk_network[entry_node].get_other_attribute_array("label_tracker"));
      these_ET.push_back(rec);

    }
    else
    {
      auto it = std::find(these_lakid.begin(), these_lakid.end(), this_lakid);
      index = std::distance(these_lakid.begin(), it);

      tlab[index] = mix_two_proportions(tsed[index], tlab[index], this->chonk_network[entry_node].get_sediment_flux() * WWS[idx_rec],this->chonk_network[entry_node].get_other_attribute_array("label_tracker"));
      twat[index] += this->chonk_network[entry_node].get_water_flux() * WWC[idx_rec];
      tsed[index] += this->chonk_network[entry_node].get_sediment_flux() * WWS[idx_rec];
      these_ET[index] = recs[idx_rec];
    }
  }


}

int ModelRunner::fill_mah_lake(EntryPoint& entry_point, std::queue<int>& iteralake)
{
  // priotrity queue to fill the lake. This container stores elevation nodes and sort them on the go by
  // elevation value. I am filling one node at a time and gathering all its neighbors in this queue
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;

  // Aliases and shortcups
  double cellarea = this->dx * this->dy;

  // Starting by adding the first node of the list
  depressionfiller.emplace(nodium(entry_point.node, topography[entry_point.node]));

  // local balance checker
  double save_entry_water = entry_point.volume_water;

  // Creating a new lake: the iterative strategy makes new lakes on the top of existing ones
  // And storing its ID
  int current_lake = this->lake_incrementor;
  this->lakes.push_back(LakeLite(this->lake_incrementor));

  // DEBUG STATEMENT
  std::cout << "Filling lake " << current_lake << " with sed " << entry_point.volume_sed << std::endl;

  // Creating an empty entry-point for the lake
  this->queue_adder_for_lake.push_back(EntryPoint(entry_point.node));

  // Finally getting the incrementor ready for the next lake
  this->lake_incrementor++;

  // The water elevation is the current node at start (-> no water at all)
  this->lakes[current_lake].water_elevation = topography[entry_point.node];

  // I also set the outlet to a values I can recognise as NO OUTLET
  int outlet = -9999;

  // is_in queue is an helper char array to check wether a node is already in the queue or not
  // 'y' -> yes, 'n' -> no. Not using boolean for memory otpimisation. See std::vector<bool> on google for why
  std::vector<char> is_in_queue(this->io_int["n_elements"],'n');
  // My first node, is in that queue
  is_in_queue[entry_point.node] = 'y';
  // Similar helper here but for which node is in this lake
  std::vector<char> is_in_lake(this->io_int["n_elements"],'n');

  // Very specific checekr in the rare case my entry point is an inactive internal node
  if(this->active_nodes[entry_point.node] == false)
  {
    outlet = entry_point.node;
  }

  // Starting the main loop
  while(entry_point.volume_water > 0 && outlet < 0)
  {
    // first get the next node in line. If first iteration, the entry node, else, the closest elevation
    nodium next_node = depressionfiller.top();
    // then pop the next node
    depressionfiller.pop();
    // go through neighbours manually and either feed the queue or detect an outlet
    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node.node, this->active_nodes, neightbors, dummy);

    // For each of the neighbouring node: checking their status
    for(auto tnode:neightbors)
    {
      // if already in the queue: I pass
      if(is_in_queue[tnode] == 'y')
        continue;

      // If flat or higher: I keep
      if(topography[tnode] >= topography[next_node.node])
      {
        depressionfiller.emplace(nodium(tnode, topography[tnode]));
        is_in_queue[tnode] = 'y';
      }
      else
      {
        // if a single neighbour is lower (and not in the queue!) then the current node is the outlet
        outlet = next_node.node;
        break;
      }
    }

    // calculating the xy surface of the lake
    double area_component_of_volume = int(this->lakes[current_lake].nodes.size()) * this->dx * this->dy;
    // Calculating the maximum volume to add until next node
    double dV = (next_node.elevation - this->lakes[current_lake].water_elevation) * area_component_of_volume;

    // Case 1: Can fill the max dV without using all water available
    if(dV <= entry_point.volume_water)
    {
      // removing the volume added to the lake
      entry_point.volume_water -= dV;
      // Updating the water elevation of the lake to the right one
      this->lakes[current_lake].water_elevation = next_node.elevation;
      // Updating the Volume of water in the lake (principle of communicating vessels)
      this->lakes[current_lake].volume_water += dV;

      // If there is an outlet, then I do not add the lake to the node
      // Otherwise...
      if(outlet < 0)
      {
        this->lakes[current_lake].nodes.push_back(next_node.node);
        is_in_lake[next_node.node] = 'y';
      }
      else
      {
        for(auto tnode : this->lakes[current_lake].nodes)
          topography[tnode] = this->lakes[current_lake].water_elevation;
        // get all flat nodes around that one but the outlet
        std::vector<int> fnodes = this->graph.get_all_flat_from_node(next_node.node, topography, this->active_nodes);
        for(auto tnode : fnodes)
        {
          if(is_in_lake[tnode] == 'n' && tnode != outlet)
          {
            this->lakes[current_lake].nodes.push_back(tnode);
            is_in_lake[tnode] = 'y';
          }
        }
      }
    }
    else
    {
      dV = entry_point.volume_water;
      entry_point.volume_water  = 0;
      this->lakes[current_lake].water_elevation += dV/(area_component_of_volume);
      this->lakes[current_lake].volume_water += dV;
      // Wether there is an outlet or not, I cancel it as it will nto have enough water to reach it
      outlet = -9999;
    }

    // going to the next part of the loop or stopping
  }

  std::cout << "Lake volume predrink is " << this->lakes[current_lake].volume_water << " and sedEP is " << entry_point.volume_sed << std::endl;
    // std::cout << "BITE0" << std::endl;

  // if there is an outlet 
  if(outlet >= 0)
    this->lakes[current_lake].outlet = outlet;
    // std::cout << "BITE1" << std::endl;

  // Now merging with lakes below adn updating the topography, and backcalculating the erosion/deposition fluxes fluxes
  double sedrate_modifuer = 0;
  std::vector<int> lakes_ingested;

  // std::cout << "BITE2" << std::endl;
  // std::cout << this->lakes[current_lake].nodes.size() << std::endl;;
  // std::cout << " BITE2.5" << std::endl;

  for(size_t talap = 0; talap< this->lakes[current_lake].nodes.size(); talap ++)
  {
    // std::cout << "BITE3" << std::endl;
    int tnode = this->lakes[current_lake].nodes[talap];
    // std::cout << tnode << std::endl;
    // updating topography
    topography[tnode] = this->lakes[current_lake].water_elevation;
    // checking if the node belongs to a lake, in which case I ingest it
    if(this->node_in_lake[tnode] >= 0)
    {
      // the underlying node belongs to a lake
      // Checking both motherlake IDs
      int tested = this->motherlake(this->node_in_lake[tnode]);
      int against = this->lakes[current_lake].id;
      // If they are different I drink
      if( tested != against)
      {
        std::cout << "Drinking lake " << this->motherlake(this->node_in_lake[tnode]) << " it has " \
        << this->lakes[this->motherlake(this->node_in_lake[tnode])].volume_sed << " | " << entry_point.volume_sed << "|" << \
        this->lakes[current_lake].volume_sed << " ---> ";

        // The drinking merges the two entities: it puts all the water together and give all the sediments
        // to the entry points
        this->drink_lake(this->lakes[current_lake].id, this->motherlake(this->node_in_lake[tnode]), \
          entry_point, iteralake);
        // Saving which lakes have been ingested
        lakes_ingested.push_back(this->motherlake(this->node_in_lake[tnode]));

        std::cout << entry_point.volume_sed << "|" << this->lakes[current_lake].volume_sed << std::endl;
      }
    }
    // Registering the node as belonging to this new lake
    this->node_in_lake[tnode] = current_lake;

    // I also need to cancel erosion/deposition that could have been done in this lake
    // And correct the sediment flux
    // Removing what was eroded before
    sedrate_modifuer -= this->chonk_network[tnode].get_erosion_flux_only_bedrock() * this->timestep * cellarea;
    sedrate_modifuer -= this->chonk_network[tnode].get_erosion_flux_only_sediments() * this->timestep * cellarea;
    // Re-add what was deposited before
    sedrate_modifuer += this->chonk_network[tnode].get_deposition_flux() * this->timestep * cellarea;
    // Static fluxes to 0 goes brrrr
    this->chonk_network[tnode].reinitialise_static_fluxes();
  }
    // std::cout << "BITE4" << std::endl;

  // adding all the nodes of the ingested lakes
  if(lakes_ingested.size()>0)
  {
    std::vector<char> is_in_dat_lake(this->io_int["n_elements"], 'n');
    for(auto tn: this->lakes[current_lake].nodes)
      is_in_dat_lake[tn] = 'y';
    for( auto tl:lakes_ingested)
    {
      for(auto tn:this->lakes[tl].nodes)
      {
        if(is_in_dat_lake[tn] == 'n')
        {
          is_in_dat_lake[tn] = 'y';
          this->lakes[current_lake].nodes.push_back(tn);
        }
      }
    }
  }
    // std::cout << "BITE5::" << outlet << std::endl;


  // Done with lake node management

  // Now, what if I have an outlet??
  if(outlet>=0)
  {
    //First: Cancelling (from the lake) the sediment flux of the outlet that were given to the lake
    std::vector<int> rec; std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec;
    // Getting all the neighborsa and the weights given to each neighbors
    this->chonk_network[outlet].copy_moving_prep( rec, wwf, wws,  strec);
    double sed_flux_corrector = 0;
      // std::cout << "BITE5.1" << std::endl;
    for ( size_t u=0; u<rec.size(); u++)
    {
      int ulak = this->node_in_lake[rec[u]];
      // std::cout << ulak << std::endl;
      if(ulak >= 0)
      {
        ulak = this->motherlake(ulak);
        // If the lake of the outlet receiver is ... the current lake
        if(ulak == current_lake)
        {
          // I remove the sediment fluxes fromt the entry point
          double tsed = wws[u] * this->chonk_network[outlet].get_sediment_flux();
          sedrate_modifuer -= tsed;
          sed_flux_corrector -= tsed;
        }
      }
    }

      // std::cout << "BITE6" << std::endl;


    this->chonk_network[outlet].reinitialise_static_fluxes();
    // NOT REMOVING IT FROM THAT NODE YET, IT WOULD BREAK THE THINGY, 
    // will do it properly in the outlet preprocessing function
    // this->chonk_network[outlet].add_to_sediment_flux(sed_flux_corrector);
  }
  
  // Applying the sediment flux deducer
  entry_point.volume_sed += sedrate_modifuer;

  std::cout << "Lake volume post drink is " << this->lakes[current_lake].volume_water << " and sedEP is " << entry_point.volume_sed << std::endl;

  // Finally adding the sediments to the lake

  //# Calculating the remaining space available in that lake
  double lake_capacity = this->lakes[current_lake].volume_water - this->lakes[current_lake].volume_sed;
  // Case 1: I have enough volume to ingest the whole sedload
  if(lake_capacity >= entry_point.volume_sed)
  {
    // I calcul the proportions
    this->lakes[current_lake].label_prop = mix_two_proportions(entry_point.volume_sed, entry_point.label_prop,
        this->lakes[current_lake].volume_sed, this->lakes[current_lake].label_prop);
    // And correct the volumes
    this->lakes[current_lake].volume_sed += entry_point.volume_sed;
    // removing all sediments 
    entry_point.volume_sed = 0;
  }
  else
  {
    // CASE 2: I am filling up the lake and transmitting sediments to the outlet
    this->lakes[current_lake].label_prop = mix_two_proportions(lake_capacity, entry_point.label_prop,
        this->lakes[current_lake].volume_sed, this->lakes[current_lake].label_prop);
    this->lakes[current_lake].volume_sed += lake_capacity;
    entry_point.volume_sed -= lake_capacity;
  }

  std::cout << entry_point.volume_sed << " will be transmitted to an outlet while " << this->lakes[current_lake].volume_sed << " is stored in the lake" << std::endl;

  // Transmission to the outlet has been moved to another function. The remaining amount of sediment is known by the entry_point

  return current_lake;

}

/// function eating a lake from another
void ModelRunner::drink_lake(int id_eater, int id_edible, EntryPoint& entry_point, std::queue<int>& iteralake)
{ 
  if(this->lakes[id_edible].is_now >= 0)
    throw std::runtime_error("OVERDRUNK");

  // Updating the id of the new lake
  this->lakes[id_edible].is_now = id_eater;

  // merging water volumes
  this->lakes[id_eater].volume_water += this->lakes[id_edible].volume_water;

  // if the current lake has an outlet:
  if(this->lakes[id_eater].outlet >= 0)
  {
    // Removing its entry point
    entry_point.ingestNkill(this->queue_adder_for_lake[id_edible]);
  }
  else
  {
    // Readding an entry point to complete the fill of this lake
    this->queue_adder_for_lake[id_eater].ingestNkill(this->queue_adder_for_lake[id_edible]);
    iteralake.emplace(entry_point.node);
  }

  // Cancelling all entry points ingested. (clean the queue of entry points to a dead lake)
  this->queue_adder_for_lake[id_edible] = EntryPoint(this->queue_adder_for_lake[id_edible].node);

  // Now if they have the same water elevation, there is something to do
  if (this->lakes[id_eater].water_elevation == this->lakes[id_edible].water_elevation)
  {
    // std::cout << "LAKE TRANSMIT ITS SUMOUTRATE :: " << this->lakes[id_edible].sum_outrate << " OUTLET IS " <<  this->lakes[id_eater].outlet << " VS " << this->lakes[id_edible].outlet<< std::endl;
    // if the outlets are different: I check wheter it is a valid outlet and integrates it
    if(this->lakes[id_eater].outlet != this->lakes[id_edible].outlet && this->lakes[id_edible].outlet > 0)
    {      
      int n_DS_n = 0;
      int n_DS_o = 0;
      std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(this->lakes[id_edible].outlet, this->active_nodes, neightbors, dummy);
      for(auto tnode:neightbors)
      {
        if(this->topography[tnode] < this->topography[this->lakes[id_edible].outlet])
        {
          int ttttlake = this->node_in_lake[tnode];
          if(ttttlake >= 0)
            ttttlake = this->motherlake(ttttlake);

          if(ttttlake != id_edible && ttttlake != id_eater)
            n_DS_n ++;
        }
      }

      // if(this->node_in_lake[this->lakes[id_edible].outlet] < 0 && n_DS_n > 0)
      if(n_DS_n > 0)
      {  
        // throw std::runtime_error("Happens 671222");
        this->lakes[id_eater].outlet = this->lakes[id_edible].outlet;
        this->lakes[id_eater].sum_outrate += this->lakes[id_edible].sum_outrate;
      }

    }
    else
    {   
    // DEPRECATED   
      this->lakes[id_eater].sum_outrate += this->lakes[id_edible].sum_outrate;
    }


  }


  // Merging sediment volumes
  entry_point.label_prop = mix_two_proportions(entry_point.volume_sed,entry_point.label_prop,
    this->lakes[id_edible].volume_sed,this->lakes[id_edible].label_prop);
  entry_point.volume_sed += this->lakes[id_edible].volume_sed;
  this->lakes[id_edible].volume_sed = 0;
}


// Function checking the active lake id of a node (ie, the top-level one)
int ModelRunner::motherlake(int this_lake_id)
{
  int output = this_lake_id;
  while(this->lakes[this_lake_id].is_now >= 0)
  {
    this_lake_id = this->lakes[this_lake_id].is_now;
    output = this->lakes[this_lake_id].id;
  };
  return output;
}

void ModelRunner::original_gathering_of_water_and_sed_from_pixel_or_flat_area(int starting_node, double& water_volume, double& sediment_volume, 
  std::vector<double>& label_prop, std::vector<int>& these_nodes)
{
  water_volume =  this->chonk_network[starting_node].get_water_flux() * this->timestep;
  sediment_volume = this->chonk_network[starting_node].get_sediment_flux();
  label_prop = this->chonk_network[starting_node].get_other_attribute_array("label_tracker");
  std::cout << "Originnal entry point is " << sediment_volume << std::endl;
  
  // return;

  std::queue<int> FIFO;
  std::vector<char> is_in_queue(this->io_int["n_elements"], 'n');
  is_in_queue[starting_node] = 'y';
  these_nodes.push_back(starting_node);
  FIFO.push(starting_node);
  
  while(FIFO.empty() == false)
  {
    int next_node = FIFO.front();
    FIFO.pop();

    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node, this->active_nodes, neightbors, dummy);
    double telev = topography[next_node];

    for(auto tnode:neightbors)
    {
      if(this->active_nodes[tnode] == false)
        continue;
      if(is_in_queue[tnode] == 'y')
        continue;

      if(this->topography[tnode] == telev)
      {
        FIFO.push(tnode);
        is_in_queue[tnode] = 'y';

        if(this->lake_status[tnode] == 0)
        {
          auto this_label_prop = this->chonk_network[tnode].get_other_attribute_array("label_tracker");
          label_prop = mix_two_proportions(sediment_volume, label_prop, this->chonk_network[tnode].get_sediment_flux(), this_label_prop);
          water_volume +=  this->chonk_network[tnode].get_water_flux()  * this->timestep;
          sediment_volume += this->chonk_network[tnode].get_sediment_flux();
          this->lake_status[tnode] = 1;
          these_nodes.push_back(tnode);
        
        }

      }
    }
  }  
  std::cout << "AfterFlatGathering entry point is " << sediment_volume << std::endl;
  std::cout << " Nodes are:";
  for(auto nn : these_nodes)
    std::cout << nn << "|";
  std::cout << std::endl;


}

void ModelRunner::process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep)
{
    // Just a check: if the lake solver is not activated, I have no reason to reprocess node
    if(this->lake_solver == false && is_processed[node])
      return;

    // if I reach this stage, this node can be labelled as processed
    is_processed[node] = true;

    // manages the fluxes before moving the particule: accumulating DA, precipitation, infiltration, evaporation, ...
    this->manage_fluxes_before_moving_prep(this->chonk_network[node],this->label_array[node] );


    // Uf the lake solving is activated, then I go through the (more than I expected) complex process of filing a lake correctly
    if(this->lake_solver) 
    {

      // 1) checking if my node has a lake ID  
      int lakeid = this->node_in_lake[node];

      // if it has no lake id (ie is not a lake) and is not the bottom of an active depression then I skip that part 
      if(lakeid == -1 && this->graph.is_depression(node) == false)
      {
        goto nolake;
        return;
      }
      
    }
    else
    {
      // If the lake solver is not activated, I am transferring the fluxes to the receiver according to Cordonnier et al. planar graph (see node graph)  
      if(this->active_nodes[node] && this->graph.is_depression(node))
      {
        // Getting the so-called node
        int next_node = this->graph.get_Srec(node);
        // Not sure this is required
        if(this->graph.is_depression(next_node) == false)
          next_node = this->graph.get_Srec(next_node);   
        // Formatting the depression moving pattern     
        this->chonk_network[node].reinitialise_moving_prep();
        this->chonk_network[node].external_moving_prep({next_node},{1.},{1.},{0});
        if(next_node == node)
          throw std::runtime_error("DUPLICATES FOUND HERE #2");
        // Applying the fluxes modifyers
        this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
        // Splitting the fluxes
        this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->surface_elevation_tp1, 
          this->sed_height_tp1, this->timestep);
        is_processed[node] = true;
        // Done        
      }
      else
      {
        // this is a normal node and I go to the nolake management routiness
        goto nolake;
      }

      return;
    }
    // LABEL
    nolake:
    
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
    {
      if(this->chonk_network[node].get_chonk_receivers_copy().size() > 0)
      {
        this->chonk_network[node].reinitialise_moving_prep();
      }

      this->manage_move_prep(this->chonk_network[node]);
    }


    // Fluxes after moving prep are active fluxes such as erosion or other thingies
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    // Apply the changes and propagate the fluxes downstream
    CHRONO_start[4] = std::chrono::high_resolution_clock::now();
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->surface_elevation_tp1, this->sed_height_tp1, this->timestep);
    CHRONO_stop[4] = std::chrono::high_resolution_clock::now();
    CHRONO_n_time[4] ++;
    CHRONO_mean[4] += std::chrono::duration<double>(CHRONO_stop[4] - CHRONO_start[4]).count();
  }

void ModelRunner::process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move)
{
  local_Qs_production_for_lakes[node] = -1 * this->chonk_network[node].get_sediment_flux() ;

  is_processed[node] = true;
  if(need_flux_before_move)
  {
    this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
  }
  
  // first step is to apply the right move method, to prepare the chonk to move
  if(need_move_prep)
  {
    this->manage_move_prep(this->chonk_network[node]);
  }
  

  // this->chonk_network[node].print_status();
  this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
  
  // if(this->chonk_network[node].get_chonk_receivers().size() == 0 && active_nodes[node] > 0)
  //   throw std::runtime_error("NoRecError::internal flux broken");
  
  this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->surface_elevation_tp1, this->sed_height_tp1, this->timestep);
  local_Qs_production_for_lakes[node] +=  this->chonk_network[node].get_sediment_flux() ;
}

void ModelRunner::process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move, std::vector<int>& ignore_some)
{
  local_Qs_production_for_lakes[node] = -1 * this->chonk_network[node].get_sediment_flux() ;

    is_processed[node] = true;
    if(need_flux_before_move)
      this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);
    
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    
    this->chonk_network[node].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);
    local_Qs_production_for_lakes[node] +=  this->chonk_network[node].get_sediment_flux() ;
}

void ModelRunner::increment_new_lake(int& lakeid)
{
  lakeid = lake_incrementor;
  lake_incrementor++;
}

void ModelRunner::finalise()
{
  // Finilising the timestep by applying the changes to the thingy
  // First gathering all the aliases 
  // xt::pytensor<double,1>& sed_height_tp1 = this->this->sed_height_tp1;
  // xt::pytensor<double,1>& sed_height = this->sed_height;

  double cellarea = this->dx * this->dy;

  // First dealing with lake deposition:
  this->drape_deposition_flux_to_chonks();

  // then actively finalising the deposition and other details
  // Iterating through all nodes
  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    if(std::isfinite(surface_elevation_tp1[i]) == false)
    {
      throw std::runtime_error("NAN IN ELEV WHILE FINIlIISAFJ");
    }
    double sedcrea = this->chonk_network[i].get_sediment_creation_flux() * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_bedrock() * cellarea * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_sediments() * cellarea * timestep;
    this->Qs_mass_balance += this->chonk_network[i].get_deposition_flux() * cellarea * timestep;


    if(this->active_nodes[i] == false)
      continue;

    // Getting the current chonk
    chonk& tchonk = this->chonk_network[i];
    // getting the current composition of the sediment flux
    auto this_lab = tchonk.get_other_attribute_array("label_tracker");

    // NANINF DEBUG CHECKER
    for(auto LAB:this_lab)
      if(std::isfinite(LAB) == false)
        std::cout << LAB << " << naninf for sedflux" << std::endl;
 

    // First applying the bedrock-only erosion flux: decrease the overal surface elevation without affecting the sediment layer
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux_only_bedrock() * timestep;

    // Applying elevation changes from the sediments
    // Reminder: sediment creation flux is the absolute rate of removal/creation of sediments

    // NANINF DEBUG CHECKER
    if(std::isfinite(sedcrea) == false)
    {
      std::cout << sedcrea << "||" << this->node_in_lake[i] << "||" << \
      this->lakes[this->node_in_lake[i]].volume_sed/this->lakes[this->node_in_lake[i]].volume_water \
      << "||" <<this->lakes[this->node_in_lake[i]].volume_sed << "||" << this->lakes[this->node_in_lake[i]].volume_water << std::endl;
      throw std::runtime_error("NAN sedcrea finalisation not possible yo");
    }


    // std::cout << "0.2" << std::endl;
    // TEMP DEBUGGER TOO
    // AT TERM THIS SHOULD NOT HAPPEN???
    // if I end up with a negative sediment layer
    if(sedcrea + sed_height_tp1[i] < 0)
    {
      // IT STILL HAPPENS
      // std::cout << "happens??" << sedcrea << "||" << sed_height_tp1[i] << "||" << this->node_in_lake[i] << std::endl;
      surface_elevation_tp1[i] -= sed_height_tp1[i];
      sed_height_tp1[i] = 0.;
      sed_prop_by_label[i] = std::vector<std::vector<double> >();
      is_there_sed_here[i] = false;
    } 
    else
    {
      // std::cout << "0.3" << std::endl;
      // Calling the function managing the sediment layer composition tracking
      this->add_to_sediment_tracking(i, sedcrea, this_lab, sed_height_tp1[i]);
      // std::cout << "0.4" << std::endl;
      // Applying the delta_h on both surface elevation and sediment layer
      surface_elevation_tp1[i] += sedcrea;
      sed_height_tp1[i] += sedcrea;
      // if(std::ceil(sed_height_tp1[i]/this->io_double["depths_res_sed_proportions"]) != int(sed_prop_by_label[i].size()))
      // {
      //   // std::cout << "SEDREGISTERING PROBLEM called with " << i << "|" << sedcrea << "|" << sed_height_tp1[i] << std::endl;
      //   // std::cout << std::ceil(sed_height_tp1[i]/this->io_double["depths_res_sed_proportions"]) << "||" << int(sed_prop_by_label[i].size()) << std::endl;
      //   // throw std::runtime_error("COCKROACH");
      // }
    }


    //Dealing now with "undifferentiated" Erosion rates
    double tadd = tchonk.get_erosion_flux_undifferentiated() * timestep;

    if(std::abs(tadd)>0)
    {
      // What was that again??
      // if(sed_height_tp1[i] < 0)
      //   tadd = tadd + sed_height_tp1[i];
      // std::cout << "0.5" << std::endl;
      this->add_to_sediment_tracking(i, -1*tadd, this_lab, sed_height_tp1[i]);
      // std::cout << "0.6" << std::endl;
      surface_elevation_tp1[i] -= tadd;
      sed_height_tp1[i] -= tadd;
    }

    // LAST_DEBUG_CHECK
    if(sed_height_tp1[i]<0)
    {
      // to_remove = std::abs(sed_height_tp1[i]);
      sed_height_tp1[i] = 0;
      sed_prop_by_label[i] = std::vector<std::vector<double> >();
      is_there_sed_here[i] = false;
      // surface_elevation_tp1[i] += to_remove;
    }
    else
    {
      if(sed_height_tp1[i]>0 && sed_prop_by_label[i].size() == 0)
        throw std::runtime_error("It should not happens!!!!!!!!!");

    }


  }

  auto tlake_depth = this->topography - this->surface_elevation;
  // Calculating the water balance thingies
  double save_Ql_out = this->Ql_out;
  this->Ql_out = 0;
  for(int i=0; i<this->io_int["n_elements"]; i++)
  {
    this->Ql_out += (tlake_depth[i] - this->io_double_array["lake_depth"][i]) * this->dx * this->dy / this->timestep;
  }

  
  // Saving the new lake depth  
  this->io_double_array["lake_depth"] = tlake_depth;


  // calculating other water mass balance.
  // xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  for(int i =0; i<this->io_int["n_elements"]; i++)
  {
    if(this->active_nodes[i] == false)
    {
      this->Qw_out += this->chonk_network[i].get_water_flux();
      this->Qs_mass_balance += this->chonk_network[i].get_sediment_flux();
    }
    // double delta_sed = (sed_height_tp1[i] - sed_height[i]);
    // double delta_elev = surface_elevation_tp1[i] - surface_elevation[i];
    // double delta_delta = delta_elev - delta_sed;
    // this->Qs_mass_balance += (delta_elev) * cellarea * this->timestep;
    // this->Qs_mass_balance -= (delta_delta) * cellarea * this->timestep;
  }
}


// this Fucntion add a height of sediment to a preexisting pile. It updates the tracking of deposited sediments
void ModelRunner::add_to_sediment_tracking(int index, double height, std::vector<double> label_prop, double sed_depth_here)
{
  // No height to add -> nothing happens yo
  if(height == 0)
    return;

  double depth_res = this->io_double["depths_res_sed_proportions"];

  // First, let's calculates stats about the current box situation
  double boxes_there = sed_depth_here/depth_res;
  double delta_boxes = std::abs(height/depth_res);

  // Breakdown into filled and underfilled
  double boxes_there_filled;
  double box_there_ufill = std::modf(boxes_there, &boxes_there_filled);

  // Breakdown of incoming fluxes:
  double boxes_ta_filled;
  double box_ta_ufill = std::modf(delta_boxes, &boxes_ta_filled);
  
  // Index of current box
  // std::cout << "A::" << boxes_there << std::endl;;
  int current_box = int(sed_prop_by_label[index].size()) - 1;
  int current_box_test = int(std::ceil(boxes_there)) - 1;
  // if(current_box != current_box_test)
  // {
    // std::cout << "Beef::" << current_box << "||" << current_box_test << "||" << boxes_there << "||" << sed_depth_here << std::endl;
    // throw std::runtime_error("C**K!!!");
  // }
  // std::cout << "B::"  << current_box<< std::endl;;


  // If I am removing sediments
  if(height < 0)
  {
    // std::cout << "C" << std::endl;;

    // Removing the full boxes I can
    for(int i = 0; i<int(boxes_ta_filled); i++)
    {
      sed_prop_by_label[index].pop_back();
      current_box--;
    }
    // Getting the height of the remaining boxe
    double this_hbox = box_there_ufill - box_ta_ufill;
    // std::cout << "D::" << this_hbox << "::" << box_there_ufill << "::" << box_ta_ufill << std::endl;;

    // I will remove prop2 from prop1
    double prop1 = box_there_ufill;
    double prop2 = box_ta_ufill;

    // But if the prop is negative, I need to remove a last box
    if(this_hbox < 0)
    {
      // std::cout << "E" << std::endl;;
      // Popping it
      sed_prop_by_label[index].pop_back();
      current_box--;
      // I am now removing from a full box
      prop1 = 1;
      // prop2 = -1 * (1 + this_hbox);
      prop2 = this_hbox;
      // std::cout << "F" << std::endl;;
    }
    // finally mixing the two depositions
    // std::cout << "G" << std::endl;;
    this->sed_prop_by_label[index][current_box] = mix_two_proportions(prop1,sed_prop_by_label[index][current_box], prop2, label_prop);
  }
  else
  {
    // I am adding sediment, first filling the current box

    // std::cout << "H" << std::endl;;
    double this_hbox = box_there_ufill + box_ta_ufill;
    // std::cout << box_there_ufill << "opo" << box_ta_ufill << std::endl;
    double prop1 = box_there_ufill;
    double prop2 = box_ta_ufill;

    if(sed_prop_by_label[index].size() == 0)
    {
      sed_prop_by_label[index].push_back(std::vector<double>(label_prop.size(), 0.) );
      current_box = 0;
    }

    if(this_hbox > 1)
    {
      // std::cout << "I" << std::endl;;
      this->sed_prop_by_label[index][current_box] = mix_two_proportions(prop1,sed_prop_by_label[index][current_box], 1 - prop1, label_prop);

      for(int i = 0; i < int(boxes_ta_filled) + 1; i++)
      {
        sed_prop_by_label[index].push_back(label_prop);
        current_box++;
      }
      // std::cout << "J" << std::endl;;
    }
    else if(boxes_ta_filled > 0)
    {
      // std::cout << "K" << std::endl;;
      this->sed_prop_by_label[index][current_box] = mix_two_proportions(prop1,sed_prop_by_label[index][current_box], (1 - prop1), label_prop);

      for(int i = 0; i < int(boxes_ta_filled); i++)
      {
        sed_prop_by_label[index].push_back(label_prop);
        current_box++;
      }
      // std::cout << "L" << std::endl;;
    }
    else
    {
      // std::cout << "M" << std::endl;;
      this->sed_prop_by_label[index][current_box] = mix_two_proportions(prop1,sed_prop_by_label[index][current_box], prop2, label_prop);
      // std::cout << "N" << std::endl;;
    }
    
  }

  // std::cout << "AFFAF::" << current_box << "||" << current_box_test << "||" << boxes_there << "||" << sed_depth_here << std::endl;


  // for(auto gabro:this->sed_prop_by_label[index][current_box])
  //   std::cout << "GABRO::" <<gabro << std::endl;

}

xt::pytensor<float,4> ModelRunner::get_sed_prop_by_label_matrice(int n_depths)
{
  xt::pytensor<float,4> output = xt::zeros<float>({this->io_int["ny"], this->io_int["nx"], n_depths, this->n_labels});
  int nx = this->io_int["nx"];
  // Watching Alex and Mikael's talk while writing that code
  for ( auto alex_attal : this->sed_prop_by_label)
  {
  // Watching Alex and Mikael's talk while writing that code

    int node = alex_attal.first;
    int row,col;
    col = int(node % nx);
    row = std::floor(node/nx);

    for(int i=0; i< int(alex_attal.second.size());i++)
      for(int j=0; j<n_labels; j++)
      {
        output(row,col,int(alex_attal.second.size()) - i - 1,j) = alex_attal.second[i][j];
      }
  }
  return output;

}


xt::pytensor<double,2> ModelRunner::get_superficial_layer_sediment_prop()
{
  xt::pytensor<double,2> output = xt::zeros<double>({n_labels,this->io_int["n_elements"]});

  for (int lab = 0; lab < this->n_labels; lab++)
  {
    for(size_t i = 0; i < this->io_int["n_elements"]; i++)
    {
      if(sed_prop_by_label[int(i)].size() != 0)
      {
        output(lab,i) = sed_prop_by_label[int(i)][sed_prop_by_label[int(i)].size() - 1][lab];
      }
      else
        output(lab,i) = 0;
    }
  }
  return output;
}



xt::pytensor<int,1> ModelRunner::get_debugint()
{
  return debugint;
}




void ModelRunner::manage_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{
  CHRONO_start[2] = std::chrono::high_resolution_clock::now();
  

  this_chonk.inplace_only_drainage_area(this->dx, this->dy);
        this->Qw_in += this->dx* this->dy;

  CHRONO_stop[2] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[2] ++;
  CHRONO_mean[2] += std::chrono::duration<double>(CHRONO_stop[2] - CHRONO_start[2]).count();
  return;
  // for(auto method:this->ordered_flux_methods)
  // {
  //   if(method == "move")
  //     break;
  //   int this_case = intcorrespondance[method];

  //   switch(this_case)
  //   {
  //     case 5:
  //       this_chonk.inplace_only_drainage_area(this->dx, this->dy);
  //       this->Qw_in += this->dx* this->dy;
  //       break;
  //     case 6:
  //       this_chonk.inplace_precipitation_discharge(this->dx, this->dy,this->io_double_array["precipitation"]);
  //       this->Qw_in += this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->dx* this->dy;
  //       break;
  //     case 7:
  //       this_chonk.inplace_infiltration(this->dx, this->dy, this->io_double_array["infiltration"]);
  //       this->Qw_out += this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->dx* this->dy;
  //       break;
  //   }

  // }
}

void ModelRunner::cancel_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{
  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 5:
        this_chonk.cancel_inplace_only_drainage_area(this->dx, this->dy);
        this->Qw_in -= this->dx* this->dy;
        break;
      case 6:
        this_chonk.cancel_inplace_precipitation_discharge(this->dx, this->dy,this->io_double_array["precipitation"]);
        this->Qw_in -= this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->dx* this->dy;
        break;
      case 7:
        this_chonk.cancel_inplace_infiltration(this->dx, this->dy, this->io_double_array["infiltration"]);
        this->Qw_out -= this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->dx* this->dy;
        break;
    }

  }
}



void ModelRunner::manage_move_prep(chonk& this_chonk)
{

  CHRONO_start[5] = std::chrono::high_resolution_clock::now();


  int this_case = intcorrespondance[this->move_method];

  std::vector<int> rec = this_chonk.get_chonk_receivers_copy();
  switch(this_case)
  {
    case 2:
      this_chonk.move_to_steepest_descent(this->graph, this->timestep,  this->topography, this->dx, this->dy, chonk_network);
      break;
    case 3:
      this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, 
   this->topography, this->dx, this->dy, chonk_network);
      break;
    case 4:
      this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep,  this->topography, 
        this->dx, this->dy, chonk_network);
      break;
      
    default:
      std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  }

  CHRONO_stop[5] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[5] ++;
  CHRONO_mean[5] += std::chrono::duration<double>(CHRONO_stop[5] - CHRONO_start[5]).count();
}

void ModelRunner::manage_fluxes_after_moving_prep(chonk& this_chonk, int label_id)
{
  CHRONO_start[3] = std::chrono::high_resolution_clock::now();

  int index = this_chonk.get_current_location();
  std::vector<double> these_sed_props(this->n_labels,0.);
  if(is_there_sed_here[index] && this->sed_prop_by_label[index].size()>0) // I SHOULD NOT NEED THE SECOND THING, WHY DO I FUTURE BORIS????
    these_sed_props = this->sed_prop_by_label[index][this->sed_prop_by_label[index].size() - 1];

  if(this->CHARLIE_I)
  {

    double this_Kr = this->labelz_list[label_id].Kr_modifyer * this->labelz_list[label_id].base_K;
    double this_Ks = this->labelz_list[label_id].Ks_modifyer * this->labelz_list[label_id].base_K;

    this_chonk.charlie_I(this->labelz_list[label_id].n, this->labelz_list[label_id].m, this_Kr, this_Ks,
    this->labelz_list[label_id].dimless_roughness, this->sed_height[index], 
    this->labelz_list[label_id].V, this->labelz_list[label_id].dstar, this->labelz_list[label_id].threshold_incision, 
    this->labelz_list[label_id].threshold_entrainment,label_id, these_sed_props, this->timestep,  this->dx, this->dy);
  }

  // Hillslope routine
  if(this->CIDRE_HS)
  {

    double this_kappas = this->labelz_list[label_id].kappa_s_mod * this->labelz_list[label_id].kappa_base;
    double this_kappar = this->labelz_list[label_id].kappa_r_mod * this->labelz_list[label_id].kappa_base;
    // std::cout << "kappe_r is " << this_kappar << " and kappa_s is " << this_kappas << " Sc = " << this->labelz_list[label_id].critical_slope << std::endl;

    this_chonk.CidreHillslopes(this->sed_height[index], this_kappas, 
            this_kappar, this->labelz_list[label_id].critical_slope,
    label_id, these_sed_props, this->timestep, this->dx, this->dy, true, this->graph, 1e-6);
  }

  CHRONO_stop[3] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[3] ++;
  CHRONO_mean[3] += std::chrono::duration<double>(CHRONO_stop[3] - CHRONO_start[3]).count();

  return;

  // bool has_moved = false;
  // for(auto method:this->ordered_flux_methods)
  // {
  //   int index = this_chonk.get_current_location();
  //   std::vector<double> these_sed_props(this->n_labels,0.);
  //   if(is_there_sed_here[index] && this->sed_prop_by_label[index].size()>0) // I SHOULD NOT NEED THE SECOND THING, WHY DO I FUTURE BORIS????
  //     these_sed_props = this->sed_prop_by_label[index][this->sed_prop_by_label[index].size() - 1];
    
  //   if(method == "move")
  //   {
  //      has_moved = true;
  //      continue;
  //   }

  //   if(has_moved == false)
  //     continue;

  //   int this_case = intcorrespondance[method];
  //   switch(this_case)
  //   {
  //     case 1:
  //     // Classic SPL /!\ probs deprecatedx
  //       this_chonk.active_simple_SPL(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["SPIL_K"][label_id], this->timestep, this->dx, this->dy, label_id);
  //       break;
  //     case 8:
  //     // The SPACE model aka CHARLIE_I
  //       this_chonk.charlie_I(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["CHARLIE_I_Kr"][label_id], 
  // this->labelz_list_double["CHARLIE_I_Ks"][label_id],
  // this->labelz_list_double["CHARLIE_I_dimless_roughness"][label_id], this->sed_height[index], 
  // this->labelz_list_double["CHARLIE_I_V"][label_id], 
  // this->labelz_list_double["CHARLIE_I_dstar"][label_id], this->labelz_list_double["CHARLIE_I_threshold_incision"][label_id], 
  // this->labelz_list_double["CHARLIE_I_threshold_entrainment"][label_id],
  // label_id, these_sed_props, this->timestep,  this->dx, this->dy);
        
  //       break;
  //     case 11:
  //     // The SPACE model aka CHARLIE_I
  //       this_chonk.charlie_I_K_fQs(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["CHARLIE_I_Kr"][label_id], 
  // this->labelz_list_double["CHARLIE_I_Ks"][label_id],
  // this->labelz_list_double["CHARLIE_I_dimless_roughness"][label_id], this->sed_height[index], 
  // this->labelz_list_double["CHARLIE_I_V"][label_id], 
  // this->labelz_list_double["CHARLIE_I_dstar"][label_id], this->labelz_list_double["CHARLIE_I_threshold_incision"][label_id], 
  // this->labelz_list_double["CHARLIE_I_threshold_entrainment"][label_id],
  // label_id, these_sed_props, this->timestep,  this->dx, this->dy, this->labelz_list_double["CHARLIE_I_Krmodifyer"]);
  //       break;

  //     case 9:
  //     // Cidre hillslope method, only on the sediment layer
  //       this_chonk.CidreHillslopes(this->sed_height[index], this->labelz_list_double["Cidre_HS_kappa_s"][label_id], 
  //         0., this->labelz_list_double["Cidre_HS_critical_slope"][label_id],
  // label_id, these_sed_props, this->timestep, this->dx, this->dy, false, this->graph, 1e-6);
  //       break;
  //     case 10:
  //     // Cidre hillslope method, both sed and bedrock
  //       this_chonk.CidreHillslopes(this->sed_height[index], this->labelz_list_double["Cidre_HS_kappa_s"][label_id], 
  //           this->labelz_list_double["Cidre_HS_kappa_r"][label_id], this->labelz_list_double["Cidre_HS_critical_slope"][label_id],
  //   label_id, these_sed_props, this->timestep, this->dx, this->dy, true, this->graph, 1e-6);
  //         break;

  //     default:
  //       break;
  //   }
  // }
  // return;
}

// Initialise ad-hoc set of internal correspondance between process names and integer
// Again this is a small otpimisation that reduce the need to initialise and call maps for each nodes as maps are slower to access
void ModelRunner::initialise_intcorrespondance()
{
  intcorrespondance = std::map<std::string,int>();
  intcorrespondance["SPIL_Howard_Kerby_1984"] = 1;
  intcorrespondance["D8"] = 2;
  intcorrespondance["MF_fastscapelib"] = 3;
  intcorrespondance["MF_fastscapelib_threshold_SF"] = 4;
  intcorrespondance["drainage_area"] = 5;
  intcorrespondance["precipitation_discharge"] = 6;
  intcorrespondance["infiltration_discharge"] = 7;
  intcorrespondance["CHARLIE_I"] = 8;
  intcorrespondance["Cidre_hillslope_diffusion_no_bedrock"] = 9;
  intcorrespondance["Cidre_hillslope_diffusion"] = 10;
  intcorrespondance["CHARLIE_I_KfQs"] = 11;

}

void ModelRunner::prepare_label_to_list_for_processes()
{
  for(auto method:this->ordered_flux_methods)
  {
    int this_case = intcorrespondance[method];
    switch(this_case)
    {
      // SPIL_full_label
      case 1:
        labelz_list_double["SPIL_m"] = std::vector<double>();
        labelz_list_double["SPIL_n"] = std::vector<double>();
        labelz_list_double["SPIL_K"] = std::vector<double>();
        for(auto& tlab:labelz_list)
        {
          labelz_list_double["SPIL_m"].push_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].push_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["SPIL_K"].push_back(tlab.double_attributes["SPIL_K"]);
        }
        break;
      // Charlie the first
      case 8:
        labelz_list_double["SPIL_m"] = std::vector<double>();
        labelz_list_double["SPIL_n"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Kr"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Ks"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_V"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dimless_roughness"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dstar"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_incision"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_entrainment"] = std::vector<double>();
        for(auto& tlab:labelz_list)
        {
          labelz_list_double["SPIL_m"].push_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].push_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["CHARLIE_I_Kr"].push_back(tlab.double_attributes["CHARLIE_I_Kr"]);
          labelz_list_double["CHARLIE_I_Ks"].push_back(tlab.double_attributes["CHARLIE_I_Ks"]);
          labelz_list_double["CHARLIE_I_V"].push_back(tlab.double_attributes["CHARLIE_I_V"]);
          labelz_list_double["CHARLIE_I_dimless_roughness"].push_back(tlab.double_attributes["CHARLIE_I_dimless_roughness"]);
          labelz_list_double["CHARLIE_I_dstar"].push_back(tlab.double_attributes["CHARLIE_I_dstar"]);
          labelz_list_double["CHARLIE_I_threshold_incision"].push_back(tlab.double_attributes["CHARLIE_I_threshold_incision"]);
          labelz_list_double["CHARLIE_I_threshold_entrainment"].push_back(tlab.double_attributes["CHARLIE_I_threshold_entrainment"]);
        }
        break;

      case 9 :
        labelz_list_double["Cidre_HS_kappa_s"] = std::vector<double>();
        labelz_list_double["Cidre_HS_critical_slope"] = std::vector<double>();
        for (auto& tlab:labelz_list)
        {
          labelz_list_double["Cidre_HS_kappa_s"].push_back(tlab.double_attributes["Cidre_HS_kappa_s"]);
          labelz_list_double["Cidre_HS_critical_slope"].push_back(tlab.double_attributes["Cidre_HS_critical_slope"]); 
        }       
        break;
      case 10 :
        labelz_list_double["Cidre_HS_kappa_s"] = std::vector<double>();
        labelz_list_double["Cidre_HS_kappa_r"] = std::vector<double>();
        labelz_list_double["Cidre_HS_critical_slope"] = std::vector<double>();
        for (auto& tlab:labelz_list)
        {
          labelz_list_double["Cidre_HS_kappa_s"].push_back(tlab.double_attributes["Cidre_HS_kappa_s"]);
          labelz_list_double["Cidre_HS_kappa_r"].push_back(tlab.double_attributes["Cidre_HS_kappa_r"]);
          labelz_list_double["Cidre_HS_critical_slope"].push_back(tlab.double_attributes["Cidre_HS_critical_slope"]);
        }
        break;

      // Charlie the first, with dynamic sedimentation
      case 11:
        labelz_list_double["SPIL_m"] = std::vector<double>();
        labelz_list_double["SPIL_n"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Kr"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Ks"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_V"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dimless_roughness"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_dstar"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_incision"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_threshold_entrainment"] = std::vector<double>();
        labelz_list_double["CHARLIE_I_Krmodifyer"] = std::vector<double>();
        for(auto& tlab:labelz_list)
        {
          labelz_list_double["SPIL_m"].push_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].push_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["CHARLIE_I_Kr"].push_back(tlab.double_attributes["CHARLIE_I_Kr"]);
          labelz_list_double["CHARLIE_I_Ks"].push_back(tlab.double_attributes["CHARLIE_I_Ks"]);
          labelz_list_double["CHARLIE_I_V"].push_back(tlab.double_attributes["CHARLIE_I_V"]);
          labelz_list_double["CHARLIE_I_dimless_roughness"].push_back(tlab.double_attributes["CHARLIE_I_dimless_roughness"]);
          labelz_list_double["CHARLIE_I_dstar"].push_back(tlab.double_attributes["CHARLIE_I_dstar"]);
          labelz_list_double["CHARLIE_I_threshold_incision"].push_back(tlab.double_attributes["CHARLIE_I_threshold_incision"]);
          labelz_list_double["CHARLIE_I_threshold_entrainment"].push_back(tlab.double_attributes["CHARLIE_I_threshold_entrainment"]);
          labelz_list_double["CHARLIE_I_Krmodifyer"].push_back(tlab.double_attributes["CHARLIE_I_Krmodifyer"]);
        }
        break;

  
      // defaut case means the law has no correspondance so it does not do anything
      default: 
        break;
    }
  }

}

void ModelRunner::initialise_label_list(std::vector<labelz> these_labelz)
{
  this->labelz_list = these_labelz;
  this->n_labels = int(these_labelz.size());
  this->prepare_label_to_list_for_processes();

  // initialising the sediment tracking device
  is_there_sed_here = std::vector<bool>(this->io_int["n_elements"], false);
  sed_prop_by_label = std::map<int, std::vector<std::vector<double> > >() ;
}


//#################################################
//#################################################
//#################################################
//######### Solving depression ####################
//#################################################
//#################################################
//#################################################





// This function processes every existing lakes and check if the new topography has an outlet for them
// if it does , it empty what needs to be emptied at the outlet
// whatever remains after all of this is added to each nodes separatedly which simulates mass-balance
// POSSIBLE OPTIMISATION::Recreating lake objects, but I would need a bit of refactoring the Lake object. Will seee if this is very slow
void ModelRunner::process_inherited_water()
{
  for(auto& tlake:this->lakes)
  {
    if(tlake.is_now >= 0)
      continue;

    // getting node underwater
    std::vector<int>& unodes = tlake.nodes;


    double minelev = std::numeric_limits<double>::max();
    int minnodor = -9999;
    double sumwat = 0;
    for( auto node:unodes)
    {
      if(surface_elevation[node] < minelev)
      {
        minnodor = node;
        minelev = surface_elevation[node];
      }
      // sumwat += this->io_double_array["lake_depth"][node] * this->dx * this->dy / this->timestep ;
    }

    this->chonk_network[minnodor].add_to_water_flux(tlake.volume_water/this->timestep);

  }

  // Done with reshuffling the water

}





//#################################################
//#################################################
//#################################################
//######### extracting attributes #################
//#################################################
//#################################################
//#################################################



xt::pytensor<double,1> ModelRunner::get_water_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_water_flux();
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_undifferentiated() + tchonk.get_erosion_flux_only_sediments() + tchonk.get_erosion_flux_only_bedrock() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_bedrock_only_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_only_bedrock() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_erosion_sed_only_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux_only_sediments() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_sediment_creation_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_sediment_creation_flux() ;
  }
  return output;

}


xt::pytensor<double,1> ModelRunner::get_sediment_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_sediment_flux();
  }
  return output;

}


xt::pytensor<double,1> ModelRunner::get_other_attribute(std::string key)
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_other_attribute(key);
  }
  return output;
}

std::vector<xt::pytensor<double,1> > ModelRunner::get_label_tracking_results()
{
  std::vector<xt::pytensor<double,1> > output;
  for(int i=0; i<this->n_labels; i++)
  {
    xt::pytensor<double,1> temp = xt::zeros_like(this->surface_elevation);
    output.push_back(temp);
  }

  for( int i=0; i< io_int["n_elements"];i++)
  {
    chonk& tchonk =this->chonk_network[i];
    for(int j=0; j<this->n_labels; j++)
    {
      output[j][i] = tchonk.get_other_attribute_array("label_tracker")[j];
    }
  }

  return output;

}





//#################################################
//#################################################
//#################################################
//######### DEBUGGING functions ###################
//#################################################
//#################################################
//#################################################

void ModelRunner::DEBUG_check_weird_val_stacks()
{
  int non_valid_mfstack = 0;
  int non_valid_mfrec = 0;
  int non_valid_don = 0;
  int non_valid_mflength = 0;
  for (size_t i=0; i<this->io_int["n_elements"]; i++)
  {
    if(this->graph.get_MF_stack_at_i(i)<0 || this->graph.get_MF_stack_at_i(i)>=io_int["n_elements"])
      non_valid_mfstack++;

    std::vector<int> rec = this->graph.get_MF_receivers_at_node(i);
    for (auto node:rec)
    {
      if(node<-2 || node>=io_int["n_elements"])
        non_valid_mfrec++;
    }
    
    std::vector<int> gurg = this->graph.get_MF_donors_at_node(i);
    for (auto node:gurg)
    {
      if(node<-2 || node>=io_int["n_elements"])
        non_valid_don++;
    }
    // std::vector<int> rec = this->graph.get_MF_donors_at_node(i);
    // for (auto node:rec)
    // {
    //   if(node<0 || (node>=io_int["n_elements"]))
    //     non_valid_don++;
    // }
  }
  std::cout << "FOUND N INVALID: " << non_valid_mfstack << " MSTACK || " << non_valid_mfrec << " MREC || " << non_valid_don << " MDON" << std::endl;
}


// Give the deposition fluxe from lakes to the 
void ModelRunner::drape_deposition_flux_to_chonks()
{


  std::vector<char> isinhere(this->io_int["n_elements"],'n');


  for(auto& loch:this->lakes)
  { 
    // Checking if this is a main lake
    if(loch.is_now >= 0)
      continue;
    if(loch.volume_sed == 0)
      continue;

    double ratio_of_dep = loch.volume_sed/loch.volume_water;

    // NEED TO DEAL WITH THAT BOBO
    if(ratio_of_dep > 1)
    {
      std::cout << "POSSIBLY MISSING " << loch.volume_sed * (ratio_of_dep - 1) << std::endl;
      ratio_of_dep = 1;
    }

    double total = loch.volume_sed;
    for(auto no:loch.nodes)
    {
      if(isinhere[no] == 'y')
        throw std::runtime_error("Double lakecognition");
      isinhere[no] = 'y';

      total -= ratio_of_dep * (topography[no] - surface_elevation[no]) * cellarea;
      double slangh = ratio_of_dep * (topography[no] - surface_elevation[no]) / timestep;
      if(!std::isfinite(slangh))
      {
        std::cout << "WARNING:: NAN IN SED CREA DURING LAKE DRAPING" << std::endl;
        std::cout << ratio_of_dep << "/" << (topography[no] - surface_elevation[no]) << std::endl;
        slangh = 0;
      }

      chonk_network[no].add_sediment_creation_flux(slangh);
      chonk_network[no].add_deposition_flux(slangh); // <--- This is solely for balance calculation
      chonk_network[no].set_other_attribute_array("label_tracker", loch.label_prop);

    }
    // Seems fine here...
    std::cout << "BALANCE LAKE = " << total << " out of " << loch.volume_sed << std::endl;
  }

}

void  ModelRunner::find_underfilled_lakes_already_processed_and_give_water(int SS_ID, std::vector<bool>& is_processed )
{
  throw std::runtime_error("ModelRunner::find_underfilled_lakes_already_processed_and_give_water is fully deprecated");
}

xt::pytensor<int,1> ModelRunner::get_lake_ID_array_raw()
{
  xt::pytensor<int,1> output = xt::zeros<int>({this->io_int["n_elements"]}) -1;

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    if(node_in_lake[i] >= 0)
      output[i] = node_in_lake[i];
  }
  return output;
}


xt::pytensor<int,1> ModelRunner::get_lake_ID_array()
{
  throw std::runtime_error("ModelRunner::get_lake_ID_array is deprecated");
}


xt::pytensor<int,1> ModelRunner::get_mstack_checker()
{
  xt::pytensor<int,1>& mstack = this->graph.get_MF_stack_full_adress();
  xt::pytensor<int,1>  output = xt::zeros<int>({this->io_int["n_elements"]}) -1;
  int i = 0;
  for(auto n: mstack)
  {
    output[n] = i;
    i++;
  }
  return output;
}


void ModelRunner::print_chonk_info(int node)
{
  chonk& tchonk = this->chonk_network[node];
  // std::cout << "===== CHONK DEBUGGER =====" std::endl;
  std::cout << "=ID:" << node << "= lake:" <<node_in_lake[node] << "= processed:" << is_processed[node]  <<" =" << std::endl;
  std::cout << "=Water flux: " << tchonk.get_water_flux() << " =" << std::endl;

}

//#################################################################################
//#################################################################################
//#################################################################################
//#################################################################################
// OTher functions, with a clear one-off utility ##################################
//#################################################################################
//#################################################################################
//#################################################################################



xt::pytensor<double,1> pop_elevation_to_SS_SF_SPIL(xt::pytensor<int,1>& stack, xt::pytensor<int,1>& rec,xt::pytensor<double,1>& length , xt::pytensor<double,1>& erosion, 
  xt::pytensor<double,1>& K, double n, double m, double cellarea)
{
  // first I need to initialise a couple of variabes
  xt::pytensor<double,1> A = xt::zeros<double>({stack.size()});
  xt::pytensor<double,1> elevation = xt::zeros<double>({stack.size()});

  std::vector<int> ndonors(stack.size(),0);
  // Accunulating drainage area first
  for(int i =  stack.size(); i>=0; i--)
  {
    int this_node = stack[i];
    A[this_node] += cellarea;
    int this_rec = rec[this_node];
    if(this_rec == this_node)
      continue;
    A[this_rec] += A[this_node];
    ndonors[this_rec] += 1;
  }


  for(auto node:stack)
  {
    if(node == rec[node])
      continue;
    int this_rec = rec[node];
    elevation[node] = elevation[this_rec] + std::exp(1/n * (std::log(erosion[node]) - m *std::log(A[node]) - std::log(K[node]) ) + std::log(length[node]) );
  }

  return elevation;

}



//+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/
//+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/

//###############################################################
//###############################################################
//###############################################################
//###### DEPRECATED ######## DEPRECATED ####### DEPRECATED ######
//###############################################################
//###############################################################
//###############################################################



// void ModelRunner::find_nodes_to_reprocess(int start, std::vector<bool>& is_processed, std::vector<int>& nodes_to_reprocess, std::vector<int>& nodes_to_relake,
//   std::vector<int>& nodes_to_recompute_neighbors_at_the_end, int lake_to_avoid)
// {
//   // Alright Now is the time to finally comment this function as it seems to be the reason of mass balance and sediment nan bug :(
//   // I have been dreading this moment as this function has been written in one go alongside all the lake reprocessing routines

//   // this was a debugging statement marking the start of the function to locate where it first segfaulted
//   // std::cout << "start" << std::endl;

//   // Some magic to keep a nice balance between cpu and memory optimisation: I initialise a vector assuming I will probably not need to reprocess > 1/4 of the nodes
//   // if I eceed this size, I use push_back whihc is slightly slower
//   int tsize = int(round(this->io_int["n_elements"]/4));
//   // I am initialising various ID to insert and read nodes in this vector
//   int insert_id_trav = 0,insert_id_lake = 0,reading_id = -1, insert_id_proc = 1;

//   // traversal is my vector storing the nodes to reprocess in a BFS search-like algorithm. I think.
//   std::vector<int> traversal;traversal.reserve(tsize);
//   // the Q storing the nodes to process
//   std::vector<int> tQ;tQ.reserve(tsize);
//   // the lake IDs to (re)process
//   std::vector<int> travlake;travlake.reserve(tsize);
//   // Sets to keep track of nodes already processed in this function. Again, should be more efficient than a full vector as the number of nodes whshould be relatively low
//   std::set<int> is_in_Q,is_in_lake;

//   // Starting the processing with the starting node ofc
//   tQ.emplace_back(start);
//   is_in_Q.insert(start);

//   // const bool is_in = container.find(element) != container.end();
//   // Iterating til' I decide it
//   while(true)
//   {
//     // Reading the next node in the Q
//     // if first time, -1 +1 =0 so it works aye
//     reading_id++;
//     // if I reach the end of me Q, tis the end
//     // It works because I reserve some space in it, so I increase its CAPACITY but not its SIZE which grows with emplace_back
//     if(reading_id >= tQ.size())
//       break;

//     // getting the node
//     int this_node = tQ[reading_id];
//     bool recomputed_neight = false;
//     // std::cout << "|Q_NODE:" << this_node;
//     // if node is not start, I put it into the traversal, cause it has already been checked
//     if(this_node != start)
//     {  
//       if(insert_id_trav < tsize)
//       {
//         traversal.emplace_back(this_node);
//         insert_id_trav++;
//       }
//       else
//         traversal.push_back(this_node);
//     }

//     std::vector<int> golog = this->graph.get_MF_receivers_at_node(this_node);
//     // iterating through all his neighbors
//     for(auto node:golog)
//     {
     
//       // if the neightbor has not been processed originally no reproc
//       if(is_processed[node] == false)
//         continue;

//       // if the node is already in Q skip
//       if(is_in_Q.find(node) != is_in_Q.end())
//         continue;
     
//       // getting the lake id
//       int this_lake_id = node_in_lake[node];

//       // If this node is in a lake, but not processed yet by this function and not the starting one
//       if(this_lake_id >= 0 && is_in_lake.find(node) == is_in_lake.end()) // && this_node != start)
//       {

//         // Checking and avoiding dedundancy-> if is in the lake to avoid -> rework the receivers of the outlet and process it
//         if(this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid)
//         {
//           std::vector<int> new_rec;
//           std::vector<int> datiznogoud = this->graph.get_MF_receivers_at_node(this_node);
//           for(auto recnode:datiznogoud)
//           {
//             if(recnode != node)
//               new_rec.push_back(recnode);
//           }
//           // std::cout << "IT HAPPENS" << std::endl;
//           this->graph.update_receivers_at_node(this_node, new_rec);
//           recomputed_neight = true;
//           continue;
//         }

//         // if it passes this test, then the node is in a lake to reprocess

//         is_in_lake.insert(node);
//         if(insert_id_lake < tsize )
//         {

//           travlake.emplace_back(node);
//           insert_id_lake++;
//         }
//         else
//           travlake.push_back(node);

//         // I do not want to reprocess this node, so I am putting "in the Q" so that it gets ignored now. I know I have to reprocess it
//         is_in_Q.insert(node);
        
//         continue;
//       }

//       // if is in lake to avoid at start, I stop here
//       if(node == start && (this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid) )
//         continue;

//       // otherwise, I'll need to check this node and put it in teh traversal yolo
//       is_in_Q.insert(node);
//       if(insert_id_proc < tsize )
//       {
//         tQ.emplace_back(node);
//         insert_id_proc++;
//       }
//       else
//         tQ.push_back(node);

//     }
//     if(recomputed_neight)
//       nodes_to_recompute_neighbors_at_the_end.push_back(this_node);

//   }


//   // for (auto bugh:traversal)
//   //   if(node_in_lake[bugh] >= 0)
//   //     throw std::runtime_error("ERROAR! this nor should not be in this traversal");
//   // if(travlake.size()>0)
//   //   std::cout << "{" << travlake.size() << "}";

//   nodes_to_reprocess = std::move(traversal);
//   nodes_to_relake = std::move(travlake);




//   // // std::cout << "end" << std::endl;
//   // std::map<int,int> node_counter;
//   // for(auto n:nodes_to_reprocess)
//   // {
//   //   node_counter[n] = 0;
//   // }
//   // for(auto n:nodes_to_reprocess)
//   // {
//   //   node_counter[n] += 1;
//   //   if(node_counter[n] > 1)
//   //     throw std::runtime_error("Doubled reprocessing error");
//   // }

// }


// void Lake::ingest_other_lake(
//    Lake& other_lake,
//    std::vector<int>& node_in_lake, 
//    std::vector<bool>& is_in_queue,
//    std::vector<Lake>& lake_network,
//    xt::pytensor<double,1>& topography
//    )
// {
//   std::cout << "LAKE " << this->lake_id << " is ingesting lake " << other_lake.get_lake_id() << std::endl;
//   // getting the attributes of the other lake
//   std::vector<int>& these_nodes = other_lake.get_lake_nodes();
//   std::vector<int>& these_node_in_queue = other_lake.get_lake_nodes_in_queue();
//   std::unordered_map<int,double>& these_depths = other_lake.get_lake_depths();
//   std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& this_PQ = other_lake.get_lake_priority_queue();

//   // if(this->water_elevation != other_lake.get_water_elevation())
//   // {
//   //   std::cout << this->water_elevation << "||" << other_lake.get_water_elevation() << std::endl;
//   //   throw std::runtime_error("IOIOIOIOIOIO");
//   // }

//   // merging them into this lake while node forgetting to label them as visited and everything like that
//   for(auto node:these_nodes)
//   {
//     if(std::find(this->nodes.begin(), this->nodes.end(), node) == this->nodes.end())
//       this->nodes.push_back(node);
//     node_in_lake[node] = this->lake_id;
//   }

//   for(auto node:these_node_in_queue)
//   {
//     this->node_in_queue.push_back(node);
//     is_in_queue[node] = true;
//   }

//   this->depths.insert(these_depths.begin(), these_depths.end());

//   this->ngested_nodes += int(other_lake.get_n_nodes());
//   // this->n_nodes += other_lake.get_n_nodes();

//   // if(int(these_nodes.size()) !=other_lake.get_n_nodes() )
//   //   throw std::runtime_error("TRRTRTTSTSK");

//   // Transferring the PQ (not ideal but meh...)
//   while(this_PQ.empty() == false)
//   {
//     nodium this_nodium = this_PQ.top();
//     is_in_queue[this_nodium.node] = true;
//     this->depressionfiller.emplace(nodium(this_nodium.node, topography[this_nodium.node] ));
//     this_PQ.pop();
//   }
//   this->pour_sediment_into_lake(other_lake.get_volume_of_sediment(), other_lake.get_lake_lab_prop());

//   this->volume += other_lake.get_lake_volume();

//   // Deleting this lake and setting its parent lake
//   int save_ID = other_lake.get_lake_id();


//   Lake temp = Lake(save_ID);
//   other_lake = temp;
//   other_lake.set_parent_lake(this->lake_id);
//   this->ingested_lakes.push_back(other_lake.get_lake_id());

//   for(auto lid:other_lake.get_ingested_lakes())
//     lake_network[lid].set_parent_lake(this->lake_id);


//   return;
// }

// void Lake::pour_sediment_into_lake(double sediment_volume, std::vector<double> label_prop)
// {
//   if(this->volume_of_sediment<0)
//     std::cout << "NEG BEFORE POURING SED" << std::endl;
//   if(sediment_volume<0)
//     std::cout << "NEG POURED" << std::endl;

//   std::vector<double> coplaklab = this->lake_label_prop;
//   std::vector<double> copaddlab = label_prop;
  
//   if(this->lake_label_prop.size()>0)
//   {
//     // std::cout << "bo";
//     for(auto lb:this->lake_label_prop)
//       if(std::isfinite(lb) == false)
//         std::cout << "NANINF in lake already" << std::endl;
//     for(auto lb:label_prop)
//       if(std::isfinite(lb) == false)
//         std::cout << "NANINF coming in" << std::endl;
//     if(this->volume_of_sediment > 0)
//     {
//       // std::cout << "go";
//       this->lake_label_prop = mix_two_proportions(this->volume_of_sediment, this->lake_label_prop, sediment_volume, label_prop);
//       // std::cout << "ris";
//     }
//     else
//     {
//       this->lake_label_prop = label_prop;
//     }

      
//     for(auto lb:this->lake_label_prop)
//       if(std::isfinite(lb) == false)
//       {
//         std::cout << "NANINF in lake after::" << this->volume_of_sediment << "||" << sediment_volume << std::endl;
//         for(auto yu:this->lake_label_prop)
//           std::cout << yu << "|";
//         std::cout << std::endl;
//         for(auto yu:label_prop)
//           std::cout << yu << "|";
//         std::cout << std::endl;
//         for(auto yu:copaddlab)
//           std::cout << yu << "|";
//         std::cout << std::endl;
//         for(auto yu:coplaklab)
//           std::cout << yu << "|";
//         std::cout << std::endl;
        
//         throw std::runtime_error("NANINF in lake sed pouring");
//       }
//     // std::cout << "ris";
//   }
//   else
//     this->lake_label_prop = label_prop;

//   this->volume_of_sediment += sediment_volume;
//   if(this->volume_of_sediment<0)
//   {
//     std::cout << "NEG AFTER POURING SED " << sediment_volume << std::endl;
//     // throw std::runtime_error("sedneg issue number 5");
//   }

// }

// // Fill a lake with a certain amount of water for the first time of the run
// void Lake::pour_water_in_lake(
//   double water_volume,
//   int originode,
//   std::vector<int>& node_in_lake,
//   std::vector<bool>& is_processed,
//   xt::pytensor<int,1>& active_nodes,
//   std::vector<Lake>& lake_network,
//   xt::pytensor<double,1>& surface_elevation,
//   xt::pytensor<double,1>& topography,
//   NodeGraphV2& graph,
//   double cellarea,
//   double dt,
//   std::vector<chonk>& chonk_network,
//   double& Ql_out
//   )
// { 

//   std::cout << "Pouring " << water_volume << " water (rate = " << water_volume/dt << ") into " << this->lake_id << " from node " << originode << std::endl;


//   std::cout << "Entering water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;


//   if(water_volume < -1)
//     throw std::runtime_error("NegWatPoured!!!");


//   double save_entering_water = water_volume;
//   double save_preexistingwater = this->volume;
//   int n_labels = int(chonk_network[originode].get_other_attribute_array("label_tracker").size());

//   // Some ongoing debugging
//   // if(originode == 8371)
//   //   std::cout << "processing the problem node W:" << chonk_network[originode].get_water_flux() << std::endl;

//   // first cancelling the outlet to make sure I eventually find a new one, If I pour water into the lake I might merge with another one, etc
//   this->outlet_node = -9999;

//   double sum_this_fill = 0;

//   // no matter if I am filling a new lake or an old one:
//   // I am filling a vector of nodes already in teh system (Queue or lake)
//   std::vector<bool> is_in_queue(node_in_lake.size(),false);
//   for(auto nq:this->node_in_queue)
//     is_in_queue[nq] = true;

//   // First checking if this node is in a lake, if yes it means the lake has already been initialised and 
//   // We are pouring water from another lake 
//   if(node_in_lake[originode] == -1)
//   {
//     // NEW LAKE
//     // Emplacing the node in the queue, It will be the first to be processed
//     if(originode == 0)
//       throw std::runtime_error("0 is originode...");

//     depressionfiller.emplace( nodium( originode, surface_elevation[originode] ) );
//     // This function fills an original lake, hence there is no lake depth yet:
//     this->water_elevation = surface_elevation[originode];
//     is_in_queue[originode] = true;
//     // this->nodes.push_back(originode);
//     // this->n_nodes ++;
//   }



//   // I am processing new nodes while I still have water OR still nodes upstream 
//   // (if an outlet in encountered, the loop is breaked anually)
//   // UPDATE_09_2020:: No idea what I meant by anually.
//   // UPDATE_10_2020:: I meant manually, but with a typo.
//   std::cout << "starting loop" << std::endl;
//   while(depressionfiller.empty() == false && water_volume > 0 )
//   {
//     // std::cout << this->water_elevation << "||" << water_volume << std::endl;
//     // Getting the next node and ...
//     // (If this is the first time I fill a lake -> the first node is returned)
//     // (If this is another node, it is jsut the next one the closest elevation to the water level)
//     nodium next_node = this->depressionfiller.top();

//     // ... removing it from the priority queue 
//     this->depressionfiller.pop();


//     // Initialising a dummy outlet, if this outlet becomes something I shall break the loop
//     int outlet = -9999;

//     // Adding the upstream neighbors to the queue and checking if there is an outlet node
//     outlet = this->check_neighbors_for_outlet_or_existing_lakes(next_node, graph, node_in_lake, lake_network, surface_elevation,
//      is_in_queue, active_nodes, chonk_network, topography);

//     bool isinnodelist = std::find(this->nodes.begin(), this->nodes.end(), next_node.node) == this->nodes.end();
//     // If I have an outlet, then the outlet node is positive
//     if(outlet >= 0)
//     {
//       // this->water_elevation = next_node.elevation;

//       // I therefore save it and break the loop
//       // std::cout << "{" << outlet << "|" << this->outlet_node << "}";
//       this->outlet_node = outlet;
//       // and readding the node to the depression
//       this->depressionfiller.emplace(next_node);


//       // break;
//     }



//     //Decreasing water volume by filling teh lake
//     double dV = this->n_nodes * cellarea * ( next_node.elevation - this->water_elevation );

//     // if(isinnodelist == false)
//     //   dV = 0;

//     sum_this_fill += dV;
//     // I SHOULD NOT HAVE TO DO THAT!!!! PROBABLY LINKED TO NUMERICAL UNSTABILITIES BUT STILL
//     if(dV > - 1e-3 && dV < 0)
//       dV = 0;

//     if(dV<0)
//     {
//       std::cout << "Arg should not have neg filling::" << dV << std::endl;
//       dV = 0;
//       // std::string ljsdfld;
//       // if(is_processed[next_node.node] == true)
//       //   ljsdfld = "true";
//       // else
//       //   ljsdfld = "false";
//       // std::cout << "DV::" << dV << " :: " << node_in_lake[next_node.node]  << " :: " << this->n_nodes << "::" << this->water_elevation << " :: " << next_node.elevation << " :: " << ljsdfld << "::" << outlet << std::endl;
//       // throw std::runtime_error("negative dV lake filling");
//     }

//     water_volume -= dV;
    

//     this->volume += dV;


//     // The water elevation is the elevation of that Nodium object
//     // (if 1st node -> elevation of the bottom of the depression)
//     // (if other node -> lake water elevation)
//     this->water_elevation = next_node.elevation;

//     // Alright, what is hapenning here:
//     // I sometimes ingest other lakes, but I do not consider the ingested ndoes in my n_nodes at ingestion
//     // Why? because I would be rising a lot more nodes to water elevation than intended and artificially fill my lakes.
//     // So I am adding tham after dealing wih dV
//     if(this->ngested_nodes>0)
//     {
//       this->n_nodes += this->ngested_nodes;
//       std::cout << "ADDED " <<  this->ngested_nodes << " AFTER DV AND INGESTING" << std::endl;
//       this->ngested_nodes = 0;
//     }

//     if(outlet >= 0)
//       break;
//         // Otehr wise, I do not have an outlet and I can save this node as in depression
//     if(isinnodelist)
//     {
//       this->nodes.push_back(next_node.node);
      
//       this->n_nodes ++;
//     }
//     node_in_lake[next_node.node] = this->lake_id;
    
//     // At this point I either have enough water to carry on or I stop the process
//   }
//   std::cout << "ending loop" << std::endl;


//   double local_balance = (save_entering_water - water_volume)/dt -  sum_this_fill/dt;

//   // if(save_entering_water = save_preexistingwater)

//   std::cout << "LOCAL BALANCE SHOULD BE 0::" << local_balance << std::endl;
//   std::cout << "After raw filling lake water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;
//   std::cout << "Outletting in " << this->outlet_node << " AND I HAVE " << this->n_nodes << std::endl;;

//   // if(this->outlet_node>=0)
//   // {
//   //   if(node_in_lake[this->outlet_node]>=0)
//   //   {
//   //     if(node_in_lake[this->outlet_node] == this->lake_id || node_in_lake[this->outlet_node] == this->get_parent_lake())
//   //     {

//   //       std::cout << "SSDF" << node_in_lake[this->outlet_node] << "||" << this->lake_id <<  "||" << this->nodes.size() << "||" << this->get_parent_lake() << std::endl;
//   //       throw std::runtime_error("Fatal Error:: outletinlake");
//   //     }
//   //   }
//   // }




//   // checking that I did not overfilled my lake:
//   if(water_volume < 0 && this->outlet_node <0)
//   {
//     double extra = abs(water_volume);
//     // this->n_nodes -= 1;
//     std::cout << this->nodes.size() - 1 << std::endl;
//     int extra_node = this->nodes[this->nodes.size() - 1];
//     std::cout << ":bulf" << std::endl;


//     this->depressionfiller.emplace(nodium(extra_node,topography[extra_node]));
//     this->nodes.erase(this->nodes.begin() + this->nodes.size() - 1);
//     sum_this_fill -= extra;
//     double dZ = extra / this->n_nodes / cellarea;


//     this->water_elevation -= dZ;
//     water_volume = 0;
//     this->volume -= extra;
//   }




//   Ql_out += sum_this_fill/dt;

//   // std::cout << "Water balance: " << this->volume - save_preexistingwater + water_volume << "should be equal to " << save_entering_water << std::endl;


//   // Labelling the node in depression as belonging to this lake and saving their depth
//   std::map<int,int> counting_nodes;
//   for(auto Unot:this->nodes)
//     counting_nodes[Unot] = 0;

//   for(auto Unot:this->nodes)
//   {
//     counting_nodes[Unot]++;
//     if(counting_nodes[Unot] >1)
//       throw std::runtime_error("double_nodation_there");
//     // std::cout << Unot << "|";
//     double this_depth = this->water_elevation - surface_elevation[Unot];
//     this->depths[Unot] = this_depth;
//     topography[Unot] = surface_elevation[Unot] + this_depth;

//     node_in_lake[Unot] = this->lake_id;
//     double temp_watflux = chonk_network[Unot].get_water_flux();
//     double temp_sedflux = chonk_network[Unot].get_sediment_flux();
//     std::vector<double> oatlab = chonk_network[Unot].get_other_attribute_array("label_tracker");
//     chonk_network[Unot].reset();
//     chonk_network[Unot].set_water_flux(temp_watflux);
//     chonk_network[Unot].set_sediment_flux(temp_sedflux,oatlab);
//     chonk_network[Unot].initialise_local_label_tracker_in_sediment_flux(n_labels);
//     // std::cout <<  chonk_network[Unot].get_sediment_creation_flux() << "||";
//   }
//     // std::cout << " in " << this->lake_id << std::endl;

//   // std::cout << "Water volume left: " << water_volume << std::endl;
//   // Transmitting the water flux to the SS receiver not in the lake
//   if(water_volume > 0 && this->outlet_node >= 0)
//   {

//     // If the node is inactive, ie if its code is 0, the fluxes can escape the system and we stop it here
  
//     // Otherwise: calculating the outflux: water_volume_remaining divided by the time step
//     double out_water_rate = water_volume/(dt);

//     // Getting all the receivers and the length to the oulet
//     std::vector<int> receivers = graph.get_MF_receivers_at_node(this->outlet_node);
//     std::vector<double> length = graph.get_MF_lengths_at_node(this->outlet_node);
//     // And finding the steepest slope 
//     int SS_ID = -9999; 
//     double SS = 0; // hmmmm I may need to change this name
//     for(size_t i=0; i<receivers.size(); i++)
//     {
//       int nodelakeid = node_in_lake[receivers[i]];

//       if(nodelakeid > -1)
//       {
//         if( nodelakeid == this->lake_id  || lake_network[nodelakeid].get_parent_lake() == this->lake_id)
//           continue;
//       } 

//       double elevA = surface_elevation[this->outlet_node];
//       double elevB = surface_elevation[receivers[i]];
//       int testlake = node_in_lake[this->outlet_node];
//       if( testlake >= 0)
//       {
//         if(lake_network[testlake].get_parent_lake() >=0)
//           testlake = lake_network[testlake].get_parent_lake();

//         elevA += lake_network[testlake].get_lake_depth_at_node(this->outlet_node, node_in_lake);

//       }
//       testlake = node_in_lake[receivers[i]] ;

//       if( testlake >= 0)
//       {
//         if(lake_network[testlake].get_parent_lake() >= 0)
//           testlake = lake_network[testlake].get_parent_lake();

//         elevB += lake_network[testlake].get_lake_depth_at_node(receivers[i], node_in_lake);

//       }

//       double this_slope = (elevA - elevB)/length[i];


//       if(this_slope >= SS )
//       {
//         SS = this_slope;
//         SS_ID = receivers[i];
//         // std::cout << "HURE::" << SS_ID << "||" << nodelakeid << "||" << node_in_lake[SS_ID]  << std::endl;

//       }
//     }

//     if(SS_ID < 0)
//     {
//       int sr = graph.get_Srec(this->outlet_node);
//       int lsr = node_in_lake[sr];
//       if(sr != this->outlet_node && lsr != this->lake_id && lake_network[lsr].get_parent_lake() != this->lake_id)
//       {
//         // std::cout << "HERE" << std::endl;
//         SS_ID = sr;
//         SS = 0;
//       }
//       // std::cout << "Warning::lake outlet is itself a lake bottom? is it normal?" << std::endl;
//       // yes it can be: flat surfaces
//       else
//       {
//         // std::cout << "HARE" << std::endl;
//         if(node_in_lake[SS_ID] < 0)
//           throw std::runtime_error("Outlet potential ambiguity");

//         SS_ID = this->outlet_node;
//         SS = 0.;
//       }
//     }

//     // here I am checking if my receiver is directly a lake, in whihc case I put my outlet directly in this lake to trigger merge. It will be simpler that way
//     int SSlid = node_in_lake[SS_ID];
//     bool SSlid_happened = false;
//     if(SSlid >= 0)
//     {
//       if(lake_network[SSlid].get_parent_lake() >= 0)
//         SSlid = lake_network[SSlid].get_parent_lake();

//       SSlid_happened = true;
//       // and I add the outlet of this thingy to the lake
//       this->n_nodes ++;
//       // removing it from the depressionfiller queue while making sure I readd others (the node should be right at the top of the queue so it will not empty and refill the whole queue)
//       std::vector<nodium> toreadd;
//       nodium dat = this->depressionfiller.top();
//       this->depressionfiller.pop();
//       while(dat.node != this->outlet_node)
//       {
//         toreadd.push_back(dat);
//         dat = this->depressionfiller.top();
//         this->depressionfiller.pop();
//       }
//       for(auto dut:toreadd)
//         this->depressionfiller.push(dut);

//       // Formerly add the node to the lake
//       if(std::find(this->nodes.begin(), this->nodes.end(), this->outlet_node) == this->nodes.end())
//         this->nodes.push_back(this->outlet_node);

//       this->depths[this->outlet_node] = this->water_elevation - surface_elevation[this->outlet_node]; // should be 0 here yo
//       node_in_lake[this->outlet_node] = this->lake_id;

//       this->outlet_node = SS_ID;

//       double temp_watflux = chonk_network[this->outlet_node].get_water_flux();
//       double temp_sedflux = chonk_network[this->outlet_node].get_sediment_flux();
//       std::vector<double> oatlab = chonk_network[this->outlet_node].get_other_attribute_array("label_tracker");
//       chonk_network[this->outlet_node].reset();
//       chonk_network[this->outlet_node].set_water_flux(temp_watflux);
//       chonk_network[this->outlet_node].set_sediment_flux(temp_sedflux,oatlab);
//       chonk_network[this->outlet_node].initialise_local_label_tracker_in_sediment_flux(n_labels);
//     }

//     // resetting the outlet CHONK
//     // if(is_processed[this->outlet_node])
//     // {
//     //   chonk_network[this->outlet_node].cancel_split_and_merge_in_receiving_chonks(chonk_network,graph,dt);
//     // }

//     this->outlet_chonk = chonk(-1, -1, false); //  this is creating a "fake" chonk so its id is -1
//     this->outlet_chonk.reinitialise_moving_prep();
//     this->outlet_chonk.initialise_local_label_tracker_in_sediment_flux( n_labels );
//     // forcing the new water flux

//     this->outlet_chonk.set_water_flux(out_water_rate);

//     // Forcing receivers
//     std::cout << "SS ID for routletting is " << SS_ID << " and should receive " << out_water_rate << std::endl;
//     std::vector<int> rec = {SS_ID};
//     std::vector<double> wwf = {1.};
//     std::vector<double> wws = {1.};
//     std::vector<double> Strec = {SS};

//     // if(SS_ID == this->outlet_node)
//     //   throw std::runtime_error("LakeReroutingError::Lake outlet is itself");


//     this->outlet_chonk.external_moving_prep(rec,wwf,wws,Strec);
//     if(chonk_utilities::has_duplicates(rec))
//       throw std::runtime_error("DUPLICATES FOUND HERE #3");

//     if(this->volume_of_sediment > this->volume)
//     {
//       double outsed = this->volume_of_sediment - this->volume;
//       this->volume_of_sediment -= outsed;
//       this->outlet_chonk.set_sediment_flux(0., this->lake_label_prop);
//       this->outlet_chonk.add_to_sediment_flux(outsed, this->lake_label_prop);
//     }
//     else
//     {
//       std::vector<double> baluf_2 (chonk_network[originode].get_other_attribute_array("label_tracker").size(),0.);
//       this->outlet_chonk.set_sediment_flux(0.,baluf_2);
//     }



//     if(active_nodes[this->outlet_node] == 0)
//     {
//       chonk_network[this->outlet_node] = this->outlet_chonk;
//       chonk_network[this->outlet_node].set_current_location(this->outlet_node);
//       chonk_network[this->outlet_node].reinitialise_moving_prep();
//     }

//     // ready for re calculation, but it needs to be in the env object
  

//   }
//   std::cout << "done with lake " << this->lake_id << " outlet is " << this->outlet_node << std::endl;


//   return;
// }

// // This function checks all the neighbours of a pixel node and return -9999 if there is no receivers
// // And the node index if there is a receiver
// // 
// int Lake::check_neighbors_for_outlet_or_existing_lakes(
//   nodium& next_node, 
//   NodeGraphV2& graph, 
//   std::vector<int>& node_in_lake, 
//   std::vector<Lake>& lake_network,
//   xt::pytensor<double,1>& surface_elevation,
//   std::vector<bool>& is_in_queue,
//   xt::pytensor<int,1>& active_nodes,
//   std::vector<chonk>& chonk_network,
//   xt::pytensor<double,1>& topography
//   )
// {

//   // Getting all neighbors: receivers AND donors
//   // Ignore dummy, is just cause I is too lazy to overlaod my functions correctly
//   // TODO::Overload your function correctly
//   std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node.node, active_nodes, neightbors, dummy);

//   // std::cout << "IS THIS EVEN CALLED???:: " << next_node.node << std::endl;

//   // No outlet so far
//   int outlet = -9999;
//   bool has_eaten = false;

//   std::vector<int> rec_of_node = chonk_network[next_node.node].get_chonk_receivers_copy();

//   // Checking all neighbours
//   for(auto node : neightbors)
//   {
//     // std::cout << "From node " << next_node.node << " checking " << node << std::endl;
//     // First checking if the node is already in the queue
//     // If it is, well I do not need it right?
//     // Right.
//     if(is_in_queue[node]  || node  == next_node.node)
//       continue;

//     // Check if the neighbour is a lake, if it is, I am gathering the ID and the depths
//     int lake_index = -1;
//     if(node_in_lake[node] > -1)
//     {
//       lake_index = node_in_lake[node];
//       // If my node is not in the queue but in the same lake (this happens when refilling the lake with more water)
//       if(lake_index == this->lake_id || lake_network[lake_index].get_parent_lake() == this->lake_id )
//         continue;
//     }


    
//     // lake depth
//     double this_depth = 0.;
//     // getting potentially inherited lake depth
//     if(lake_index >= 0)
//     {
//       if(lake_network[lake_index].get_parent_lake()>=0)
//         lake_index = lake_network[lake_index].get_parent_lake();
//       this_depth = lake_network[lake_index].get_lake_depth_at_node(node, node_in_lake);
//     }

//     // It gives me the elevation to be considered
//     double tested_elevation = surface_elevation[node] + this_depth;

//     // However if there is another lake, and that his elevation is mine I am ingesting it
//     // if the lake has a lower elevation, I am outletting in it
//     // if the lake has greater elevation, I am considering this node as a potential donor
//     if(lake_index > -1 && tested_elevation == next_node.elevation)
//     {
//       // Well, before drinking it I need to make sure that I did not already ddid it
//       if(lake_network[lake_index].get_parent_lake() == this->lake_id)
//         continue;

//       // OK let's try to drink it 
//       this->ingest_other_lake(lake_network[lake_index], node_in_lake, is_in_queue,lake_network,topography);
//       has_eaten = true; 
//       continue;
//     }


    
//     // If the node is at higher (or same) elevation than me water surface, I set it in the queue
//     else if(tested_elevation >= next_node.elevation)
//     {
//       // std::cout << "depressionfiller ingests " << node << " from " << next_node.node << std::endl;
//       this->depressionfiller.emplace(nodium(node,tested_elevation));
//       // Making sure I mark it as queued
//       is_in_queue[node] = true;
//       // Adding the node to the list of nodes in me queue
//       this->node_in_queue.push_back(node);
//     }

//     // Else, if not in queue and has lower elevation, then the current mother node IS an outlet
//     else
//     {

//       int outlake = node_in_lake[next_node.node] ;
//       // std::cout << "{" << outlake << "}";
//        // In some rare cases the outlet is already labelled as in this lake: for example if I already processed the lake before and readd water or more convoluted situations where I had a single pixeld lake
//       if(outlake < 0)
//       {
//         // std::cout <<"gabul1  " << next_node.node << std::endl;
//         outlet = next_node.node;
//       }
//       else if(outlake != this->lake_id && lake_network[outlake].get_parent_lake() != this->lake_id)
//       {
//         // std::cout <<"gabul2 " << next_node.node << std::endl;
//         outlet = next_node.node;
//       }


//       // IMPORTANT::not breaking the loop: I want to get all myneighbors in the queue for potential repouring water in the thingy
//     }

//     // Moving to the next neighbour
//   }

//   // if a lake ahs been ingested, it makes the outlet ambiguous and need reprocessing of this node
//   if(has_eaten)
//   {
//     this->depressionfiller.emplace(next_node);
//     outlet = -9999;
//   }

//   // // outlet is >= 0 -> tehre is an outlet
//   // if(outlet>=0)
//   //   std::cout << "POCHTRAC::" << outlet << "||" << node_in_lake[outlet] << "||" << this->lake_id << std::endl;

//   if(active_nodes[next_node.node] == false)
//   {
//     outlet = next_node.node;
    
//   }

//   return outlet;
// }



// // Give the deposition fluxe from lakes to the 
// void Lake::drape_deposition_flux_to_chonks(std::vector<chonk>& chonk_network, xt::pytensor<double,1>& surface_elevation, double timestep)
// {

//   if(this->volume == 0)
//     return;

//   double ratio_of_dep = this->volume_of_sediment/this->volume;

//   // NEED TO DEAL WITH THAT BOBO
//   if(ratio_of_dep>1)
//     ratio_of_dep = 1;

//   for(auto& no:this->nodes)
//   {
//     double pre = chonk_network[no].get_sediment_creation_flux();
//     if(std::isfinite(pre) ==false)
//      throw std::runtime_error("already naninf before draping");// std::cout <<  chonk_network[no].get_sediment_creation_flux() << "||";
    

//     double slangh = ratio_of_dep * (this->water_elevation - surface_elevation[no]) / timestep;
//     chonk_network[no].add_sediment_creation_flux(slangh);
//     chonk_network[no].set_other_attribute_array("label_tracker", this->outlet_chonk.get_other_attribute_array("label_tracker"));

//     pre = chonk_network[no].get_sediment_creation_flux();
//     if(std::isfinite(pre) == false)
//     {
//       std::cout << ratio_of_dep << "||" << (this->water_elevation - surface_elevation[no]) << "||" << timestep << std::endl;
//       throw std::runtime_error(" naninf after draping");
//     }
//   }
// }

#endif

