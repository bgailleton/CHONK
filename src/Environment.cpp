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
#include <fstream>


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

  // Saving all the attributes
  this->timestep = ttimestep;
  this->ordered_flux_methods = tordered_flux_methods;
  this->move_method = tmove_method;
  
  // By default the lake solver is activated
  this->lake_solver = true;
  
  // Initialising the labelling stuff
  this->initialise_intcorrespondance();
  this->prepare_label_to_list_for_processes(); 

  // And the mass balance recorder/checker/proctor/whatever
  this->Qw_in = 0;
  this->Qw_out = 0;
  this->Ql_in = 0;
  this->Ql_out = 0;

  this->NTIME_DRAPEONLYSEDHAPPENED = 0;
  this->NTIME_DRAPEONLYSEDHAPPENEDMAX = 0;
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{
  CHRONO_start[0] = std::chrono::high_resolution_clock::now();

  // DEBUG STUFF IGNORER
  this->NTIMEPREFLUXCALLED = 0;
  this->NTIME_DRAPEONLYSEDHAPPENED = 0;
  this->NTIME_DRAPEONLYSEDHAPPENEDMAX = 0;

  // Creating the nodegraph and preprocessing the depression nodes
  this->dx = this->io_double["dx"];
  this->dy = this->io_double["dy"];
  this->cellarea = this->dx * this->dy;
  calculated_K = xt::zeros_like(this->topography);

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

  // Dat is the real stuff:
  // Initialising the graph
  double threshold = 0.1;
  if(this->initial_carving)
  {
    threshold = 1e32;
    this->initial_carving = false;
  }
    // threshold = 1e32;

  // std::cout << "BEEF " << sumvecdouble(this->surface_elevation) << std::endl;

  this->graph = NodeGraphV2(this->surface_elevation, this->active_nodes,this->dx, this->dy,
                            this->io_int["n_rows"], this->io_int["n_cols"], this->lake_solver, threshold);
  
  // std::cout << "AFFT " << sumvecdouble(this->surface_elevation) << std::endl;
  // topography is the surface elevation + lake surface and is used to calculated its volume and mass balance  
  this->topography = xt::pytensor<double,1>(this->surface_elevation);

  // vector of lakes
  this->lake_to_process = std::vector<int>();

  // If lake evaporation is on, the model formates an evaporation array
  if(this->lake_evaporation)
  {
    // If single value->rasterise
    if(this->lake_evaporation_spatial == false)
    {
      this->lake_evaporation_rate_spatial = this->lake_evaporation_rate + xt::zeros_like(this->topography);
    }

    // Preprocessing the potential of evaporation for each lake
    this->graph.depression_tree.preprocess_lake_evaporation_potential(this->lake_evaporation_rate_spatial, this->cellarea, this->timestep);
  }

  // Also initialising the Lake graph
  //# incrementor reset to 0
  lake_incrementor = 0;
  this->lakes.clear();

  //# Labelling the nodes in the lake
  this->node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);

  //# Registering the pitnodes of the depression as belonging to their base-depression to be detected as lake
  for(int i = 0; i < this->graph.depression_tree.get_n_dep(); i++)
  {
    if(this->graph.depression_tree.has_children(i) == false)
    {
      this->node_in_lake[this->graph.depression_tree.pitnode[i]] = i;
    }
  }

  // ready to run this time step!

  // Timers
  CHRONO_stop[0] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[0] ++;
  CHRONO_mean[0] += std::chrono::duration<double>(CHRONO_stop[0] - CHRONO_start[0]).count();
}




// This is the main running function
void ModelRunner::run()
{

  // Timer stuff
  CHRONO_start[1] = std::chrono::high_resolution_clock::now();

  // Incrementing the number of timesteps
  this->ndt++;

  // Keeping track of which node is processed, for debugging and lake management
  this->is_processed = std::vector<bool>(io_int["n_elements"],false);
  
  // Deprecated???  
  this->local_Qs_production_for_lakes = std::vector<double>(this->io_int["n_elements"],0);
  
  // Debugging maps for lake stuff
  this->ORIGINALPIT2WAT.clear();
  this->ORIGINALPIT2SED.clear();

  // Cellarea
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
    // if(this->chonk_network[node].get_sediment_flux() < -0.1)
    // {
    //   std::cout << "SEDFLUXNEG_GLOBALCHECEKR" << std::endl;
    //   throw std::runtime_error("Before proc.");
    // }  
    this->process_node(node, is_processed, lake_incrementor, underfilled_lake, this->active_nodes, cellarea, surface_elevation, true);   
    // Switching to the next node in line
    // if(this->chonk_network[node].get_sediment_flux() < -0.1)
    // {
    //   std::cout << "SEDFLUXNEG_GLOBALCHECEKR" << std::endl;
    //   throw std::runtime_error("After proc.");
    // }
  }

  // Calling the finalising function: it applies the changes in topography and I think will apply the lake sedimentation
  this->finalise();

  // Write a debugging file for the lakes
  // this->DEBUG_write_lakes_to_file("debug_lake_" + std::to_string(this->ndt) +".csv");
  // Done

  // keeping this debugger in case at the moment. It detects when a node is processed
  int ndfgs=0;
  for( size_t i=0; i< is_processed.size() ; i++)
  {
    if(is_processed[i] == false)
      ndfgs++;
  }
  if(ndfgs>0)
  {
    std::cout << "WARNING::Node Unprocessed::"  + std::to_string(ndfgs) << std::endl;
  }


  // Timer stuff
  CHRONO_stop[1] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[1] ++;
  CHRONO_mean[1] += std::chrono::duration<double>(CHRONO_stop[1] - CHRONO_start[1]).count();

  // if(this->NTIME_DRAPEONLYSEDHAPPENED >0 )
  //   std::cout << "Draped only sed " << this->NTIME_DRAPEONLYSEDHAPPENED << " time max was " << this->NTIME_DRAPEONLYSEDHAPPENEDMAX << std::endl;

  // timer reports
  // std::cout << std::endl << "--------------------- START OF TIME REPORT ---------------------" << std::endl;
  // for(int i=0; i< this->n_timers; i++)
  //   std::cout << CHRONO_name[i] << " took " << double(CHRONO_mean[i])/CHRONO_n_time[i] << " seconds out of " << CHRONO_n_time[i] << " runs" << std::endl;
  // std::cout << "--------------------- END OF TIME REPORT ---------------------" << std::endl << std::endl;
  // // Done
}


// Core function processing a single node
void ModelRunner::process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep)
{
  // Just a check: if the lake solver is not activated, I have no reason to reprocess node
  // if(this->lake_solver == false && is_processed[node])
  if(is_processed[node])
    return;

  // If I reach this stage, this node can be labelled as processed
  is_processed[node] = true;

  // Manages the fluxes before moving the particule: accumulating DA, precipitation, infiltration, evaporation, ...
  this->manage_fluxes_before_moving_prep(this->chonk_network[node],this->label_array[node] );

  // If the lake solving is activated, then I go through the (more than I expected) complex process of filing a lake correctly
  if(this->lake_solver) 
  {
    // 1) checking if my node has a lake ID  
    int lakeid = this->node_in_lake[node];
    // if not in a lake: not in a lake yo
    if(lakeid == -1)
    {
      // JUMPING TO LABEL NO LAKE
      // First time I use labels, it is not a really good practise but here is bloody convenient
      goto nolake;
    }

    // else -> lake solver
    this->lake_solver_v4(node);

    // and done if lake solver!
    return;
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
      this->chonk_network[node].external_moving_prep({next_node},{1.},{0},{0});

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
  if(this->chonk_network[node].get_sediment_flux() < -0.1)


  // Apply the changes and propagate the fluxes downstream
  //# timer
  CHRONO_start[4] = std::chrono::high_resolution_clock::now();
  // Function
  this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->surface_elevation_tp1, this->sed_height_tp1, this->timestep);
  // Timer
  CHRONO_stop[4] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[4] ++;
  CHRONO_mean[4] += std::chrono::duration<double>(CHRONO_stop[4] - CHRONO_start[4]).count();
  
}


// Alternative processing function for cases where I know for fact that I do not want lakes
void ModelRunner::process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move)
{

  if(this->is_processed[node])
    return;

  this->is_processed[node] = true;
  
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
  
  this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->surface_elevation_tp1, this->sed_height_tp1, this->timestep);
  // doen
}

// See above
void ModelRunner::process_node_nolake_for_sure(int node, std::vector<bool>& is_processed,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move, std::vector<int>& ignore_some)
{

    is_processed[node] = true;
    if(need_flux_before_move)
      this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);
    
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    
    this->chonk_network[node].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);
}


// Function finaliseing a timestep
void ModelRunner::finalise()
{

  double cellarea = this->dx * this->dy;

  // First dealing with lake deposition:
  this->drape_deposition_flux_to_chonks();

  // USEFUL debugging statement, not ready to delete it in case
  // for (int i = 0; i < this->graph.depression_tree.get_n_dep(); i++)
  // {
  //   if(this->graph.depression_tree.active[i] == true)
  //   {
  //     // std::cout.precision(12);
  //     std::cout << "Lake Evaporation:: " << this->graph.depression_tree.actual_amount_of_evaporation[i]/this->timestep << " vs " << std::fixed << this->graph.depression_tree.volume_water[i]/this->timestep << std::endl;
  //     std::cout << "N nodes = " << this->graph.depression_tree.get_all_nodes(i).size() << " elev pit " << this->surface_elevation[this->graph.depression_tree.pitnode[i]] << " elev externode " << this->surface_elevation[this->graph.depression_tree.externode[i]] << " elev tippingnode " << this->surface_elevation[this->graph.depression_tree.tippingnode[i]] << std::endl;
  //     std::cout << " pit " << this->graph.depression_tree.pitnode[i] << " externode " << this->graph.depression_tree.externode[i] << " tipping " << this->graph.depression_tree.tippingnode[i] << std::endl;
  //     this->Ql_out += this->graph.depression_tree.actual_amount_of_evaporation[i]/this->timestep;
  //   }
  // }


  // then actively finalising the deposition and other details
  // Iterating through all nodes
  for(int i=0; i< this->io_int["n_elements"]; i++)
  {

    double sedcrea = this->chonk_network[i].get_sediment_creation_flux() * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_bedrock() * cellarea * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_sediments() * cellarea * timestep;
    this->Qs_mass_balance += this->chonk_network[i].get_deposition_flux() * cellarea * timestep;

    if(this->active_nodes[i] == false)
      continue;

    // Getting the current chonk by address for readability
    chonk& tchonk = this->chonk_network[i];

    // getting the current composition of the sediment flux
    auto this_lab = tchonk.get_label_tracker();

    // NANINF DEBUG CHECKER
    for(auto LAB:this_lab)
      if(std::isfinite(LAB) == false)
        std::cout << LAB << " << naninf for sedflux" << std::endl;

    // First applying the bedrock-only erosion flux: decrease the overal surface elevation without affecting the sediment layer
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux_only_bedrock() * timestep;

    // Applying elevation changes from the sediments
    // Reminder: sediment creation flux is the absolute rate of removal/creation of sediments

    // NANINF DEBUG CHECKER II
    if(std::isfinite(sedcrea) == false)
    {
      sedcrea = 0;
      throw std::runtime_error("NAN sedcrea finalisation not possible yo");
    }

    // TEMP DEBUGGER TOO
    // AT TERM THIS SHOULD NOT HAPPEN???
    // if I end up with a negative sediment layer
    if(sedcrea + sed_height_tp1[i] < 0)
    {
      // IT STILL HAPPENS
      surface_elevation_tp1[i] -= sed_height_tp1[i];
      sed_height_tp1[i] = 0.;
      sed_prop_by_label[i] = std::vector<std::vector<double> >();
      is_there_sed_here[i] = false;
    } 
    else
    {
      // Calling the function managing the sediment layer composition tracking
      this->add_to_sediment_tracking(i, sedcrea, this_lab, sed_height_tp1[i]);

      // Applying the delta_h on both surface elevation and sediment layer
      surface_elevation_tp1[i] += sedcrea;
      sed_height_tp1[i] += sedcrea;
    }

    //Dealing now with "undifferentiated" Erosion rates
    double tadd = tchonk.get_erosion_flux_undifferentiated() * timestep;

    if(std::abs(tadd)>0)
    {
      this->add_to_sediment_tracking(i, -1*tadd, this_lab, sed_height_tp1[i]);
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
  // double save_Ql_out = this->Ql_out;
  // this->Ql_out = 0;
  for(int i=0; i<this->io_int["n_elements"]; i++)
  {
    this->Ql_out += (tlake_depth[i] - this->lake_depth[i]) * this->dx * this->dy / this->timestep;

    if(tlake_depth[i] > 0 && this->active_nodes[i] == 0)
      std::cout <<"HAYAAAAA" << std::endl;

    if(tlake_depth[i] > 0)
    {
      std::vector<int> neightbouring_nodes; std::vector<double> length2neigh;
      this->graph.get_D8_neighbors( i, this->active_nodes,  neightbouring_nodes,  length2neigh);
      for(auto tn:neightbouring_nodes)
      {
        if(this->topography[tn] < this->topography[i])
          std::cout << i << " HAYA22:: " << this->topography[i] << " vs " << this->topography[tn] << std::endl;
      }
    }

  }

  // Saving the new lake depth  
  this->lake_depth  = tlake_depth;




  // calculating other water mass balance.
  for(int i = 0; i<this->io_int["n_elements"]; i++)
  {
    if(this->active_nodes[i] == false)
    {
      this->Qw_out += this->chonk_network[i].get_water_flux();
      this->Qs_mass_balance += this->chonk_network[i].get_sediment_flux();
    }
  }
  // Finalisation done
}

void ModelRunner::lake_solver_v4(int node)
{

  // this depression is?
  int this_dep = this->node_in_lake[node];

  // Lines to uncomment to check the continuity of lakes (just transmitting through)
  // double toadd = this->chonk_network[node].get_water_flux() - this->graph.depression_tree.volume_max_with_evaporation[this_dep] / this->timestep;
  // this->Qw_out += this->graph.depression_tree.volume_max_with_evaporation[this_dep] / this->timestep;
  // this->chonk_network[this->graph.depression_tree.externode[this->graph.depression_tree.get_ultimate_parent(this_dep)]].add_to_water_flux(toadd);
  // return;

  // Getting the top depression of the local tree
  int master_dep = this->graph.depression_tree.get_ultimate_parent(this_dep);

  // Checking how many of the level 0 depressions have been oprocessed
  // This is to ensure that all nodes above the depression have been processed
  if( this->graph.depression_tree.n_0level_children_in_total_done[master_dep] <  this->graph.depression_tree.n_0level_children_in_total[master_dep])
  {
    // Adding to the count
    this->graph.depression_tree.n_0level_children_in_total_done[master_dep]++;
    
    // if not all dep level0 have been processed ---> not ready yet and pass
    if( this->graph.depression_tree.n_0level_children_in_total_done[master_dep] <  this->graph.depression_tree.n_0level_children_in_total[master_dep])
      return;
  }

  // If I reach here: all the depressions of the system have been proc and the lake solver is ready to roll
  // Initiating a bunch of container to store the amount outletting the depressions
  //# Does it outlets? 
  std::map<int,bool> outflows;
  //# how much water outlets?
  std::map<int,double> outwat;
  //# how much sed outlets?
  std::map<int,double> outsed;
  //# in which proportions of labels?
  std::map<int,std::vector<double> > outlab;
  //# Getting the treestack: lakes from bottom to top
  std::vector<int> treestack = this->graph.depression_tree.get_local_treestack(master_dep);

  // First iterations thourgh the lakes to initialise the containers to 0
  for(auto dep : treestack )
  {
    // Initialising the map to not-outflowing
    outflows[dep] = false;
    // and the quantities to 0;
    outwat[dep] = 0;
    outsed[dep] = 0;
    outlab[dep] = std::vector<double>();
  }

  // Second iterations: every level 0 depression gets their water from their chonks
  // Every depression with a parent propagates their water up
  for(auto dep : treestack )
  {
    // Parent ID (-1 if no parent)
    int parent = this->graph.depression_tree.parentree[dep];

    // If the level is 0 -> getting the total volume of water/sed from this timestep
    if(this->graph.depression_tree.level[dep] == 0)
    {
      this->graph.depression_tree.add_water(this->chonk_network[this->graph.depression_tree.pitnode[dep]].get_water_flux() * this->timestep, dep);
      this->graph.depression_tree.add_sediment(this->chonk_network[this->graph.depression_tree.pitnode[dep]].get_sediment_flux(), this->chonk_network[this->graph.depression_tree.pitnode[dep]].get_label_tracker(), dep);
      // Gathering debugging info to check wether the original water/sed input is modified them
      this->ORIGINALPIT2WAT[this->graph.depression_tree.pitnode[dep]] = this->chonk_network[this->graph.depression_tree.pitnode[dep]].get_water_flux();
      this->ORIGINALPIT2SED[this->graph.depression_tree.pitnode[dep]] = this->chonk_network[this->graph.depression_tree.pitnode[dep]].get_sediment_flux()/this->timestep;
    }

    // If there is a parent -> fire up
    if(parent > -1)
    {
      this->graph.depression_tree.transmit_up(dep);
    }

    // if the depression is full -> labelled as outflowing
    if(this->graph.depression_tree.is_full(dep))
      outflows[dep] = true;

    // gets the outletting quantities in place
    this->correct_extras(dep,outwat[dep],outsed[dep],outlab[dep]);

    // If < 0 --> 0 (not outflowing)
    int cpt = 0;
    if(outsed[dep]==0)
    {
      cpt++;
    }
    if(outwat[dep]==0)
    {
      cpt++;
    }
    if(cpt == 2)
      outflows[dep] = false;

    // Done with 2nd iteration
  }

  // Now going through the stack from the higher level to the lower one
  // The idea is that now all of the deps have their Q content, we stop at the first fillable parents and skip all the children
  for(int i = int(treestack.size())-1; i >= 0; i--)
  {

    // Getting dep ID
    int dep = treestack[i];

    // If it is already processed -> Stop ehre
    if(this->graph.depression_tree.processed[dep])
      continue;

    int twin = this->graph.depression_tree.get_twin(dep);

    // If there is a twin, I am deciding which one should be considered as main depression
    if(twin > -1)
    {
      // I am simply considering the one the most likely to outflow (i.e with the highest ratio between volume to fill and volume of stuff)   
      double ratio_of_filling_dep = std::max(this->graph.depression_tree.volume_water[dep]/this->graph.depression_tree.volume_max_with_evaporation[dep],this->graph.depression_tree.volume_sed[dep]/this->graph.depression_tree.volume[dep]);
      double ratio_of_filling_twin = std::max(this->graph.depression_tree.volume_water[twin]/this->graph.depression_tree.volume_max_with_evaporation[twin],this->graph.depression_tree.volume_sed[twin]/this->graph.depression_tree.volume[twin]);

      if(ratio_of_filling_twin > ratio_of_filling_dep)
      {
        int intermediaire = dep;
        dep = twin;
        twin = intermediaire;
      }    
    }

  
    // Otherwise label as processed
    this->graph.depression_tree.processed[dep] = true;

    // Calculating the minimum amount of volume necessary to fill the depression
    double lowerboundvolwat = this->graph.depression_tree.get_volume_of_children_water(dep);
    double lowerboundvolsed = this->graph.depression_tree.get_volume_of_children_sed(dep);
    
    // POTENTIAL IMPROVEMENT HERE: IN CASE LAKE EVAPORATION IS ON AND THERE ARE MORE SEDIMENT INPUT THAN WATER IN THE LAKE, THE VOLUME MIGHT BE OVERESTIMATED

    // If I have enough water to fill the depression:
    if(this->graph.depression_tree.volume_water[dep] > lowerboundvolwat || this->graph.depression_tree.volume_sed[dep] > lowerboundvolsed )
    {

      // I am processing the depression: filling it with the water/sed I have in it and outletting the relevant fluxes. 
      // Note that if also recalculate the outlets fluxes function of the cancelling of lake sediments, outlet water rerouting and stuff like that
      this->process_dep(dep, outsed[dep], outlab[dep], outflows[dep], outwat[dep]);
      // this->correct_extras(dep,outwat[dep],outsed[dep],outlab[dep]);  
      // I now have a new state for outsed[dep], outlab[dep], outflows[dep], outwat[dep]
      // I need to check whether I have a twin here. If I do, I need to transmit it my outletting stuff
      if(twin > -1)
      {

        // Transmitting to an arbitrary path to level 0
        int chd = twin;
        while(chd != -1)
        {
          this->graph.depression_tree.add_water(outwat[dep], chd);
          this->graph.depression_tree.add_sediment(outsed[dep], outlab[dep], chd);

          if(this->graph.depression_tree.is_full(chd))
            outflows[chd] = true;

          this->correct_extras(chd,outwat[chd],outsed[chd],outlab[chd]);

          // If < 0 --> 0 (not outflowing)
          int cpt = 0;
          if(outsed[chd]==0)
          {
            cpt++;
          }
          if(outwat[chd]==0)
          {
            cpt++;
          }
          if(cpt == 2)
            outflows[chd] = false;
          
          chd = this->graph.depression_tree.treeceivers[chd][0];
        }
      }

      std::vector<int> children = this->graph.depression_tree.get_all_children(dep,true);

      for(auto chd:children)
      {
        this->graph.depression_tree.processed[chd] = true;
      }
    }

    if(twin > -1)
    {
      // if there is a twin, I am also processing it!
      double lowerboundvoltwin_wat = this->graph.depression_tree.get_volume_of_children_water(twin);
      double lowerboundvoltwin_sed = this->graph.depression_tree.get_volume_of_children_sed(twin);
      this->graph.depression_tree.processed[twin] = true;

      if(this->graph.depression_tree.volume_water[twin] >= lowerboundvoltwin_wat || this->graph.depression_tree.volume_sed[twin] >= lowerboundvoltwin_sed)
      {
        
        this->process_dep(twin, outsed[twin], outlab[twin], outflows[twin], outwat[twin]);
        this->graph.depression_tree.processed[twin] = true;
        std::vector<int> children = this->graph.depression_tree.get_all_children(twin,true);
        for(auto chd:children)
        {
          this->graph.depression_tree.processed[chd] = true;
        }
        // std::cout << std::endl;

        if(outwat[twin]>0)
        {
          std::cout << "Twin outlets? hmmm" << std::endl;;
          // throw std::runtime_error("Twin outlets? hmmm");
        }
      }
    }
  }
}

// Short function that calculates the outletting fluxes based on the input vs volume of the dep
void ModelRunner::correct_extras(int dep, double& extra_wat, double& extra_sed, std::vector<double>& extra_lab)
{
  extra_wat = std::max(this->graph.depression_tree.volume_water[dep] - this->graph.depression_tree.volume_max_with_evaporation[dep], 0.);
  extra_sed = std::max(this->graph.depression_tree.volume_sed[dep] - this->graph.depression_tree.volume[dep], 0.);
  extra_lab = this->graph.depression_tree.label_prop[dep];
}

// Main function processing a single depression
void ModelRunner::process_dep(int dep, double& extra_sed, std::vector<double>& extra_lab, bool does_outlet, double& extra_wat)
{

  // std::cout << std::endl ;
  // std::cout << "Actually processing " << dep << " Master dep is " << this->graph.depression_tree.get_ultimate_parent(dep) << std::endl;
  // std::cout << "pit is " << this->graph.depression_tree.pitnode[dep] << std::endl;
  // std::cout << "outlet is " << this->graph.depression_tree.tippingnode[dep] << std::endl;
  // std::cout << "externode is " << this->graph.depression_tree.externode[dep] << std::endl;
  // std::cout  << " which is processed?? " << this->is_processed[this->graph.depression_tree.externode[dep]] << std::endl;
  // std::cout << "Vw at start is " << this->graph.depression_tree.volume_water[dep] << std::endl;

  // Mark the depression as active -> will count in the mass balance calculations and the inherited water/topo
  this->graph.depression_tree.active[dep] = true;

  this->correct_extras(dep, extra_wat, extra_sed, extra_lab);

  // Check whether is does outlet or is not filled to the top (the algorithm are radically different)
  if(extra_wat > 0 || extra_sed > 0)
  {

    //The lake outflows-> filling it will be easy, reprocessing the downstream lakes/nodes won't.
    // First I need to fill the water to the top:
    // Put the topo to the max water height of the lake, label all nodes as being part of the lake, ..
    if(extra_wat > 0)
    {
      this->fill_lake_to_top(dep);
    }
    else
    {
      this->fill_underfilled_lake(dep);
    }


    // Important step: defluvialisation
    // actiually the name is not optimal I should rethink it
    // It cancels all the erosion/depostion made by other processes in the lake
    // It back-calculates this->graph.depression_tree.volume_sed and the extra sed and lab in place
    this->defluvialise_lake(dep, extra_sed, extra_lab);

    // Again recorrecting the extra post defluv
    this->correct_extras(dep,extra_wat, extra_sed, extra_lab);


    // Geting the outlet node ID
    int outlet = this->graph.depression_tree.tippingnode[dep];

    if(this->active_nodes[outlet] == 0)
      std::cout << "OUTLET IS 0" << std::endl;

    // getting the local sediment flux (ie what has been locally eroded/deposited)
    double locsedflux = this->chonk_network[outlet].get_local_sedflux(this->timestep, this->cellarea);
    double globasedflux = this->chonk_network[outlet].get_sediment_flux();


    // In some rare cases outlet is not processed (deprecated I believe, it was because of flat surfaces, I leave it there for legacy)
    if(this->is_processed[outlet] == false)
    {
      this->process_node_nolake_for_sure(outlet, this->is_processed, this->active_nodes, this->cellarea, this->topography, true, true);
    }
    
    if(extra_wat > 0)
    {  
      // Getting ready to back-calcualte water fluxes and sediment fluxes from the outlet:
      // going through the receivers of the outet
      // If the receiver is in the lake -> I remove the amount of sediment that have been given to the lake
      // if out of the lake, I ssave the amount of water this node use to give to the outlets and back-calculate it
      std::vector<int> rec; std::vector<double> wwf; std::vector<double> wws; std::vector<double> strec;
      this->chonk_network[outlet].copy_moving_prep(rec,wwf,wws,strec);
      double sed2remove = 0;
      // double sed2remove_only_outlet = 0;
      double water2add = 0;
      for(size_t i = 0; i < rec.size(); i++)
      {
        if(this->node_in_lake[rec[i]] == dep)
        {
          sed2remove -= wws[i] * globasedflux;
        }
        else if(this->is_processed[rec[i]] == false)
        {
          water2add += wwf[i] * this->chonk_network[outlet].get_water_flux() * this->timestep;
          // sed2remove_only_outlet -= wws[i] * locsedflux;
        }
      }
      
      // Putting the outlet reprocessing info into the extrasedwatlab stuff
      this->graph.depression_tree.add_sediment(sed2remove, this->chonk_network[outlet].get_label_tracker(), dep);
      double wat2add2outletonly = water2add;

      // this reputs in the node the amount of sediment that were given to the outlet by donors (not in lake)
      std::vector<double> sed2add2outletonly_lab = this->chonk_network[outlet].get_label_tracker();// mix_two_proportions(extra_sed, extra_lab, this->chonk_network[outlet].get_sediment_flux() - locsedflux, this->chonk_network[outlet].get_label_tracker());
      double sed2add2outletonly = this->chonk_network[outlet].get_sediment_flux() - locsedflux;
      this->graph.depression_tree.add_sediment(sed2add2outletonly, this->chonk_network[outlet].get_label_tracker(), dep);

      this->correct_extras(dep, extra_wat, extra_sed, extra_lab);


      // now only I can correct the rest
      extra_wat += wat2add2outletonly;

      if(extra_sed < 0)
        extra_sed = 0;

      // Ready to reproc the outlet:
      // #1 cancel what it use to give to its receivers
      this->chonk_network[outlet].cancel_split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->timestep);
      // #2 reset the receiver
      this->chonk_network[outlet].reset();
      // #3 Force it to give water to the external node and not back to lake
      this->chonk_network[outlet].external_moving_prep({this->graph.depression_tree.externode[dep]},
       {1.}, {1.}, {(this->topography[outlet] - this->topography[this->graph.depression_tree.externode[dep]])/this->dx});
      // #4 Relabel the node as not-processed
      this->is_processed[outlet] = false;
      // #5 manually set the sed/water fluxes to what has been calculated above 
      this->chonk_network[outlet].set_sediment_flux(extra_sed, extra_lab, 1.);
      this->chonk_network[outlet].set_water_flux(extra_wat/this->timestep);
      this->graph.depression_tree.volume_water_outlet[dep] = extra_wat;
      this->graph.depression_tree.volume_sed_outlet[dep] = extra_sed;
      // #6 and finally reprocess the node (no move prep as it is forced, no preflux as already included in the outlet calculation)
      this->process_node_nolake_for_sure(outlet, this->is_processed, this->active_nodes, this->cellarea, this->topography, false, false);
    }
    else if (extra_sed > 0)
    {
      this->chonk_network[this->graph.depression_tree.externode[dep]].add_to_sediment_flux(extra_sed, extra_lab, 1.);
      this->graph.depression_tree.volume_sed_outlet[dep] = extra_sed;
    }

  }
  else
  {


    if(extra_sed < 0)
      extra_sed = 0;

    // much "simpler" scenario:
    // Just needs to fill the lake
    this->fill_underfilled_lake(dep);
    this->defluvialise_lake(dep, extra_sed, extra_lab);
    this->correct_extras(dep,extra_wat, extra_sed, extra_lab);
  }


  // now ther tot amount in lakes needs correction and becomes the sum of extra vs this->graph.depression...
  // As my lake is full, my actual amount of volume_water is the max I can accomodate. The rest being stored into the out container
  if(this->graph.depression_tree.volume_water[dep] > this->graph.depression_tree.volume_max_with_evaporation[dep])
    this->graph.depression_tree.volume_water[dep] = this->graph.depression_tree.volume_max_with_evaporation[dep];

  // Same for the sed
  if(this->graph.depression_tree.volume_sed[dep] > this->graph.depression_tree.volume[dep])
    this->graph.depression_tree.volume_sed[dep] = this->graph.depression_tree.volume[dep];
}


// Function to fill lake to top
void ModelRunner::fill_lake_to_top(int dep)
{
  
  // the water height of this lake is max, as the lake is full
  this->graph.depression_tree.hw[dep] = this->graph.depression_tree.hw_max[dep];

  // getting the outlet or "tipping node"
  int outlet = this->graph.depression_tree.tippingnode[dep];

  // if(this->graph.depression_tree.get_all_nodes(dep).size() == 1)
  // {
  //   std::cout << "Trapes" << std::endl;
  //   int node=this->graph.depression_tree.get_all_nodes(dep)[0];
  //   std::vector<int> stuff; std::vector<double> neight; this->graph.get_D8_neighbors(node,this->active_nodes, stuff, neight);
  //   std::cout << "Node + hw -> " << this->graph.depression_tree.hw[dep] << std::endl;
  //   std::cout << "Others: ";
  //   for(auto n:stuff)
  //   {
  //     std::cout << "|" <<this->topography[n];
  //     if(this->topography[n] <this->graph.depression_tree.hw[dep] )
  //       std::cout  << std::endl << "IOASDFJSDJFLSDJFLKJSDKLFJKLSDJFKLJSDLKFJLKSJDFLKJASKLD" << std::endl;
  //   }
  //   std::cout << std::endl;
  // }

  //going through each nodes of the lake to label them and adjust the topography
  for(auto n:this->graph.depression_tree.get_all_nodes(dep))
  {
    // Technically, the outlet is not in the lake, jsut a special river
    if(this->lake_evaporation)
    {
      this->graph.depression_tree.actual_amount_of_evaporation[dep] += this->lake_evaporation_rate_spatial[n] * this->cellarea * this->timestep;
    }

    if(n == outlet)
    {
      continue;
    }

    // node in dat lake yo
    this->node_in_lake[n] = dep;
    // topo to hw
    this->topography[n] = this->graph.depression_tree.hw[dep];
    // and if lake evaporation, calculating the local amount
  }
  // Done, it was mot too hard


}

void ModelRunner::fill_underfilled_lake(int dep)
{

  // Getting the outlet of the lake
  int outlet = this->graph.depression_tree.tippingnode[dep];

  // gathering nodes and sorting them by elevation for the partial filling
  auto tnodes = this->graph.depression_tree.get_all_nodes(dep);
  // -> Data structure for the sorting
  std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > Sorter;
  // -> Actual sorting
  for(auto tn: tnodes)
  {
      Sorter.emplace(PQ_helper<int,double>(tn, this->surface_elevation[tn]));
  }

  // And filling the depression
  // -> number of nodes currently in
  int n_nodes = 0;
  // -> nodes to raise to the HW
  std::vector<int> nodes2topogy;
  nodes2topogy.reserve(tnodes.size());
  // -> remaining volume of water to add into the thingy
  double remaining_volume = this->graph.depression_tree.volume_water[dep];
  // -> will cumulate the amount stored in lake
  double cumul_V = 0;
  while(Sorter.size() > 0)
  {
    // Getting node and elev
    int this_node = Sorter.top().node;
    double this_elev = Sorter.top().score;
    // Poping the node from the thingy
    Sorter.pop();

    // Getting node and elev of the next node in line, or the outlet if we are at the last node, ie if the PQ is empty
    int top_node;
    double top_elev;
    if(Sorter.empty())
    {
      top_node = this->graph.depression_tree.tippingnode[dep];
      top_elev = this->surface_elevation[top_node];
    }
    else
    {
      top_node = Sorter.top().node;
      top_elev = Sorter.top().score;
    }

    // Incrementing the counter of nodes
    n_nodes++;

    // while not the outlet, registering it
    if(this_node != outlet)
    {
      nodes2topogy.emplace_back(this_node);
      this->node_in_lake[this_node] = dep;
    }

    // Delaing with local lake evaporation if any of course
    double lcoal_evaporation = 0;
    if(this->lake_evaporation)
    {
      lcoal_evaporation = this->lake_evaporation_rate_spatial[this_node] * this->cellarea * this->timestep;
    }

    // removing the evaporation from the volume
    remaining_volume -= lcoal_evaporation;
    if(remaining_volume < 0)
    {
      lcoal_evaporation -= abs(remaining_volume);
      remaining_volume = 0;
    }
    this->graph.depression_tree.actual_amount_of_evaporation[dep] += lcoal_evaporation;

    // Calculating the incrementing volume
    // -> dz
    double deltelev = top_elev - this_elev;
    // -> local Volume = n nodes to be raised to the next elev
    double dV = n_nodes * this->cellarea * deltelev;



    if(remaining_volume > dV)
    {
      remaining_volume -= dV;
      cumul_V += dV;
    }
    else
    {
      double ratio = 0; 
      if(dV > 0)
        ratio = remaining_volume / dV;

      // At the moment the lake evaporation is approximated at a pixel pret:
      // If I fall in between 2 nodes, I assume it stick at the top of the last node and backcalculate the evaporation as "in-between"
      // this could eventually be rethought a bit more accurately, although it is a detail so we'll leave it to a next publication refining the lake evaporation method

      if(ratio < 0)
      {
        // std::cout << "Negative ratio" << std::endl;
        ratio = 0;
        this->graph.depression_tree.actual_amount_of_evaporation[dep] += remaining_volume;
      }

      this->graph.depression_tree.hw[dep] = this_elev + ratio * deltelev;
      
      // if(nodes2topogy.size() == 1)
      // {
      //   std::cout << "Tripes" << std::endl;
      // }

      for(auto nj:nodes2topogy)
        this->topography[nj] = this->graph.depression_tree.hw[dep];

      break;
    }
  }
}

void ModelRunner::defluvialise_lake(int dep, double& extra_sed, std::vector<double>& extra_sed_prop)
{
  int outlet = this->graph.depression_tree.tippingnode[dep];
  
  double defluvialisation_of_sed = 0;
  std::vector<double> defluvialisation_of_sed_label_edition(this->n_labels,0.);

  double original_extra_sed = extra_sed;
  std::vector<double> original_extra_sed_lab = extra_sed_prop;

  double sum_erV = 0;
  double sum_DV = 0;
  for (auto n : this->graph.depression_tree.get_all_nodes(dep))
  {
    // std::cout << n << "-" << this->graph.depression_tree.node2tree[n] << "-";
    if(this->node_in_lake[n] == dep)
    {
      if(n ==  this->graph.depression_tree.tippingnode[dep])
        continue;

      // HERE I WILL NEED TO REMOVE THE SED FROM EROSION?DEP WITH THE RIGHT PROPORTIONS
      double tsed = 0;

      // REMOVING EROSION OF BEDROCK
      tsed = this->chonk_network[n].get_erosion_flux_only_bedrock() * this->timestep * this->cellarea;
      std::vector<double> temp(this->n_labels,0.); temp[this->label_array[n]] = 1.;
      defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
      defluvialisation_of_sed += tsed;
      sum_erV += tsed;

      tsed = this->chonk_network[n].get_erosion_flux_only_sediments() * this->timestep * this->cellarea;
      if(this->sed_prop_by_label[n].size() > 0)
        temp = this->sed_prop_by_label[n][this->sed_prop_by_label[n].size() - 1];
      else
        temp = this->chonk_network[n].get_label_tracker();

      defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
      defluvialisation_of_sed += tsed;
      sum_erV += tsed;
      
      // REMOVING DEPOSITION
      tsed = -1 * this->chonk_network[n].get_deposition_flux() * this->timestep * this->cellarea;
      temp = this->chonk_network[n].get_label_tracker();
      defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
      defluvialisation_of_sed += tsed;
      sum_DV -=  tsed;

      // And finally resetting the sediment flux of the chonk
      this->chonk_network[n].reset_sed_fluxes();

    }
  }
  this->graph.depression_tree.add_sediment(-1 * defluvialisation_of_sed, extra_sed_prop, dep);
  this->graph.depression_tree.volume_sed_defluvialised[dep] = defluvialisation_of_sed;

}


void ModelRunner::DEBUG_write_lakes_to_file(std::string filename)
{

  std::ofstream out(filename);
  out << "dep_ID,active,level,nnodes,parent,twin,child0,child1,pitnode,tippingnode,externode,V,Vwmax,Vw,Vs,Vw_outlet,Vs_outlet,evaporation,defluvialisation,fpitwat,opitwat,totowat,totosed" << std::endl;
  for( int i = 0; i < this->graph.depression_tree.get_n_dep(); i++)
  {
    double totowat = 0;
    double totosed = 0;
    for(auto chd:this->graph.depression_tree.get_all_children(i,true))
    {
      if(this->graph.depression_tree.level[chd] == 0)
      {
        totowat += this->ORIGINALPIT2WAT[this->graph.depression_tree.pitnode[chd]];
        totosed += this->ORIGINALPIT2SED[this->graph.depression_tree.pitnode[chd]];
      }
    }
    out << i << ",";
    out << this->graph.depression_tree.active[i] << ",";
    out << this->graph.depression_tree.level[i] << ",";
    out << this->graph.depression_tree.get_all_nodes(i).size() << ",";
    out << this->graph.depression_tree.parentree[i] << ",";
    out << this->graph.depression_tree.get_twin(i) << ",";
    out << this->graph.depression_tree.treeceivers[i][0] << ",";
    out << this->graph.depression_tree.treeceivers[i][1] << ",";
    out << this->graph.depression_tree.pitnode[i] << ",";
    out << this->graph.depression_tree.tippingnode[i] << ",";
    out << this->graph.depression_tree.externode[i] << ",";
    out << this->graph.depression_tree.volume[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_max_with_evaporation[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_water[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_sed[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_water_outlet[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_sed_outlet[i]/this->timestep << ",";
    out << this->graph.depression_tree.actual_amount_of_evaporation[i]/this->timestep << ",";
    out << this->graph.depression_tree.volume_sed_defluvialised[i]/this->timestep << ",";
    out << this->chonk_network[this->graph.depression_tree.pitnode[i]].get_water_flux() << ",";
    out << ORIGINALPIT2WAT[this->graph.depression_tree.pitnode[i]]<< ",";
    out << totowat << ",";
    out << totosed;
    
    out << std::endl;

  }
  out.close();
}





// this Fucntion add a height of sediment to a preexisting pile. It updates the tracking of deposited sediments
void ModelRunner::add_to_sediment_tracking(int index, double height, std::vector<double> label_prop, double sed_depth_here)
{
  // No height to add -> nothing happens yo
  if(height == 0)
    return;

  double depth_res = depths_res_sed_proportions;

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
      sed_prop_by_label[index].emplace_back(std::vector<double>(label_prop.size(), 0.) );
      current_box = 0;
    }

    if(this_hbox > 1)
    {
      // std::cout << "I" << std::endl;;
      this->sed_prop_by_label[index][current_box] = mix_two_proportions(prop1,sed_prop_by_label[index][current_box], 1 - prop1, label_prop);

      for(int i = 0; i < int(boxes_ta_filled) + 1; i++)
      {
        sed_prop_by_label[index].emplace_back(label_prop);
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
        sed_prop_by_label[index].emplace_back(label_prop);
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
  NTIMEPREFLUXCALLED++;
  
  if(this->precipitations_enabled == false)
  {
    this_chonk.inplace_only_drainage_area(this->dx, this->dy);
    this->Qw_in += this->dx* this->dy;
  }
  else
  {
    this->Qw_in += this_chonk.inplace_precipitation_discharge(this->dx, this->dy, this->precipitations);
  }

  CHRONO_stop[2] = std::chrono::high_resolution_clock::now();
  CHRONO_n_time[2] ++;
  CHRONO_mean[2] += std::chrono::duration<double>(CHRONO_stop[2] - CHRONO_start[2]).count();
  return;
}

void ModelRunner::cancel_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{
  if(this->precipitations_enabled == false)
  {
    this_chonk.cancel_inplace_only_drainage_area(this->dx, this->dy);
    this->Qw_in -= this->dx* this->dy;
  }
  else
  {
    this->Qw_in -= this_chonk.cancel_inplace_precipitation_discharge(this->dx, this->dy, this->precipitations);
  }
  
  this->NTIMEPREFLUXCALLED -= 1;

  return;

}



void ModelRunner::manage_move_prep(chonk& this_chonk)
{

  CHRONO_start[5] = std::chrono::high_resolution_clock::now();
  this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->thresholdMF2SF, this->timestep,  this->topography, 
        this->dx, this->dy, chonk_network);

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


  // This is where the connectivity happens
  double fluvprop = this_chonk.get_fluvialprop_sedflux();
  fluvprop += this->hillslope2fluvial_connectivity;
  if(fluvprop>1)
    fluvprop = 1;
  this_chonk.set_fluvialprop_sedflux(fluvprop);


  //
  double this_Kr;
  double this_Ks;
  double this_kappar;
  double this_kappas;
  double S_c;
  this->manage_K_kappa(label_id, this_chonk, this_Kr, this_Ks, this_kappar, this_kappas, S_c);

  if(this->CHARLIE_I)
  {

    this_chonk.charlie_I(this->labelz_list[label_id].n, this->labelz_list[label_id].m, this_Kr, this_Ks,
    this->labelz_list[label_id].dimless_roughness, this->sed_height[index], 
    this->labelz_list[label_id].V, this->labelz_list[label_id].dstar, this->labelz_list[label_id].threshold_incision, 
    this->labelz_list[label_id].threshold_entrainment,label_id, these_sed_props, this->timestep,  this->dx, this->dy);
  }

  // Hillslope routine
  if(this->CIDRE_HS)
  {

    this_chonk.CidreHillslopes(this->sed_height[index], this_kappas, 
            this_kappar, S_c,
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

// Calculates the local fluvial and hillslope coefficiant
void ModelRunner::manage_K_kappa(int label_id, chonk& this_chonk, double& K_r, double& K_s, double& kappa_r, double& kappa_s, double& S_c)
{

  // K_ref =  Sqs / Sbed * (1-Qs/Qc) * k_process * K_efficiency


  // First calculating the values depending on local conditions
  K_r = this->labelz_list[label_id].Kr_modifyer * this->labelz_list[label_id].base_K;
  K_s = this->labelz_list[label_id].Ks_modifyer * this->labelz_list[label_id].base_K;

  kappa_s = this->labelz_list[label_id].kappa_s_mod * this->labelz_list[label_id].kappa_base;
  kappa_r = this->labelz_list[label_id].kappa_r_mod * this->labelz_list[label_id].kappa_base;

  S_c = this->labelz_list[label_id].critical_slope;

  // Applying the tool effect
  if(this->tool_effect_rock)
  {  
    int i = -1;
    double tool_k = 0;
    bool only_0 =true;
    for( auto val: this_chonk.get_label_tracker())
    {
      i++;
      if(val > 0)
      {
        only_0 = false;
        tool_k += val * std::pow((this->labelz_list[label_id].Kr_modifyer/this->labelz_list[i].Kr_modifyer), this->labelz_list[label_id].sensitivity_tool_effect);
      }
    }
    if(only_0 == false)
    {
      K_r = K_r *  tool_k;
      if(this->tool_effect_sed)
        K_s = K_s * tool_k;
    }
  }

  this->calculated_K[this_chonk.get_current_location()] = K_r;




}

// Initialise ad-hoc set of internal correspondance between process names and integer
// Again this is a small otpimisation that reduce the need to initialise and call maps for each nodes as maps are slower to access
// DEPRECATED
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
          labelz_list_double["SPIL_m"].emplace_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].emplace_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["SPIL_K"].emplace_back(tlab.double_attributes["SPIL_K"]);
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
          labelz_list_double["SPIL_m"].emplace_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].emplace_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["CHARLIE_I_Kr"].emplace_back(tlab.double_attributes["CHARLIE_I_Kr"]);
          labelz_list_double["CHARLIE_I_Ks"].emplace_back(tlab.double_attributes["CHARLIE_I_Ks"]);
          labelz_list_double["CHARLIE_I_V"].emplace_back(tlab.double_attributes["CHARLIE_I_V"]);
          labelz_list_double["CHARLIE_I_dimless_roughness"].emplace_back(tlab.double_attributes["CHARLIE_I_dimless_roughness"]);
          labelz_list_double["CHARLIE_I_dstar"].emplace_back(tlab.double_attributes["CHARLIE_I_dstar"]);
          labelz_list_double["CHARLIE_I_threshold_incision"].emplace_back(tlab.double_attributes["CHARLIE_I_threshold_incision"]);
          labelz_list_double["CHARLIE_I_threshold_entrainment"].emplace_back(tlab.double_attributes["CHARLIE_I_threshold_entrainment"]);
        }
        break;

      case 9 :
        labelz_list_double["Cidre_HS_kappa_s"] = std::vector<double>();
        labelz_list_double["Cidre_HS_critical_slope"] = std::vector<double>();
        for (auto& tlab:labelz_list)
        {
          labelz_list_double["Cidre_HS_kappa_s"].emplace_back(tlab.double_attributes["Cidre_HS_kappa_s"]);
          labelz_list_double["Cidre_HS_critical_slope"].emplace_back(tlab.double_attributes["Cidre_HS_critical_slope"]); 
        }       
        break;
      case 10 :
        labelz_list_double["Cidre_HS_kappa_s"] = std::vector<double>();
        labelz_list_double["Cidre_HS_kappa_r"] = std::vector<double>();
        labelz_list_double["Cidre_HS_critical_slope"] = std::vector<double>();
        for (auto& tlab:labelz_list)
        {
          labelz_list_double["Cidre_HS_kappa_s"].emplace_back(tlab.double_attributes["Cidre_HS_kappa_s"]);
          labelz_list_double["Cidre_HS_kappa_r"].emplace_back(tlab.double_attributes["Cidre_HS_kappa_r"]);
          labelz_list_double["Cidre_HS_critical_slope"].emplace_back(tlab.double_attributes["Cidre_HS_critical_slope"]);
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
          labelz_list_double["SPIL_m"].emplace_back(tlab.double_attributes["SPIL_m"]);
          labelz_list_double["SPIL_n"].emplace_back(tlab.double_attributes["SPIL_n"]);
          labelz_list_double["CHARLIE_I_Kr"].emplace_back(tlab.double_attributes["CHARLIE_I_Kr"]);
          labelz_list_double["CHARLIE_I_Ks"].emplace_back(tlab.double_attributes["CHARLIE_I_Ks"]);
          labelz_list_double["CHARLIE_I_V"].emplace_back(tlab.double_attributes["CHARLIE_I_V"]);
          labelz_list_double["CHARLIE_I_dimless_roughness"].emplace_back(tlab.double_attributes["CHARLIE_I_dimless_roughness"]);
          labelz_list_double["CHARLIE_I_dstar"].emplace_back(tlab.double_attributes["CHARLIE_I_dstar"]);
          labelz_list_double["CHARLIE_I_threshold_incision"].emplace_back(tlab.double_attributes["CHARLIE_I_threshold_incision"]);
          labelz_list_double["CHARLIE_I_threshold_entrainment"].emplace_back(tlab.double_attributes["CHARLIE_I_threshold_entrainment"]);
          labelz_list_double["CHARLIE_I_Krmodifyer"].emplace_back(tlab.double_attributes["CHARLIE_I_Krmodifyer"]);
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
  this->tot_inherited_water = 0;
  this->inherited_water_added = std::vector<double>(this->io_int["n_elements"],0.);
  for(int tlake = 0; tlake < this->graph.depression_tree.get_n_dep(); tlake++)
  {
    if(this->graph.depression_tree.active[tlake] == false)
      continue;

    // getting node underwater
    std::vector<int> unodes = this->graph.depression_tree.get_all_nodes(tlake);


    double minelev = std::numeric_limits<double>::max();
    int minnodor = -9999;
    double sumwat = 0;
    for( auto node:unodes)
    {
      if(this->surface_elevation[node] < minelev)
      {
        minnodor = node;
        minelev = this->surface_elevation[node];
      }
    }
    this->chonk_network[minnodor].add_to_water_flux((this->graph.depression_tree.volume_water[tlake] - this->graph.depression_tree.actual_amount_of_evaporation[tlake])/this->timestep);
    this->inherited_water_added[minnodor] += (this->graph.depression_tree.volume_water[tlake] - this->graph.depression_tree.actual_amount_of_evaporation[tlake])/this->timestep;
    this->tot_inherited_water += (this->graph.depression_tree.volume_water[tlake] - this->graph.depression_tree.actual_amount_of_evaporation[tlake])/this->timestep;
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

xt::pytensor<double,1> ModelRunner::get_deposition_flux()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_deposition_flux() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_fluvial_Qs()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_fluvial_Qs() ;
  }
  return output;

}

xt::pytensor<double,1> ModelRunner::get_hillslope_Qs()
{
  xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  for(auto& tchonk:chonk_network)
  {
    output[tchonk.get_current_location()] = tchonk.get_hillslope_Qs();
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
  std::cout << "ModelRunner::get_other_attribute is deprecated" << std::endl;
  // xt::pytensor<double,1> output = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  // for(auto& tchonk:chonk_network)
  // {
  //   output[tchonk.get_current_location()] = tchonk.get_other_attribute(key);
  // }
  // return output;
}

std::vector<xt::pytensor<double,1> > ModelRunner::get_label_tracking_results()
{
  std::vector<xt::pytensor<double,1> > output;
  for(int i=0; i<this->n_labels; i++)
  {
    xt::pytensor<double,1> temp = xt::zeros_like(this->surface_elevation);
    output.emplace_back(temp);
  }

  for( int i=0; i< io_int["n_elements"];i++)
  {
    chonk& tchonk =this->chonk_network[i];
    for(int j=0; j<this->n_labels; j++)
    {
      output[j][i] = tchonk.get_label_tracker()[j];
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

  double DEBUG_totsed = 0;
  for(int i = 0; i < this->graph.depression_tree.get_n_dep(); i++)
  { 
    // auto& loch = this->graph.depression_tree[i];
    // Checking if this is a main laked
    if(this->graph.depression_tree.active[i] == false)
      continue;

    this->Ql_out += this->graph.depression_tree.actual_amount_of_evaporation[i]/this->timestep;

    if(this->graph.depression_tree.volume_water[i] < 0)
      throw std::runtime_error("NEGWATLAKE");

    if(this->graph.depression_tree.volume_sed[i] < 0)
      throw std::runtime_error("NEGsedLAKE");

    auto dat = this->graph.depression_tree.get_all_nodes(i);

    if(dat.size() < 5 )
    {
      auto thw = this->graph.depression_tree.hw[i];
      double lmax = 0;
      for( auto n:dat)
      {
        if(thw - this->topography[n] > lmax)
        {
          lmax = thw - this->topography[n];
        }
      }
      if(lmax > 100)
        throw std::runtime_error("wtf lake");
    }

    double totsed = 0;
    double totwat = 0;
    double ratio_of_dep = this->graph.depression_tree.volume_sed[i]/(this->graph.depression_tree.volume_water[i] - this->graph.depression_tree.actual_amount_of_evaporation[i]);
    if(this->graph.depression_tree.volume_water[i] - this->graph.depression_tree.actual_amount_of_evaporation[i] < this->graph.depression_tree.volume_sed[i])
    {
      
      if(this->graph.depression_tree.volume_sed[i] > 0)
      {
        this->drape_dep_only_sed(i);
      }

      continue;
    }

    double total = this->graph.depression_tree.volume_sed[i];
    for(auto no:this->graph.depression_tree.get_all_nodes(i) )
    {
      if(isinhere[no] == 'y')
      {
        throw std::runtime_error("Double lakecognition");
        // std::cout << "Double lakecognition" << std::endl/
      }
      isinhere[no] = 'y';

      if(this->graph.depression_tree.tippingnode[i] == no)
        continue;

      if(this->node_in_lake[no] == -1)
      {
        if(i ==1)
        continue;

      }
        // throw std::runtime_error("afshdffdaglui;regji;");

      // if(this->topography[no] != this->graph.depression_tree.hw[i])
      // {

      //   std::cout <<"DN::" << no << " [] " <<  this->node_in_lake[no] << " || " << this->topography[no] << "||" << this->graph.depression_tree.hw[i] << " || " << this->surface_elevation[no] << " is outlet? " << this->graph.depression_tree.is_outlet(i) << std::endl;
      //   // throw std::runtime_error("NOT THE RIGHT ELEV");
      // }

      totsed += ratio_of_dep * (topography[no] - surface_elevation[no]) * cellarea;

      totwat += (this->topography[no] - surface_elevation[no]) * cellarea;

      double slangh = ratio_of_dep * (topography[no] - surface_elevation[no]) / timestep;
      if(topography[no] == surface_elevation[no])
        slangh = 0;

      if(!std::isfinite(slangh))
      {
        std::cout << "WARNING:: NAN IN SED CREA DURING LAKE DRAPING" << std::endl;
        throw std::runtime_error("WARNING:: NAN IN SED CREA DURING LAKE DRAPING");
        std::cout << ratio_of_dep << "/" << (topography[no] - surface_elevation[no]) << std::endl;
        slangh = 0;
      }
      // std::cout << "SLANGH IS " << slangh << " VS " << chonk_network[no].get_erosion_flux_only_bedrock() << "|" << chonk_network[no].get_erosion_flux_only_sediments() << std::endl;
      
      // if(slangh > 0.1)
      //   throw std::runtime_error("drape_deposition_flux_to_chonks::sedcrea too high");

      chonk_network[no].add_sediment_creation_flux(slangh);
      chonk_network[no].add_deposition_flux(slangh); // <--- This is solely for balance calculation
      chonk_network[no].set_label_tracker(this->graph.depression_tree.label_prop[i]);
      
      // chonk_network[no].reset_sed_fluxes();
    }


    DEBUG_totsed += totsed;
    // Seems fine here...
    // std::cout << "BALANCE LAKE = " << total << " out of " << this->graph.depression_tree.volume_sed[i] << std::endl;
    // if(double_equals(totsed , this->graph.depression_tree.volume_sed[i], 1) == false || double_equals(totwat + this->graph.depression_tree.actual_amount_of_evaporation[i] , this->graph.depression_tree.volume_water[i],1) == false)
    // if(true)
    // {
    //   std::cout << i << " TOT IN SED = " << totsed << " out of " << this->graph.depression_tree.volume_sed[i] << " AND VOL WAS " << this->graph.depression_tree.volume[i] << std::endl;
    //   std::cout << i << " TOT IN WATER = " << totwat + this->graph.depression_tree.actual_amount_of_evaporation[i] << " out of " << this->graph.depression_tree.volume_water[i] << " AND VOL WAS " << this->graph.depression_tree.volume_max_with_evaporation[i] << std::endl;
    //   std::cout << i << " hw " << this->graph.depression_tree.hw[i] << " vs max " << this->graph.depression_tree.hw_max[i] << " pitelev " << this->surface_elevation[this->graph.depression_tree.pitnode[i]] << " outelev " << this->surface_elevation[this->graph.depression_tree.tippingnode[i]] << std::endl;
    //   double totvolmax = 0;
    //   for(auto no:this->graph.depression_tree.get_all_nodes(i) )
    //   {
    //     // std::cout << no << " in lake " << i << " -> " << this->node_in_lake[no] <<  " this->topography[no]" << this->topography[no] << std::endl;
    //     // std::cout << no << " in lake " << i << " -> " << this->node_in_lake[no] << std::endl;
    //     totvolmax += (this->topography[no] - this->surface_elevation[no]) * this->cellarea;
    //   }
    //   std::cout << "totvolmax is " << totvolmax << std::endl; 
    //   std::cout << std::endl;

    // }
  }

  // std::cout << "DEBUG::drape_lake_sed::tototsed=" << DEBUG_totsed << std::endl;

}


void ModelRunner::drape_dep_only_sed(int dep_ID)
{

  double Vs = this->graph.depression_tree.volume_sed[dep_ID];

  if(Vs == 0)
    return;

  this->NTIME_DRAPEONLYSEDHAPPENED++;
  // Depressions which have only sed need special treatment
  auto nodes = this->graph.depression_tree.get_all_nodes(dep_ID);

  std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > filler;

  double  Totlea_local_sed = 0, tootlyc = 0;
  
  for(auto n:nodes)
  {
    filler.emplace(PQ_helper<int,double>(n,this->surface_elevation[n]));
  }

  std::vector<int> nodes2fill; nodes2fill.reserve(nodes.size());

  double hs = filler.top().score;
  int nnodes = 0;
  while(Vs > 0 && filler.empty() == false)
  {
    int node = filler.top().node;
    filler.pop();

    nnodes ++;
    double next_hs;

    if(filler.empty() == true)
    {
      next_hs = this->surface_elevation[this->graph.depression_tree.tippingnode[dep_ID]];
    }
    else
    {
      next_hs = filler.top().score;
    }

    double dz = next_hs - hs;
    double dV = nnodes * this->cellarea * dz;

    double ratio = Vs/dV;
    if(ratio > 1)
      ratio = 1;

    dV = ratio * dV;
    Vs -= dV;

    Totlea_local_sed += dV;
    
    nodes2fill.emplace_back(node);

    hs += ratio * (dz);
  }

  for(auto node:nodes2fill)
  {

    double slangh = (hs - this->surface_elevation[node])/this->timestep;

    if(slangh > 1e3)
      throw std::runtime_error("slangh is > 1e3 in lake draping onlysed");

    if(std::isfinite(slangh) == false)
      throw std::runtime_error("NAN in lake draping"); 

    // if(tootlyc < 0)
    //   std::cout << "HAPPENS YOLOYLOL BENG BENG " << hs << "|" << this->surface_elevation[node] << std::endl;;

    if(slangh<0)
      slangh = 0;

    tootlyc += slangh * this->cellarea * this->timestep;

    if( slangh * this->cellarea * this->timestep > this->NTIME_DRAPEONLYSEDHAPPENEDMAX)
      this->NTIME_DRAPEONLYSEDHAPPENEDMAX = slangh * this->cellarea * this->timestep;


    this->chonk_network[node].add_sediment_creation_flux(slangh);
    this->chonk_network[node].add_deposition_flux(slangh);
  }

  // std::cout << "Had " << this->graph.depression_tree.volume_sed[dep_ID] << " stored " << Totlea_local_sed << " or " << tootlyc << std::endl;

  if(double_equals(this->graph.depression_tree.volume_sed[dep_ID],tootlyc, 100) == false)
    throw std::runtime_error("drape_dep_only_sed::SedDrapedMassBalanceError");

  // Just double checking there are not a huge amount of extra sediment
  if(Vs > 1e2)
    std::cout << "EXTRA SED IN THE FILLING OF SED-ONLY DEP " << Vs << std::endl;

}



xt::pytensor<double,2> ModelRunner::get_Qsprop_bound( std::string which)
{

  // Preformatting the output
  xt::pytensor<double,2> output;
  if(which == "N" || which == "S")
    output = xt::zeros<double>({this->io_int["nx"], this->n_labels});
  else
    output = xt::zeros<double>({this->io_int["ny"], this->n_labels});

  if(which == "N")
  {
    int ti = -1;
    for (int i=0; i < this->io_int["nx"]; ++i)
    {
      ti++;
      auto dese = this->chonk_network[i].get_label_tracker();
      for(int j = 0; j<this->n_labels;j++)
        output(ti,j) = dese[j] * this->chonk_network[i].get_sediment_flux() / this->timestep;
    }
  }
  else if(which == "S")
  {
    int ti = -1;
    for (int i=this->io_int["n_elements"] - this->io_int["nx"]; i < this->io_int["n_elements"]; ++i)
    {
      ti++;
      auto dese = this->chonk_network[i].get_label_tracker();
      for(int j = 0; j<this->n_labels;j++)
        output(ti,j) = dese[j] * this->chonk_network[i].get_sediment_flux() / this->timestep;
    }
  }
  else if(which == "E")
  {
    int ti = -1;
    for (int i=0; i < this->io_int["n_elements"]; i = i + this->io_int["ny"])
    {
      ti++;
      auto dese = this->chonk_network[i].get_label_tracker();
      for(int j =0; j < this->n_labels;j++)
        output(ti,j) = dese[j] * this->chonk_network[i].get_sediment_flux() / this->timestep;
    }
  }
  else if(which == "W")
  {
    int ti = -1;
    for (int i=this->io_int["nx"] - 1; i < this->io_int["n_elements"]; i = i + this->io_int["ny"])
    {
      ti++;
      auto dese = this->chonk_network[i].get_label_tracker();
      for(int j = 0; j < this->n_labels;j++)
        output(ti,j) = dese[j]* this->chonk_network[i].get_sediment_flux() / this->timestep;
    }
  }


  return output;


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


void ModelRunner::gather_nodes_to_reproc(std::vector<int>& local_mstack, 
  std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> >& ORDEEEEEER,
   std::vector<char>& is_in_queue, int outlet)
{

  // DEPRECATED BUT I AM KEEPING IT IN CASE

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

    if(is_in_queue[next_node] == 'l')
      continue;

    // If this is not the outlet, which will be treated separatedly I add it in the PQ to reproc
    // if(next_node != outlet )
    if(true )
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
      if(is_in_queue[tnode] == 'y' || is_in_queue[tnode] == 'l' || this->is_processed[tnode] == false)
        continue;
      
      // if it is below my current node and not a lake 
      if(topography[tnode] < topography[next_node] && this->node_in_lake[tnode] == -1 )
      {
        //Else, this is a node to reproc, well done
        transec.emplace(tnode);
        is_in_queue[tnode] = 'y';
        continue;
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



// Getting the strati proportions output info
// CAN BE A BIT CONVOLUTED BUT IS THE BEST solution I CAN THINK OF
// output structure: 
// output[0] is a 2D array of node coordinates storing [0] zmax and [1] zmin
// output[1] is a 1D array of node coordinates storing the number of depth cell by node (BE CAREFULL, CELLS WITH NO DEPTHS ARE STILL COUNTED)
// output[2] is a 1D array of number of cells in depth (0 depths cells are counted as 1 empty) * n_label coordinates storing the props by labels
// output[3] is a 1D array of number of cells in depth (0 depths cells are counted as 1 empty) * n_label coordinates storing the volume of sed store there
// output[4] is the n_labels, for convenience to have it here
std::tuple< xt::pytensor<float,2>, xt::pytensor<int,1>,  xt::pytensor<float,1>,  xt::pytensor<float,1>, int> ModelRunner::get_stratiprop()
{
  // First getting the number of cells
  // #-> considering that each pixel will have at least an entry
  int ncellsdepths = this->io_int["n_elements"];
  for(auto& val : this->sed_prop_by_label)
  {
    // # if there is more than one cell-> I need to increment the number minus the mandatory cell
    if(val.second.size()>1)
      ncellsdepths += val.second.size() - 1;
  }

  xt::pytensor<float,2> A1;
  xt::pytensor<int,1> A2;
  xt::pytensor<float,1> A3;
  xt::pytensor<float,1> A4;
  int A5;

  // Preformatting the output
  std::tuple< xt::pytensor<float,2>, xt::pytensor<int,1>,  xt::pytensor<float,1>,  xt::pytensor<float,1>, int> output = std::make_tuple(A1, A2, A3, A4, A5);

  // going through all the nodes to get the output
  int j=0;
  for(int i=0; i<this->io_int["n_elements"];  ++i)
  {
    std::get<0>(output)(i,0) = this->surface_elevation_tp1[i];
    std::get<0>(output)(i,1) = this->surface_elevation_tp1[i] - this->sed_height_tp1[i];
    std::get<1>(output)[i] = this->sed_prop_by_label[i].size();
    if(std::get<1>(output)[i] == 0)
    {
      std::get<1>(output)[i] = 1;
      for(int k=0; k < this->n_labels; k++)
      {
        std::get<2>(output)[j] = 0;
        std::get<3>(output)[j] = 0;
        j++;
      }
    }
    else
    {
      for(int i2 = 0; i2 < int(std::get<1>(output)[i]); ++i2)
      {
        for(int k=0; k < this->n_labels; k++)
        {
          std::get<2>(output)[j] = this->sed_prop_by_label[i][i2][k];
          std::get<3>(output)[j] = this->sed_prop_by_label[i][i2][k];
          j++;
        }
      }
    }
  }
  std::get<4>(output) = this->n_labels;
  return output;

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



















void ModelRunner::iterative_lake_solver()
{
  // DEPRECATED
}


void ModelRunner::reprocess_nodes_from_lake_outlet_v2(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, 
  EntryPoint& entry_point)
{
  // DEPRECATED
}

void ModelRunner::unpack_entry_points_from_delta_maps(std::queue<int>& iteralake, std::vector<std::vector<double> >& label_prop_of_delta,
std::vector<double>& delta_sed, std::vector<double>& delta_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre,
std::vector<double>& pre_sed, std::vector<double>& pre_water)
{ 
  // DEPRECATED
}

void ModelRunner::label_nodes_with_no_rec_in_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, std::vector<char>& has_recs_in_local_stack)
{
  // Deprecated

}



bool ModelRunner::has_valid_outlet(int lakeid)
{
  // Deprecated
}


void ModelRunner::reprocess_nodes_from_lake_outlet(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<int>& iteralake, EntryPoint& entry_point)
{
  std::cout << "ONGOING DEPRECATION" << std::endl;
}



void ModelRunner::check_what_gives_to_lake(int entry_node, std::vector<int>& these_lakid , std::vector<double>& twat, std::vector<double>& tsed,
 std::vector<std::vector<double> >& tlab, std::vector<int>& these_ET, int lake_to_ignore)
{
  // Deprecated


}

int ModelRunner::fill_mah_lake(EntryPoint& entry_point, std::queue<int>& iteralake)
{
  // DEPRECATED
  return 0;
}

/// function eating a lake from another
void ModelRunner::drink_lake(int id_eater, int id_edible, EntryPoint& entry_point, std::queue<int>& iteralake)
{ 
 // DEPRECATED
}


// Function checking the active lake id of a node (ie, the top-level one)
int ModelRunner::motherlake(int this_lake_id)
{
  // DEPRECATED
  return 0;
}

void ModelRunner::original_gathering_of_water_and_sed_from_pixel_or_flat_area(int starting_node, double& water_volume, double& sediment_volume, 
  std::vector<double>& label_prop, std::vector<int>& these_nodes)
{
  // Deprecated

}



void ModelRunner::reprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, int outlet, int current_lake,
  std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >& SL_corrector)
{
  // DEPRECATED DEPRECATED DEPRECATED
  return;
}


void ModelRunner::deprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, int outlet)
{
  // DERPCATED DEPRECATED DEPRECATED
}


void ModelRunner::check_what_give_to_existing_outlets(std::map<int,double>& WF_corrector,  std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >&  SL_corrector, std::vector<int>& local_mstack)
{
  // Deprecated
  return;
}

void ModelRunner::check_what_give_to_existing_lakes(std::vector<int>& local_mstack, int outlet, int current_lake, std::vector<double>& this_sed,
   std::vector<double>& this_water, std::vector<int>& this_entry_node, std::vector<std::vector<double> >& label_prop_of_this)
{

  // Deprecated
  return;
}


chonk ModelRunner::preprocess_outletting_chonk(chonk tchonk, EntryPoint& entry_point, int current_lake, int outlet,
 std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, std::map<int,std::vector<double> >& SL_corrector,
 std::vector<double>& pre_sed, std::vector<double>& pre_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre)
{
  // Deprecated
  return chonk();
}



// DEPREACTED??
void ModelRunner::increment_new_lake(int& lakeid)
{
// DEPRECATED}
}


void ModelRunner::lake_solver_v3(int node)
{
  // DEPRECATED
  return;
}




















#endif

