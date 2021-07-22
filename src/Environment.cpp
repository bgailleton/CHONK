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

  // DEBUG STUFF IGNORER
  this->NTIMEPREFLUXCALLED = 0;


  // std::cout << "initiating nodegraph..." <<std::endl;
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
  this->graph = NodeGraphV2(this->surface_elevation, this->active_nodes,this->dx, this->dy,
                            this->io_int["n_rows"], this->io_int["n_cols"], this->lake_solver);
  
  this->topography = xt::pytensor<double,1>(this->surface_elevation);


  this->lake_to_process = std::vector<int>();

  if(this->lake_evaporation)
  {
    if(this->lake_evaporation_spatial == false)
    {
      //
      this->lake_evaporation_rate_spatial = this->lake_evaporation_rate + xt::zeros_like(this->topography);
    }

    this->graph.depression_tree.preprocess_lake_evaporation_potential(this->lake_evaporation_rate_spatial, this->cellarea, this->timestep);


  }

  // Also initialising the Lake graph
  //# incrementor reset to 0
  lake_incrementor = 0;
  this->lakes.clear();
  //# Labelling the nodes in the lake
  this->node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);
  for(int i = 0; i < this->graph.depression_tree.get_n_dep(); i++)
  {
    if(this->graph.depression_tree.has_children(i) == false)
    {
      this->node_in_lake[this->graph.depression_tree.pitnode[i]] = i;
      std::cout << "registering dep " << i << " at " << this->graph.depression_tree.pitnode[i] << std::endl;
    }

  }

  // I need the topoogical order of my depressions: which depressions will i get first
  this->lake_in_order = this->graph.get_Cordonnier_order();
  // Initialising the lake status array
  // this->lake_status = std::vector<int>(this->io_int["n_elements"],-1);
  // // Initialising the depression pits to 0
  // for(auto tn:lake_in_order)
  // {
  //   this->lake_status[tn] = 0;
  // }

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
    // std::cout << "done" << std::endl;
    // ### Saving the local production of sediment, in order to cancel it later
    this->local_Qs_production_for_lakes[node] = this->chonk_network[node].get_sediment_flux() - templocalQS; 

    // Switching to the next node in line
  }

  // First pass is done, all my nodes have been processed once. The flux is done is lake solver is implicit and we can finalise.

  // // If the lake solver is explicit though, I can start the iterative process
  // if(this->lake_solver)
  // {
  //   // Debug variable to ignore
  //   DEBUG_GLOBDELT = 0;
  //   // Running the iterative lake solver
  //   this->iterative_lake_solver();
  // }

  // Calling the finalising function: it applies the changes in topography and I think will apply the lake sedimentation
  this->finalise();
  // Done

  int ndfgs=0;
  for( size_t i=0; i< is_processed.size() ; i++)
  {
    if(is_processed[i] == false)
      ndfgs++;
  }
  if(ndfgs>0)
  {
    // throw std::runtime_error("NODE UNPROCESSED::" + std::to_string(ndfgs));
    std::cout << "WARNING::Node Unprocessed::"  + std::to_string(ndfgs) << std::endl;
  }

  std::cout << NTIMEPREFLUXCALLED << " TIMES CALLING PREFLUX METHOD" <<  std::endl;

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
          ignore_some.emplace_back(ttnode);
          continue;
        }
        // I also ignore the nodes in the current lake (they have a 'y' signature)
        else if (this->node_in_lake[ttnode] >= 0)
        {
          ignore_some.emplace_back(ttnode);
          continue;
        }
      }

      // # Ignore_some has the node i so not want
      // # So I transmit my fluxes to the nodes I do not ignore
      this->chonk_network[tnode].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);

      // std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
      // this->chonk_network[tnode].copy_moving_prep(rec,wwf, wws, strec);
      // for(size_t u =0; u< rec.size(); ++u)
      // {
      //   int ttnode = rec[u]; 

      //   if(std::find(ignore_some.begin(), ignore_some.end(), ttnode) != ignore_some.end())
      //     continue;
      //   if(this->is_this_node_in_this_lake(ttnode, current_lake))
      //     continue;

      //   if(this->active_nodes[ttnode])
      //     this->sed_added_by_donors += wws[u] *  this->chonk_network[tnode].get_sediment_flux();
      //   // else
      //   //   this->sed_added_by_donors -= wws[u] *  this->chonk_network[tnode].get_sediment_flux();

      // }
      

    }
    else
    {
      // # I am not a nodor:
      // if ( this->has_been_outlet[tnode] == 'y' )
      // {
      //   this->n_outlets_remodelled ++;
      //   this->chonk_network[tnode].add_to_water_flux( WF_corrector[tnode]);
      //   this->chonk_network[tnode].add_to_sediment_flux( -1 * local_Qs_production_for_lakes[tnode], this->chonk_network[tnode].get_fluvialprop_sedflux());
      //   this->chonk_network[tnode].add_to_sediment_flux( SF_corrector[tnode], SL_corrector[tnode], this->chonk_network[tnode].get_fluvialprop_sedflux());
      //   this->sed_added_by_donors -= SF_corrector[tnode];
      // }
      // # So I need full reproc yaaay
      // full_delta -= this->chonk_network[tnode].get_sediment_flux();
      this->process_node_nolake_for_sure(tnode, is_processed, this->active_nodes, 
        cellarea,topography, true, true);
  //     this->chonk_network[tnode].check_sums();
  //     // full_delta += this->chonk_network[tnode].get_sediment_flux();
  //     // local_Qs_production_for_lakes[tnode] = full_delta;
  //     this->sed_added_by_prod += this->chonk_network[tnode].get_erosion_flux_only_bedrock()\
  //  * this->timestep * this->dx * this->dy;
  // this->sed_added_by_prod += this->chonk_network[tnode].get_erosion_flux_only_sediments()\
  //  * this->timestep * this->dx * this->dy;
  // this->sed_added_by_prod -= this->chonk_network[tnode].get_deposition_flux()\
  //  * this->timestep * this->dx * this->dy;

      // std::vector<int> rec;std::vector<double> wwf;std::vector<double> wws; std::vector<double> strec; 
      // this->chonk_network[tnode].copy_moving_prep(rec,wwf, wws, strec);
      // for(size_t u =0; u< rec.size(); ++u)
      // {
      //   int ttnode = rec[u]; 
      //   // if(ttnode == 1752)
      //   //   std::cout << "node y giving " << wws[u] *  this->chonk_network[tnode].get_sediment_flux() << " to 1752" << std::endl;
      //   if((is_in_queue[ttnode] == 'y' || ttnode == this->lakes[current_lake].outlet) && this->active_nodes[ttnode])
      //     continue;
      //   if(this->is_this_node_in_this_lake(ttnode, current_lake))
      //     continue;

      //   sed_outletting_system += wws[u] *  this->chonk_network[tnode].get_sediment_flux();
      // }
   

    }

    // std::cout << " Now is " << this->chonk_network[tnode].get_sediment_flux() << std::endl;;
    // this->chonk_network[tnode].print_status();

  }

    // std::cout << " Full delta is " << full_delta << std::endl;;

}
 
void ModelRunner::deprocess_local_stack(std::vector<int>& local_mstack, std::vector<char>& is_in_queue, int outlet)
{
  // Now deprocessing the receivers in potential lake while saving their contribution to lakes in order to calculate the delta
  for(int i = local_mstack.size() - 1; i>=0; i--)
  {
    int tnode = local_mstack[i];
    // # If my node is just a donor, I am not resetting it
    if (is_in_queue[tnode] != 'y')
      continue;

    // std::cout
    // # Cancelling the fluxes before moving prep (i.e. the precipitation, infiltrations, ...)
    // # This is not for water purposes as it gets reproc anyway, but for mass balance calculation
    std::cout << "cancels:" << tnode << "|";
    this->chonk_network[tnode].cancel_split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->timestep);

    if(tnode != outlet)
    {
      this->cancel_fluxes_before_moving_prep(this->chonk_network[tnode], tnode);
    }
    // this->chonk_network[tnode].reset();
    // this->chonk_network[tnode].set_label_tracker(std::vector<double>(this->n_labels,0));
    // this->is_processed[tnode] = false;

  }


  // std::cout << "outlet is " << outlet << " and nodes are: ";

  for(auto tnode : local_mstack)
  {
    // std::cout << tnode << ":" << is_in_queue[tnode] << "|";
    if (is_in_queue[tnode] != 'y')
      continue;

    if(tnode == outlet)
      this->chonk_network[tnode].reset();
    else
    {
      this->chonk_network[tnode].reset_sed_fluxes();
      this->chonk_network[tnode].reinitialise_moving_prep();
    }

    this->chonk_network[tnode].set_label_tracker(std::vector<double>(this->n_labels,0));
    this->is_processed[tnode] = false;
  }
}

void ModelRunner::check_what_give_to_existing_outlets(std::map<int,double>& WF_corrector,  std::map<int,double>& SF_corrector, 
  std::map<int,std::vector<double> >&  SL_corrector, std::vector<int>& local_mstack)
{
  // Deprecated
  // // going through the nodes of the local stack
  // for( auto node : local_mstack)
  // {
  //   chonk& tchonk = this->chonk_network[node];

  //   //getting the weights
  //   // # Initialising a bunch of intermediate containers and variable
  //   std::vector<int> tchonk_recs;
  //   std::vector<double> tchonk_slope_recs;
  //   std::vector<double> tchonk_weight_water_recs, tchonk_weight_sed_recs;

  //   // copying the weights from the current 
  //   tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);

  //   // checking all neighboiyrs
  //   for(size_t i =0; i < tchonk_recs.size(); i++)
  //   {
  //     // node indice of the receiver
  //     int tnode = tchonk_recs[i];


  //     // Now checking if the rec is an outlet:
  //     if(this->has_been_outlet[tnode] == 'y')
  //     {
  //       if(WF_corrector.count(tnode) == 0)
  //       {
  //         WF_corrector[tnode] = 0;
  //         SF_corrector[tnode] = 0;
  //         SL_corrector[tnode] = {};
  //       }
  //       WF_corrector[tnode] -= tchonk_weight_water_recs[i] * tchonk.get_water_flux();
  //       SL_corrector[tnode] = mix_two_proportions(SF_corrector[tnode],SL_corrector[tnode], -1 * tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux(), tchonk.get_label_tracker());
  //       SF_corrector[tnode] -= tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux();
  //       std::cout << "CORRECTOR ON " << tnode << " IS " << SF_corrector[tnode] << std::endl;
        
  //     }
  //   }
  // }
}

void ModelRunner::check_what_give_to_existing_lakes(std::vector<int>& local_mstack, int outlet, int current_lake, std::vector<double>& this_sed,
   std::vector<double>& this_water, std::vector<int>& this_entry_node, std::vector<std::vector<double> >& label_prop_of_this)
{

  // Deprecated
  // for( auto node : local_mstack)
  // {
  //   chonk& tchonk = this->chonk_network[node];

  //   //getting the weights
  //   // # Initialising a bunch of intermediate containers and variable
  //   std::vector<int> tchonk_recs;
  //   std::vector<double> tchonk_slope_recs;
  //   std::vector<double> tchonk_weight_water_recs, tchonk_weight_sed_recs;

  //   // copying the weights from the current 
  //   tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);

  //   for(size_t i =0; i < tchonk_recs.size(); i++)
  //   {
  //     // node indice of the receiver
  //     int tnode = tchonk_recs[i];
  //     // And checking if the rec is a lake
  //     int lakid = this->node_in_lake[tnode];
  //     if(lakid >= 0)
  //     {
  //       lakid = this->motherlake(lakid);
  //       if(lakid != current_lake)
  //       {
  //         // Old debug statement
  //         // std::cout << "Adding " << tchonk_weight_water_recs[i] * tchonk.get_water_flux() << " to " << lakid << " from " << node << "->" << tnode << " Breakdown: " << tchonk_weight_water_recs[i] << " * " <<  tchonk.get_water_flux() << std::endl; ;
  //         label_prop_of_this[lakid] = mix_two_proportions(this_sed[lakid],label_prop_of_this[lakid], tchonk_weight_sed_recs[i]* tchonk.get_sediment_flux(), tchonk.get_label_tracker());
  //         this_sed[lakid] += tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux();
  //         this_water[lakid] += tchonk_weight_water_recs[i] * tchonk.get_water_flux();
  //         this_entry_node[lakid] = tnode;
  //       }
  //     }
  //   }
  // }
}


chonk ModelRunner::preprocess_outletting_chonk(chonk tchonk, EntryPoint& entry_point, int current_lake, int outlet,
 std::map<int,double>& WF_corrector, std::map<int,double>& SF_corrector, std::map<int,std::vector<double> >& SL_corrector,
 std::vector<double>& pre_sed, std::vector<double>& pre_water, std::vector<int>& pre_entry_node, std::vector<std::vector<double> >& label_prop_of_pre)
{
  // Deprecated
  return chonk();
 //  // Getting the additioned water rate
 //  // std::cout << "I WATER RATE IS " << tchonk.get_water_flux() << std::endl;
 //  double water_rate = entry_point.volume_water / this->timestep;
 //  // std::cout << "II WATER RATE IS " << water_rate << std::endl;
 //  // Summing it to the previous one
 //  water_rate += tchonk.get_water_flux();
 //  // std::cout << "III WATER RATE IS " << water_rate << std::endl;

 //  // Dealing with sediments
 //  std::vector<double> label_prop = entry_point.label_prop;//mix_two_proportions(entry_point.volume_sed,entry_point.label_prop, tchonk.get_sediment_flux(), tchonk.get_label_tracker());
  
 //  double sedrate = entry_point.volume_sed + tchonk.get_sediment_flux() - local_Qs_production_for_lakes[outlet];
 //  std::cout << "Outlet = " << entry_point.volume_sed << " + " <<  \
 //  tchonk.get_sediment_flux() << " - " << local_Qs_production_for_lakes[outlet] << " = " << sedrate << std::endl;
 //  entry_point.volume_sed = 0;

 //  this->sed_already_outletted += tchonk.get_sediment_flux() - local_Qs_production_for_lakes[outlet];

 //  //getting the weights
 //  // # Initialising a bunch of intermediate containers and variable
 //  std::vector<int> ID_recs, tchonk_recs;
 //  std::vector<double> slope_recs,tchonk_slope_recs;
 //  std::vector<double> weight_water_recs, weight_sed_recs, tchonk_weight_water_recs, tchonk_weight_sed_recs;
 //  double sumW = 0;
 //  double sumS = 0;
 //  int nrecs = 0;
 //  // copying the weights from the current 
 //  tchonk.copy_moving_prep(tchonk_recs,tchonk_weight_water_recs,tchonk_weight_sed_recs,tchonk_slope_recs);
  
 //  // Iterating through the receivers of the outlet and removing the water given to the current lake by this outlet prior reprocessing
 //  // this water is already in the lake outletting thingy!
 //  for(size_t i = 0; i < tchonk_recs.size(); i++)
 //  {
 //    // Node
 //    int tnode = tchonk_recs[i];
 //    // lake?
 //    int tlak = this->node_in_lake[tnode];
 //    if(tlak >= 0)
 //      tlak = this->motherlake(tlak);
 //    else
 //      continue;

 //    // Is current lake?
 //    if(tlak == current_lake)
 //    {
 //      // yes, removing sed and water rate
 //      water_rate -= tchonk_weight_water_recs[i] * tchonk.get_water_flux();
 //      // label_prop = mix_two_proportions(sedrate,  label_prop, -1 * tchonk_weight_water_recs[i] * tchonk.get_sediment_flux(),  tchonk.get_label_tracker());
 //      // sedrate -= tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux();
 //      // std::cout << "Subtracting II " << tchonk_weight_sed_recs[i] * tchonk.get_sediment_flux() << std::endl;; 
 //    }
 //  }

 //  // Now iterating thorugh the neighbours
 //  std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(outlet, this->active_nodes, neightbors, dummy);

 //  // calculating the slope too 
 //  double SS = -9999;
 //  int SS_ID = -9999;
 //  for(size_t i = 0; i< neightbors.size(); i++)
 //  {
 //    // node indice of the receiver
 //    int tnode = neightbors[i];

 //    // J is the indice of this specific node in the tchonk referential
 //    int j = -1;
 //    auto itj = std::find(tchonk_recs.begin(), tchonk_recs.end(), tnode);
 //    if(itj != tchonk_recs.end() )
 //      j = std::distance(tchonk_recs.begin(), itj);
 //    // Note that if j is still < 0, tnode is not in the chonk

 //    if(topography[tnode] >= topography[outlet])
 //    {
 //      // double tsedfromdon = this->chonk_network[tnode].sed_flux_given_to_node(outlet);
 //      // label_prop = mix_two_proportions(tsedfromdon, this->chonk_network[tnode].get_label_tracker(), sedrate, label_prop);
 //      // sedrate += tsedfromdon;
 //      continue;
 //    }

 //    // Checking wether it is giving to the original lake or not
 //    if(this->is_this_node_in_this_lake(tnode, current_lake) ==  false)
 //    {
 //      // get the node
 //      ID_recs.emplace_back(tnode);
 //      // calculate the slope
 //      double tS = topography[outlet] - topography[tnode];
 //      tS = tS / dummy[i];
 //      // If j exists push the weights
 //      if(j >= 0)
 //      {
 //        weight_water_recs.emplace_back(tchonk_weight_water_recs[j]);
 //        weight_sed_recs.emplace_back(tchonk_weight_sed_recs[j]);
 //        sumW += tchonk_weight_water_recs[j];
 //        sumS += tchonk_weight_sed_recs[j];
 //      }
 //      else
 //      {
 //        // else 0
 //        weight_water_recs.emplace_back(0);
 //        weight_sed_recs.emplace_back(0);
 //      }
      
 //      // Slope      
 //      slope_recs.emplace_back(tS);
 //      nrecs++;

 //    }
 //    // else
 //    // {
 //    //   if(j >= 0)
 //    //   {
 //    //     this->Qs_mass_balance -= tchonk_weight_sed_recs[j] * tchonk.get_sediment_flux();
 //    //   }
 //    // }

 //    // Now checking if the rec is an outlet and pushing the correctors:
 //    if(this->has_been_outlet[tnode] == 'y')
 //    {
 //      if(WF_corrector.count(tnode) == 0)
 //      {
 //        WF_corrector[tnode] = 0;
 //        SF_corrector[tnode] = 0;
 //        SL_corrector[tnode] = {};
 //      }

 //      if(j>=0)
 //      {
 //        WF_corrector[tnode] -= tchonk_weight_water_recs[j] * tchonk.get_water_flux();
 //        SL_corrector[tnode] = mix_two_proportions(SF_corrector[tnode],SL_corrector[tnode], -1 * tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux(), tchonk.get_label_tracker());
 //        SF_corrector[tnode] -= tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux();
 //      }
 //    }

 //    // And finally checking if the rec is a lake
 //    int lakid = this->node_in_lake[tnode];
 //    if(lakid >= 0 && j >= 0)
 //    {
 //      lakid = this->motherlake(lakid);
 //      if(lakid != current_lake)
 //      {
 //        label_prop_of_pre[lakid] = mix_two_proportions(pre_sed[lakid],label_prop_of_pre[lakid], tchonk_weight_sed_recs[j]* tchonk.get_sediment_flux(), tchonk.get_label_tracker());
 //        pre_sed[lakid] += tchonk_weight_sed_recs[j] * tchonk.get_sediment_flux() ;
 //        pre_water[lakid] += tchonk_weight_water_recs[j] * tchonk.get_water_flux();
 //        pre_entry_node[lakid] = tnode;
 //      }
 //    }
 //  }

 //  // Normalising the weights to their new states
 //  if(sumW > 0)
 //  {
 //    for(size_t i =0; i < weight_water_recs.size(); i++)
 //    {
 //      weight_water_recs[i] = weight_water_recs[i]/sumW;
 //    }
 //  }
 //  else
 //  {
 //    weight_water_recs = std::vector<double>(weight_water_recs.size(), 1./int(weight_water_recs.size()));
 //  }

 //  // if(sumS > 0)
 //  // {
 //  //   for(size_t i =0; i < weight_sed_recs.size(); i++)
 //  //   {
 //  //     weight_sed_recs[i] = weight_sed_recs[i]/sumS;
 //  //   }

 //  // }
 //  // else
 //  // {
 //  //   weight_sed_recs = std::vector<double>(weight_sed_recs.size(), 1./int(weight_sed_recs.size()));

 //  // }
 //  weight_sed_recs = std::vector<double>(ID_recs.size(),0);

 //  std::cout << "outlet was " << tchonk.get_sediment_flux() << " and is now " << sedrate << std::endl;;
 //  if(sedrate < 0)
 //  {
 //    std::cout << "Warning::Sedrate recasted to 0" << std::endl;;
 //    sedrate = 0;
 //  }


 //  // Resetting the CHONK
 //  tchonk.reset();
 //  tchonk.external_moving_prep(ID_recs,weight_water_recs,weight_sed_recs,slope_recs);
 //  tchonk.set_water_flux(water_rate);
 //  tchonk.set_sediment_flux(sedrate,label_prop, 1.);
 //  tchonk.I_solemnly_swear_all_my_sediments_are_fluvial();

 //  // for(auto idf:ID_recs)
 //  //   std::cout << idf << "||";
 //  // std::cout << std::endl;
 // // Old debug statement
 //  // std::cout << "Set to outlet::" << water_rate << " wat and sed: " << sedrate << std::endl; 
 //  // Ready to go ??!!
 //  return tchonk;
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

void ModelRunner::process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<bool,1>& active_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep)
{
    // Just a check: if the lake solver is not activated, I have no reason to reprocess node
    // if(this->lake_solver == false && is_processed[node])
    if(is_processed[node])
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
      // if not in a lake: not in a lake yo
      if(lakeid == -1)
        goto nolake;
 
      this->lake_solver_v4(node);


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
    // std::cout << this->chonk_network[node].get_sediment_creation_flux() << std::endl;;
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

  if(this->is_processed[node])
    return;

  this->is_processed[node] = true;
  
  // if(this->node_in_lake[node] > -1)
  //   return;

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
  // std::cout << "WOLO::0.5" << std::endl;

  // First dealing with lake deposition:
  this->drape_deposition_flux_to_chonks();
  // std::cout << "WOLO::1" << std::endl;

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

  // std::cout << "WOLO::2" << std::endl;

  // then actively finalising the deposition and other details
  // Iterating through all nodes
  for(int i=0; i< this->io_int["n_elements"]; i++)
  {

    // std::cout << "WULUT:" << std::endl;

    // if(std::isfinite(surface_elevation_tp1[i]) == false)
    // {
    //   throw std::runtime_error("NAN IN ELEV WHILE FINIlIISAFJ");
    // }

    // std::cout << i << "|"<< std::endl;


    double sedcrea = this->chonk_network[i].get_sediment_creation_flux() * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_bedrock() * cellarea * timestep;
    this->Qs_mass_balance -= this->chonk_network[i].get_erosion_flux_only_sediments() * cellarea * timestep;
    this->Qs_mass_balance += this->chonk_network[i].get_deposition_flux() * cellarea * timestep;

    // std::cout << "A|"<< std::endl;



    if(this->active_nodes[i] == false)
      continue;

    // std::cout << "B|"<< std::endl;


    // Getting the current chonk
    chonk& tchonk = this->chonk_network[i];

    // std::cout << "C|"<< std::endl;


    // getting the current composition of the sediment flux
    auto this_lab = tchonk.get_label_tracker();
    // std::cout << "C1|"<< std::endl;

    // NANINF DEBUG CHECKER
    for(auto LAB:this_lab)
      if(std::isfinite(LAB) == false)
        std::cout << LAB << " << naninf for sedflux" << std::endl;
    // std::cout << "C2|"<< std::endl;
 

    // First applying the bedrock-only erosion flux: decrease the overal surface elevation without affecting the sediment layer
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux_only_bedrock() * timestep;
    // std::cout << "C3|"<< std::endl;

    // Applying elevation changes from the sediments
    // Reminder: sediment creation flux is the absolute rate of removal/creation of sediments

    // NANINF DEBUG CHECKER
    if(std::isfinite(sedcrea) == false)
    {
      std::cout << "C3.1|" << sedcrea << std::endl;

      std::cout << sedcrea << "||" << this->node_in_lake[i] << "||"<< std::endl;
      throw std::runtime_error("NAN sedcrea finalisation not possible yo");
    }

    // std::cout << "D|"<< std::endl;


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
      // std::cout << "0.3 " << sedcrea << " Lake?? " << this->node_in_lake[i] << std::endl;
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

    // std::cout << "E|"<< std::endl;
;



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

    // std::cout << "F|"<< std::endl;
;


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

    // std::cout << "F|" << std::endl;;

  }

  // std::cout << "WOLO::3" << std::endl;


  auto tlake_depth = this->topography - this->surface_elevation;
  // Calculating the water balance thingies
  // double save_Ql_out = this->Ql_out;
  // this->Ql_out = 0;
  for(int i=0; i<this->io_int["n_elements"]; i++)
  {
    this->Ql_out += (tlake_depth[i] - this->io_double_array["lake_depth"][i]) * this->dx * this->dy / this->timestep;
  }

  
  // Saving the new lake depth  
  this->io_double_array["lake_depth"] = tlake_depth;


  // calculating other water mass balance.
  // xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  for(int i = 0; i<this->io_int["n_elements"]; i++)
  {
    if(this->active_nodes[i] == false)
    {
      this->Qw_out += this->chonk_network[i].get_water_flux();
      this->Qs_mass_balance += this->chonk_network[i].get_sediment_flux();
    }
  }
}

void ModelRunner::lake_solver_v4(int node)
{

  // Deug statement -> to remove once efixed
  int this_dep = this->node_in_lake[node];
  std::cout << std::endl ;
  std::cout << "lake_solver_v4 -> starting " << this_dep << std::endl;

  // Lines to uncomment to check the continuity
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
      // this->graph.depression_tree.label_prop[dep];
    }

    // If there is a parent -> fire up
    if(parent > -1)
    {
      this->graph.depression_tree.transmit_up(dep);
    }

    // if the depression is full -> labelled as outflowing
    if(this->graph.depression_tree.is_full(dep))
      outflows[dep] = true;
    
    // // Outletting quantities are the delta between potential accomodation space and the inputted Q
    // outwat[dep] = this->graph.depression_tree.volume_water[dep] - this->graph.depression_tree.volume_max_with_evaporation[dep];
    // outsed[dep] = this->graph.depression_tree.volume_sed[dep] - this->graph.depression_tree.volume[dep];
    // outlab[dep] = this->graph.depression_tree.label_prop[dep];

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
  for(int i = int(treestack.size()-1); i >= 0; i--)
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

      // I now have a new state for outsed[dep], outlab[dep], outflows[dep], outwat[dep]
      // I need to check whether I have a twin here. If I do, I need to transmit it my outletting stuff
      if(twin > -1)
      {
        std::cout << "HAPPEEEEEENS" << std::endl;

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

          // outwat[chd] = this->graph.depression_tree.volume_water[chd] - this->graph.depression_tree.volume_max_with_evaporation[chd];
          // outsed[chd] = this->graph.depression_tree.volume_sed[chd] - this->graph.depression_tree.volume[chd];
          // outlab[chd] = this->graph.depression_tree.label_prop[chd];
          
          // if(outsed[chd] < 0)
          //   outsed[chd] = 0;
          // if(outwat[chd] < 0)
          //   outwat[chd] = 0;
          
          chd = this->graph.depression_tree.treeceivers[chd][0];
        }


        double lowerboundvoltwin_wat = this->graph.depression_tree.get_volume_of_children_water(twin);
        double lowerboundvoltwin_sed = this->graph.depression_tree.get_volume_of_children_sed(twin);

        if(this->graph.depression_tree.volume_water[twin] >= lowerboundvoltwin_wat || this->graph.depression_tree.volume_sed[twin] >= lowerboundvoltwin_sed)
        {
          std::vector<int> children = this->graph.depression_tree.get_all_children(twin,true);
          if(this->graph.depression_tree.processed[twin])
            throw std::runtime_error("IzAlreadyProc");

          for(auto chd:children)
          {
            this->graph.depression_tree.processed[chd] = true;
          }



          this->process_dep(twin, outsed[twin], outlab[twin], outflows[twin], outwat[twin]);
          this->graph.depression_tree.processed[twin] = true;
        }
      }

      std::vector<int> children = this->graph.depression_tree.get_all_children(dep,true);

      for(auto chd:children)
      {
        this->graph.depression_tree.processed[chd] = true;
      }
    }

  }

}

void ModelRunner::correct_extras(int dep, double& extra_wat, double& extra_sed, std::vector<double>& extra_lab)
{
  extra_wat = std::max(this->graph.depression_tree.volume_water[dep] - this->graph.depression_tree.volume_max_with_evaporation[dep], 0.);
  extra_sed = std::max(this->graph.depression_tree.volume_sed[dep] - this->graph.depression_tree.volume[dep], 0.);
  extra_lab = this->graph.depression_tree.label_prop[dep];
}

void ModelRunner::process_dep(int dep, double& extra_sed, std::vector<double>& extra_lab, bool does_outlet, double& extra_wat)
{
  // std::cout << std::endl ;
  std::cout << "Actually processing " << dep << " Master dep is " << this->graph.depression_tree.get_ultimate_parent(dep) << std::endl;
  std::cout << "pit is " << this->graph.depression_tree.pitnode[dep] << std::endl;
  std::cout << "outlet is " << this->graph.depression_tree.tippingnode[dep] << std::endl;
  std::cout << "externode is " << this->graph.depression_tree.externode[dep] << std::endl;
  std::cout  << " which is processed?? " << this->is_processed[this->graph.depression_tree.externode[dep]] << std::endl;



  // Mark the depression as active -> will count in the mass balance calculations and the inherited water/topo
  this->graph.depression_tree.active[dep] = true;

  this->correct_extras(dep, extra_wat, extra_sed, extra_lab);


  // Check whether is does outlet or is not filled to the top (the algorithm are radically different)
  if(extra_wat > 0 || extra_sed > 0)
  {

    //The lake outflows-> filling it will be easy, reprocessing the downstream lakes/nodes won't.
    std::cout << std::endl << "OUTFLOWS" << std::endl;
    // std::cout << "original pot of water " << this->graph.depression_tree.volume_water[dep] << std::endl;

    // std::cout << "BEEF::" << outsed[dep] << " vs " << this->graph.depression_tree.volume_sed[dep] - this->graph.depression_tree.volume[dep] << std::endl;

    // std::cout << "Wat in dep " << this->graph.depression_tree.volume_water[dep] << std::endl;

    // First I need to fill the water to the top:
    // Put the topo to the max water height of the lake, label all nodes as being part of the lake, ..
    if(extra_wat > 0)
    {
      std::cout << "FILLED2TOP" <<std::endl;
      this->fill_lake_to_top(dep);
    }
    else
    {
      std::cout << "UNDERFILLED" <<std::endl;
      this->fill_underfilled_lake(dep);
    }

    // std::cout << "After_fill pot of water " << this->graph.depression_tree.volume_water[dep] << " + " << extra_wat << " = " << this->graph.depression_tree.volume_water[dep] + extra_wat << std::endl;


    // Important step: defluvialisation
    // actiually the name is not optimal I should rethink it
    // It cancels all the erosion/depostion made by other processes in the lake
    // It back-calculates this->graph.depression_tree.volume_sed and the extra sed and lab in place
    this->defluvialise_lake(dep, extra_sed, extra_lab);

    this->correct_extras(dep,extra_wat, extra_sed, extra_lab);

    // Geting the outlet node ID
    int outlet = this->graph.depression_tree.tippingnode[dep];

    // getting the local sediment flux (ie what has been locally eroded/deposited)
    double locsedflux = this->chonk_network[outlet].get_local_sedflux(this->timestep, this->cellarea);
    double globasedflux = this->chonk_network[outlet].get_sediment_flux();


    // In some rare cases outlet is not processed (deprecated I believe, it was because of flat surfaces, I leave it there for legacy)
    if(this->is_processed[outlet] == false)
    {
      std::cout << "Warning::outlet was not processed at lake time." << std::endl;
      this->process_node_nolake_for_sure(outlet, this->is_processed, this->active_nodes, this->cellarea, this->topography, true, true);
    }
      
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
        // std::cout << "Rec " << rec[i] << " of outlet " << outlet << " is proc?  " << this->is_processed[rec[i]] << std::endl;
        
        // if(wwf[i]>0)
        //   std::cout << "Wat2add2out" << std::endl;
        
        water2add += wwf[i] * this->chonk_network[outlet].get_water_flux() * this->timestep;
        // sed2remove_only_outlet -= wws[i] * locsedflux;
      }
    }
    
    // Putting the outlet reprocessing info into the extrasedwatlab stuff
    std::cout << sed2remove << " <- sed2remove" << std::endl;
    this->graph.depression_tree.add_sediment(sed2remove, this->chonk_network[outlet].get_label_tracker(), dep);

    // extra_lab = mix_two_proportions(extra_sed, extra_lab, sed2remove, this->chonk_network[outlet].get_label_tracker());
    // extra_sed += sed2remove;
    
    // if(extra_wat > 0)
    //   extra_wat += water2add;
    double wat2add2outletonly = water2add;


    // std::cout << "After_fill pot of water " << this->graph.depression_tree.volume_water[dep] << " + " << extra_wat << " = "
    //  << this->graph.depression_tree.volume_water[dep] + extra_wat << " and " << water2add << " should be from the outlet spill out" << std::endl;


    // if(this->node_in_lake[this->graph.depression_tree.externode[dep]] > -1)
    //   extra_wat -= 10000000;

    // this reputs in the node the amount of sediment that were given to the outlet by donors (not in lake)
    std::vector<double> sed2add2outletonly_lab = this->chonk_network[outlet].get_label_tracker();// mix_two_proportions(extra_sed, extra_lab, this->chonk_network[outlet].get_sediment_flux() - locsedflux, this->chonk_network[outlet].get_label_tracker());
    double sed2add2outletonly = this->chonk_network[outlet].get_sediment_flux() - locsedflux;

    // // if my new extra amount of sediment is bellow 0 -> this amount needs now to be taken from the lake itself
    // if(extra_sed < 0)
    // {
    //   this->graph.depression_tree.add_sediment(extra_sed, extra_lab, dep);
    //   extra_sed = 0;
    // }
    this->graph.depression_tree.add_sediment(sed2add2outletonly, this->chonk_network[outlet].get_label_tracker(), dep);

    this->correct_extras(dep, extra_wat, extra_sed, extra_lab);

    // now only I can correct the rest
    // std::cout << "sed2add2outletonly->" << sed2add2outletonly << std::endl;
    extra_wat += wat2add2outletonly;
    // extra_lab = mix_two_proportions(extra_sed, extra_lab,sed2add2outletonly,sed2add2outletonly_lab);
    // extra_sed += sed2add2outletonly;
    

    if(extra_sed < 0)
      extra_sed = 0;
    
    if(extra_wat > 0)
    {
      std::cout << "extra_wat::wat::" << extra_wat <<"::extra_sed::" << extra_sed << std::endl;
      // Ready to reproc the outlet:
      // #1 cancel what it use to give to its receivers
      // std::cout << "427 beef " << this->chonk_network[427].get_water_flux() << std::endl;
      this->chonk_network[outlet].cancel_split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->timestep);
      // std::cout << "427 aft " << this->chonk_network[427].get_water_flux() << std::endl;
      // #2 reset the receiver
      this->chonk_network[outlet].reset();
      // #3 Force it to give water to the external node and not back to lake
      this->chonk_network[outlet].external_moving_prep({this->graph.depression_tree.externode[dep]},
       {1.}, {1.}, {(this->topography[outlet] - this->topography[this->graph.depression_tree.externode[dep]])/this->dx});
      // #4 Relabel the node as not-processed
      // if(this->is_processed[outlet])
      this->is_processed[outlet] = false;
      // #5 manually set the sed/water fluxes to what has been calculated above 
      this->chonk_network[outlet].set_sediment_flux(extra_sed, extra_lab, 1.);
      this->chonk_network[outlet].set_water_flux(extra_wat/this->timestep);


      // std::cout << std::endl << "need_fluxbeef? " << need_fluxbeef << std::endl;
      // std::cout << "WBEEF::" << this->chonk_network[outlet].get_water_flux() << " " << this->chonk_network[outlet].get_sediment_flux() << std::endl;

      // #6 and finally reprocess the node (no move prep as it is forced, no preflux as already included in the outlet calculation)
      this->process_node_nolake_for_sure(outlet, this->is_processed, this->active_nodes, this->cellarea, this->topography, false, false);
      // std::cout << "WBaFt::" << this->chonk_network[outlet].get_water_flux() << " " << this->chonk_network[outlet].get_sediment_flux() << std::endl;
    
    }
    else if (extra_sed > 0)
    {
      std::cout << "extra_sed" << std::endl;
      this->chonk_network[this->graph.depression_tree.externode[dep]].add_to_sediment_flux(extra_sed, extra_lab, 1.);
    }


  }
  else
  {
    std::cout << "DOES NOT OUTFLOWS" << std::endl;


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

void ModelRunner::fill_lake_to_top(int dep)
{
  
  // the water height of this lake is max, as the lake is full
  this->graph.depression_tree.hw[dep] = this->graph.depression_tree.hw_max[dep];

  // getting the outlet or "tipping node"
  int outlet = this->graph.depression_tree.tippingnode[dep];

  // // As my lake is full, my actual amount of volume_water is the max I can accomodate. The rest being stored into the out container
  // if(this->graph.depression_tree.volume_water[dep] > this->graph.depression_tree.volume_max_with_evaporation[dep])
  //   this->graph.depression_tree.volume_water[dep] = this->graph.depression_tree.volume_max_with_evaporation[dep];

  // // Same for the sed
  // if(this->graph.depression_tree.volume_sed[dep] > this->graph.depression_tree.volume[dep])
  //   this->graph.depression_tree.volume_sed[dep] = this->graph.depression_tree.volume[dep];

  //going through each nodes of the lake to label them and adjust the topography
  for(auto n:this->graph.depression_tree.get_all_nodes(dep))
  {
    // Technically, the outlet is not in the lake, jsut a special river
    if(n == outlet)
      continue;

    // std::cout << "Raising " << n << " to " << this->graph.depression_tree.hw[dep] << " in dep " << dep << std::endl;

    // node in dat lake yo
    this->node_in_lake[n] = dep;
    // topo to hw
    this->topography[n] = this->graph.depression_tree.hw[dep];
    // and if lake evaporation, calculating the local amount
    if(this->lake_evaporation)
    {
      this->graph.depression_tree.actual_amount_of_evaporation[dep] += this->lake_evaporation_rate_spatial[n] * this->cellarea * this->timestep;
    }
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
  // -> keeping track of nodes
  int last_node = 0;
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
        std::cout << "Negative ratio" << std::endl;
        ratio = 0;
        this->graph.depression_tree.actual_amount_of_evaporation[dep] += remaining_volume;
      }

      this->graph.depression_tree.hw[dep] = this_elev + ratio * deltelev;
      
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
  std::cout << "DEBUG:: sum_DV:" <<sum_DV << " sum_erV:" << sum_erV << std::endl;

  // extra_sed_prop = mix_two_proportions(extra_sed, extra_sed_prop, -1 * defluvialisation_of_sed, this->chonk_network[outlet].get_label_tracker());
  // extra_sed -= defluvialisation_of_sed;

  // double delta_delta = extra_sed - original_extra_sed;

  // if(extra_sed < 0)
  // {
  //   std::cout << "Extra_sed removing stuff to lake directly::" << extra_sed << std::endl;
  this->graph.depression_tree.add_sediment(-1 * defluvialisation_of_sed, extra_sed_prop, dep);
  //   extra_sed = 0;
  // }

}

void ModelRunner::lake_solver_v3(int node)
{

  // First I am getting hte depression
  int this_dep = this->node_in_lake[node];
  
  // Deug statement -> to remove once efixed
  std::cout << "lake_solver_v3 -> starting " << this_dep << std::endl;

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


  // Getting all nodes of the depression and sorting them accordingly (Probs deprecated)
  std::vector<int> lstack = this->graph.depression_tree.get_all_nodes(master_dep);
  std::vector<int> nodes; nodes.reserve(lstack.size());

  std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > > PQstack;
  for(auto n: lstack)
  {
    PQstack.emplace(PQ_helper<int,int>(n,this->graph.get_index_MF_stack_at_i(n)));
  }

  while(PQstack.empty() == false)
  {
    nodes.emplace_back(PQstack.top().node);
    PQstack.pop();
  }


  // Checking that all nodes of that depression have been processed, and processing them in case (Probs deprecate dnow)
  for(size_t i = 0; i < nodes.size(); i++)
  {
    int tnode = nodes[i];
    if(this->is_processed[tnode] == true)
      continue;

    this->process_node_nolake_for_sure(tnode, this->is_processed, this->active_nodes, this->cellarea, this->topography, true, true);
  }


  // Get the topological order of depressions - > depressions from bottom to top
  std::vector<int> treestak = this->graph.depression_tree.get_local_treestack(master_dep);
  
  // Getting ready to gather the water and sediment fluxes I will have for that depression
  // Water for the whole system
  double tot_water_volume = 0;
  // sed for the whole system
  double tot_sed_volume = 0;
  // representative chonk for the whole system (I am lazily managing label prop and tracking here)
  chonk tot_representative_chonk(-1,-1,false);

  // Same as above but by local depression
  std::map<int,double> water_volume;
  std::map<int,double> sed_volume;
  std::map<int,bool> is_processed;
  std::map<int,chonk> representative_chonk;


  // Going a first time throught the local stack to initialise everything
  for(int i=0; i<int(treestak.size());i++)
  {

    int this_dep = treestak[i];
    this->node_in_lake[this->graph.depression_tree.pitnode[this_dep]] = -1;
    water_volume[this_dep] = 0;
    sed_volume[this_dep] = 0;
    representative_chonk[this_dep] = chonk(-1,-1,false);
    is_processed[this_dep] = false;;;;;
  }

  // Gathering all the water/sed/prop of the depression and propagating it to the top
  for(int i=0; i<int(treestak.size());i++)
  {
    // local dep ID
    int this_dep = treestak[i];
    // Only gathering if level 0 (will spread upstream if needed later)
    if(this->graph.depression_tree.level[this_dep] == 0)
    {
      // Node ID of the local pit
      int pit = this->graph.depression_tree.pitnode[this_dep];

      // DEBUG CHECKER TO REMOVE IF NOT TRIGGERD DURING EXTENSIVE TESTS
      if(this->is_processed[pit] == false)
        throw std::runtime_error("PIT NOT PROC");

      // gathering local water/sed/prop and adding to the top and...
      tot_water_volume += this->chonk_network[pit].get_water_flux() * this->timestep;
      tot_sed_volume += this->chonk_network[pit].get_sediment_flux();
      tot_representative_chonk.add_to_sediment_flux(
        this->chonk_network[pit].get_sediment_flux(), 
        this->chonk_network[pit].get_label_tracker(),
        this->chonk_network[pit].get_fluvialprop_sedflux()
      );

      // ... in the local depressions
      water_volume[this_dep] += this->chonk_network[pit].get_water_flux() * this->timestep;
      sed_volume[this_dep] += this->chonk_network[pit].get_sediment_flux();
      representative_chonk[this_dep].add_to_sediment_flux(
        this->chonk_network[pit].get_sediment_flux(), 
        this->chonk_network[pit].get_label_tracker(),
        this->chonk_network[pit].get_fluvialprop_sedflux()
      );



    }


    // Propagating the water and sed up if there is a parent
    int parent = this->graph.depression_tree.parentree[this_dep];
    if(parent == -1)
      continue;

    water_volume[parent] += water_volume[this_dep];
    sed_volume[parent] += sed_volume[this_dep];
    representative_chonk[parent].add_to_sediment_flux(
      sed_volume[this_dep], 
      representative_chonk[this_dep].get_label_tracker(),
      representative_chonk[this_dep].get_fluvialprop_sedflux()
    );

  }

  // Done with the gathering phase
  // Let's fill the depression for real now

  // Calculating which dep to reproc -> not all depression will need processing
  std::vector<int> local_deps2reproc;

  // initialising the tree to not done
  std::map<int,bool> treestak_done_question_mark;
  std::map<int,bool> treestak_in_localocalocal;
  std::map<int,bool> outlets2twin;
  for (auto dep : treestak){treestak_done_question_mark[dep] = false; treestak_in_localocalocal[dep] = false;outlets2twin[dep] = false;};
  // no data depressions are marked as processed
  treestak_done_question_mark[-1] = true;

  // Going from top to bottom in the local binary tree
  for(int i = int(treestak.size()) - 1; i >= 0; i--)
  {
    // getting dep ID as well as its potential twin
    int tdep = treestak[i];
    int twin = this->graph.depression_tree.get_twin(tdep);
    int save_twiny = twin;

    // counting the number of dep outflowing and saving its ID (only used in case there is only one outflowing)
    short n_outflowing = 0;
    int which_one_outflows = -1;

    // going through the twin depressions
    for( auto dep:{tdep,twin})
    {
      // Is dep proc or -1?
      if(treestak_done_question_mark[dep] == true)
        continue;

      // Now it will be
      treestak_done_question_mark[dep] = true;

      // This calculate the minimum amount of water required to enter this lake, based on the max volume accumulatable in the Children
      // It does take account of lake evaporation when needed
      double lowerboundvol = this->graph.depression_tree.get_volume_of_children_water(dep);

      // Do I have enough water to fill the current master_depression
      if(lowerboundvol <= water_volume[dep])
      {

        // Checking and recording cases where my local depression outflows
        // This is in case one of my local dep outflows but not its twin (if exists)
        if(this->graph.depression_tree.volume_max_with_evaporation[dep] <= water_volume[dep])
        {
          n_outflowing++;
          which_one_outflows = dep;
        }

        // his dep is marked to be filled
        // local_deps2reproc.emplace_back(dep);
        treestak_in_localocalocal[dep] = true;

        // And I am therefore marking all its children to processed and not to be filled a second/third/... time
        auto doz_children = this->graph.depression_tree.get_all_children(dep);
        for (auto ch:doz_children)
          treestak_done_question_mark[ch] = true;

      }
    }

    // Last check: only one outflowing?? redistributing the water to the second one 
    // (which itself will not outflow otherwise the parent would have already been done)
    // Also if this is a single dep, I do not outflow
    if(n_outflowing != 1 || twin == -1)
    {
      if(treestak_in_localocalocal[tdep])
        local_deps2reproc.emplace_back(tdep);
      continue;
    }
    else
    {
      local_deps2reproc.emplace_back(which_one_outflows);      
    }
    outlets2twin[which_one_outflows] = true;

    std::cout<< std::endl << "HAPPPPEEEEENNNNNSSSSS" << std::endl;

    // this is the hardest part, thanksfully does not happens much
    // Getting the twin of the depression which outflowed
    twin = this->graph.depression_tree.get_twin(which_one_outflows);

    // Getting the extra water/sed and transferring them to transfer
    double extra_water = water_volume[which_one_outflows] - this->graph.depression_tree.volume_max_with_evaporation[which_one_outflows];   
    double extra_sed = sed_volume[which_one_outflows] - this->graph.depression_tree.volume[which_one_outflows];

    // Transferring the water from Otwin to Utwin
    water_volume[which_one_outflows] -= extra_water;
    water_volume[twin] += extra_water;

    // transferring sediments if and only if I have enough of them
    if(extra_sed > 0)
    {
      sed_volume[which_one_outflows] -= extra_sed; 
      sed_volume[twin] += extra_sed; 
      representative_chonk[twin].add_to_sediment_flux(
        sed_volume[which_one_outflows], 
        representative_chonk[which_one_outflows].get_label_tracker(),
        representative_chonk[which_one_outflows].get_fluvialprop_sedflux()
      );
    }


    // tricky part:
    // If my depression is not registered as TO PROCESS
    // It can happen when a twin dep overflows, but its twin did not have enough water so its children will be processed instead
    // Well here, we want to reassess the situation with 
    if(treestak_in_localocalocal[twin] == false)
    {
      // Calculating the minimum amount of water required to fill this depression
      double lowerboundvol = this->graph.depression_tree.get_volume_of_children_water(twin);

      // treestak_in_localocalocal[twin] = true;

      // Do I have enough water to fill the new depression
      if(lowerboundvol <= water_volume[twin])
      {
        if(this->graph.depression_tree.volume_max_with_evaporation[twin] <= water_volume[twin])
        {
          n_outflowing++;
          which_one_outflows = twin;
        }

        // if yes, this dep is marked to be filled
        local_deps2reproc.emplace_back(twin);
        // And I am therefore marking all its children to processed
        auto doz_children = this->graph.depression_tree.get_all_children(twin);
        for (auto ch:doz_children)
          treestak_done_question_mark[ch] = true;
      }
      else
      {

        // If I arrive here, my dep got more water but yet does not have enough to fill the minimum level. Need to propagate water to children
        int origin_dep = twin;
        // By default getting the first child
        int achild = this->graph.depression_tree.treeceivers[twin][0];
        // WHile I found children, propagate to the first child
        while(achild != -1)
        {
          if(extra_sed > 0)
          {
            sed_volume[achild] += extra_sed; 
            representative_chonk[achild].add_to_sediment_flux(
              sed_volume[which_one_outflows], 
              representative_chonk[which_one_outflows].get_label_tracker(),
              representative_chonk[which_one_outflows].get_fluvialprop_sedflux()
            );
          }
          water_volume[achild] += extra_water;

          achild = this->graph.depression_tree.treeceivers[achild][0];

        }

      }
      // DOne with rechecking the new dep
    }
    else
      local_deps2reproc.emplace_back(twin);


    // DOne with navigating the tree

  }


  // Now ready to fill the depression that need it

  double totwatchecekr = 0;
  std::vector<int> counter_of_dep(local_deps2reproc.size(), 0);

  for(auto dep: local_deps2reproc)
  {
    std::cout << "Actually processing " << dep << std::endl;

    // I dont remember what is this but I guess it is a bug checker
    totwatchecekr += water_volume[dep];

    // Keeping memory of this lake for the finilising process
    this->lake_to_process.push_back(dep);

    // Activate the lake for deposition
    this->graph.depression_tree.active[dep] = true;
    // finding the outlet
    int outlet = this->graph.depression_tree.tippingnode[dep];
    // And their receiver
    int receiver_extern = this->graph.depression_tree.externode[dep];
    // getting all children depressions (and self)
    std::vector<int> children = this->graph.depression_tree.get_all_children(dep, true);

    // Before reprocessing the outlet, I need to reprocess everything downstream of it
    // reprocessing the outlet and the downstream nodes
    // Initialising a priority queue sorting the nodes to reprocess by their id in the Mstack. the smallest Ids should come first
    // Because I am only processing nodes in between 2 discrete lakes, it should not be a problem
    std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> > ORDEEEEEER; // I am not politically aligned with John Bercow, this is jsut for the joke haha


    //----------------------------------------------------
    //------- FORMING THE STACK OF NODE TO REPROC --------
    //----------------------------------------------------
    // This section aims to gather all nodes to reprocess. All node of the landscape have a type sort them by type: 
    // # 'n' if not concerned, 'l' if in lake, 'd' if donor, 'r' if potential reproc and 'y' if in stack
    std::vector<int> local_mstack;
    // keeping track of which nodes are already in teh queue (or more generally to avoid) and which one have receivers in the local stack
    std::vector<char> is_in_queue(this->io_int["n_elements"],'n');
    // Before starting to gather the nodes downstream of me outlet, let's gather mark the node in me lake as not available (as if already in the queue)
    // Avoinding nodes in the lakes
    lstack = this->graph.depression_tree.get_all_nodes(dep);
    std::vector<int>nodes; nodes.reserve(lstack.size());

    PQstack = std::priority_queue< PQ_helper<int,int>, std::vector<PQ_helper<int,int> >, std::greater<PQ_helper<int,int> > >() ;
    for(auto n: lstack)
      PQstack.emplace(PQ_helper<int,int>(n,this->graph.get_index_MF_stack_at_i(n)));

    while(PQstack.empty() == false)
    {
      nodes.emplace_back(PQstack.top().node);
      PQstack.pop();
    }


    bool need_to_remove_lakevap = false;
    // TEMPORARY MEASURE, ASSUMING hw = hw max
    // So, First checking if I can fill the dep entirely or partially.
    if(water_volume[dep] >= this->graph.depression_tree.volume_max_with_evaporation[dep] ||  double_equals(water_volume[dep] - this->graph.depression_tree.volume_max_with_evaporation[dep],0., 1e-5) )
    {
      std::cout << "A" << std::endl;
      // Yes, is simple - > water to max and volume to max
      this->graph.depression_tree.volume_water[dep] = this->graph.depression_tree.volume_max_with_evaporation[dep];
      this->graph.depression_tree.hw[dep] = this->graph.depression_tree.hw_max[dep]; 
      // making sure nodes are maked as lake
      double this_laevapop = 0;
      for (auto gh:this->graph.depression_tree.get_all_nodes(dep))
      {
        if(gh != outlet)
        {
          is_in_queue[gh] = 'l'; 
          if(this->lake_evaporation)
            this_laevapop += this->lake_evaporation_rate_spatial[gh] * this->cellarea * this->timestep;
        }

      }

      this->graph.depression_tree.actual_amount_of_evaporation[dep] += this_laevapop;

    }
    else
    {
      std::cout << "B" << std::endl;
      need_to_remove_lakevap = true;
      // No -> less simple

      //all water goes to dep 
      this->graph.depression_tree.volume_water[dep] = water_volume[dep];

      // gathering nodes and sorting them by elevation for the partial filling
      auto tnodes = this->graph.depression_tree.get_all_nodes(dep);
      std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > Sorter;
      bool checekr_switch = false;
      for(auto tn: tnodes)
      {
          Sorter.emplace(PQ_helper<int,double>(tn, this->surface_elevation[tn]));
      }

      // And filling the depression
      int n_nodes = 0;
      std::vector<int> nodes2topogy;
      bool is_changed = false;
      int last_node = 0, last_top_node = 0;
      double remaining_volume = water_volume[dep];
      double cumul_V = 0;
      while(Sorter.size() > 0)
      {
        // Getting node and elev
        int this_node = Sorter.top().node;
        double this_elev = Sorter.top().score;
        Sorter.pop();

        // Getting node and elev of the next node
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
        last_top_node = top_node;

        // Incrementing the counter of nodes
        n_nodes++;

        // while not the outlet, registering it
        if(this_node != outlet)
        {
          nodes2topogy.push_back(this_node);
          is_in_queue[this_node] = 'l';
          this->node_in_lake[this_node] = dep;
        }

        double lcoal_evaporation = 0;
        if(this->lake_evaporation)
        {
          lcoal_evaporation = this->lake_evaporation_rate_spatial[this_node] * this->cellarea * this->timestep;
        }

        remaining_volume -= lcoal_evaporation;
        this->graph.depression_tree.actual_amount_of_evaporation[dep] += lcoal_evaporation;

        // Calculating the incrementing volume
        double deltelev = top_elev - this_elev;
        double dV = n_nodes * this->cellarea * deltelev;
        if(remaining_volume > dV)
        {
          remaining_volume -= dV;
          cumul_V += dV;
        }
        else
        {
          is_changed = true;
          double ratio = remaining_volume / dV;

          // At the moment the lake evaporation is approximated at a pixel pret:
          // If I fall in between 2 nodes, I assume it stick at the top of the last node and backcalculate the evaporation as "in-between"
          // this could eventually be rethought a bit more accurately, although it is a detail so we'll leave it to a next publication refining the lake evaporation method

          if(ratio < 0)
          {
            ratio = 0;
            this->graph.depression_tree.actual_amount_of_evaporation[dep] += remaining_volume;
          }
          this->graph.depression_tree.hw[dep] = this_elev + ratio * deltelev;
          for(auto nj:nodes2topogy)
            this->topography[nj] = this->graph.depression_tree.hw[dep];
          break;
        }
      }

      // this->graph.depression_tree.volume_water[dep] -= this->graph.depression_tree.actual_amount_of_evaporation[dep];
      // water_volume[dep] -= this->graph.depression_tree.actual_amount_of_evaporation[dep];;

    }

    // Checkwhether it outflows
    bool outflows = (water_volume[dep] > this->graph.depression_tree.volume_max_with_evaporation[dep]);
    double wat2remove = 0, wat2add = 0;

    // if it does, I need to reprocess the unprocessed nodes bellow
    if(outflows)
    {
      //----------------------------------------------------
      //----------- Special needs for an outlet ------------
      //----------------------------------------------------

      // Copying the move status to get rec info
      double sed2remove = 0; std::vector<double> labrador;
      std::vector<int> rec; std::vector<double> wwf; std::vector<double> wws; std::vector<double> strec;
      this->chonk_network[outlet].copy_moving_prep(rec,wwf,wws,strec);
      for(size_t r=0; r<rec.size(); r++)
      {
        int trec = rec[r];
        if(is_in_queue[trec] == 'l')
        {
          double tadd = wws[r] * this->chonk_network[outlet].get_sediment_flux();
          labrador = mix_two_proportions(tadd,this->chonk_network[outlet].get_label_tracker(), sed2remove, labrador);
          sed2remove += tadd;
          wat2remove += wwf[r] * this->chonk_network[outlet].get_water_flux() * this->timestep;
          // this->graph.depression_tree.print_all_lakes_from_node(trec);
        }
        else
        {
          wat2add += wwf[r] * this->chonk_network[outlet].get_water_flux() * this->timestep;
        }
      }

      representative_chonk[dep].add_to_sediment_flux(sed2remove,labrador,1.);
      double was = sed_volume[dep];
      sed_volume[dep] -= sed2remove;
  
      if(sed_volume[dep] < 0)
      {
        std::cout << "Dep: " << dep << " Sedvol = " << sed_volume[dep] << " was removed: "  << sed2remove << " dep_volume is " << this->graph.depression_tree.volume[dep] << std::endl;
        std::cout << "N Nodes in dep: " << this->graph.depression_tree.get_all_nodes(dep).size() << std::endl;
        throw std::runtime_error("Dep: " + std::to_string(dep) + "sed_volume[dep] < 0 :: " + std::to_string(sed_volume[dep]) + " was " + std::to_string(was));
      }
      
      if(this->graph.depression_tree.volume_sed[dep] < 0)
        throw std::runtime_error("Dep: " + std::to_string(dep) + "depression_tree < 0");

      
      // This function gathers all the nodes to be reprocessed. Including their donor which will be partially reprocessed
      // This function is only geometrical, it does not assume any existing transfer of sed/water
      this->gather_nodes_to_reproc(local_mstack,  ORDEEEEEER,  is_in_queue,  outlet);
      this->deprocess_local_stack(local_mstack, is_in_queue, outlet);


    }



    // Need to "cancel" erosion and deposition to backcalculate the sediment in the lake

    // Now I am post-processing the lakes: adding the lake depth, and ID and all

    double defluvialisation_of_sed = 0;
    std::vector<double> defluvialisation_of_sed_label_edition(this->n_labels,0.);
    for (auto n:this->graph.depression_tree.get_all_nodes(dep))
    {
      // std::cout << n << "-" << this->graph.depression_tree.node2tree[n] << "-";
      if(is_in_queue[n] == 'l')
      {
        if(n ==  outlet)
          continue;

        // if(this->graph.depression_tree.potential_volume[n] <= this->graph.depression_tree.volume_water[dep])
        if(true)
        {
          this->topography[n] = this->graph.depression_tree.hw[dep];
          // if(this->node_in_lake[n] != -1)
          //   throw std::runtime_error("Node already in lake");
          this->node_in_lake[n] = dep;

          // HERE I WILL NEED TO REMOVE THE SED FROM EROSION?DEP WITH THE RIGHT PROPORTIONS
          double tsed = 0;

          // REMOVING EROSION OF BEDROCK
          tsed = this->chonk_network[n].get_erosion_flux_only_bedrock() * this->timestep * this->cellarea;
          std::vector<double> temp(this->n_labels,0.); temp[this->label_array[n]] = 1.;
          defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
          defluvialisation_of_sed += tsed;
          // REMOVING EROSION OF SEDIMENT LAYER
          // if(this->is_there_sed_here[n])
          if(true)
          {
            tsed = this->chonk_network[n].get_erosion_flux_only_sediments() * this->timestep * this->cellarea;
            if(this->sed_prop_by_label[n].size() > 0)
              temp = this->sed_prop_by_label[n][this->sed_prop_by_label[n].size() - 1];
            else
              temp = this->chonk_network[n].get_label_tracker();

            defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
            defluvialisation_of_sed += tsed;
          }
          // REMOVING DEPOSITION
          tsed = -1 * this->chonk_network[n].get_deposition_flux() * this->timestep * this->cellarea;
          temp = this->chonk_network[n].get_label_tracker();
          defluvialisation_of_sed_label_edition = mix_two_proportions(tsed,temp,defluvialisation_of_sed,defluvialisation_of_sed_label_edition);
          defluvialisation_of_sed += tsed;

          // And finally resetting the sediment flux of the chonk
          this->chonk_network[n].reset_sed_fluxes();
        }
      }
    }


    representative_chonk[dep].add_to_sediment_flux(-1 * defluvialisation_of_sed,defluvialisation_of_sed_label_edition,1.);
    sed_volume[dep] -= defluvialisation_of_sed;

    
    if(sed_volume[dep] < this->graph.depression_tree.volume[dep])
      this->graph.depression_tree.volume_sed[dep] = sed_volume[dep];
    else
      this->graph.depression_tree.volume_sed[dep] = this->graph.depression_tree.volume[dep] ;
    
    this->graph.depression_tree.label_prop[dep] = representative_chonk[dep].get_label_tracker();


    // transmitting eventual Q to the outlet
    double extra_sed = sed_volume[dep] - this->graph.depression_tree.volume_sed[dep];
    double extra_wat = 0;
    if(need_to_remove_lakevap)
      extra_wat = (water_volume[dep] - 
                        this->graph.depression_tree.volume_water[dep] 
                        - this->graph.depression_tree.actual_amount_of_evaporation[dep] + wat2add)
                        /this->timestep;
    else
      extra_wat = (water_volume[dep] - 
                        this->graph.depression_tree.volume_water[dep] + wat2add)
                        /this->timestep;

    if(extra_sed < 0)
      extra_sed = 0;

    if(extra_wat < 0)
    {
      outflows = false;
      extra_wat = 0;
    }

    // Now reprocessing the stack of stuff

    // Iterating through the local stack
    std::cout << "OUTLET IS:" << outlet << std::endl;
    for(auto tnode:local_mstack)
    {
      // std::cout << "REPROCESSING NODES::" << tnode << "--" << this->inherited_water_added[tnode] << " OO " << is_in_queue[tnode]  << std::endl;
      // Discriminating between the donors and the receivers
      if(is_in_queue[tnode] == 'd')
      {
        // # I am a donor, I just need to regive the sed/water without other reproc
        // # Let me just just check which of my receivers I need to give fluxes (I shall not regive to the nodes not in my local stack)
        std::vector<int> ignore_some; 
        std::vector<int> rec; std::vector<double> wwf; std::vector<double> wws; std::vector<double> strec;
        this->chonk_network[tnode].copy_moving_prep(rec,wwf,wws,strec);
        for(size_t cat =0; cat < rec.size(); cat++)
        {
          int ttnode = rec[cat];
          if(ttnode == outlet)
          {
            std::cout << "OUTLETFLUX::" << this->chonk_network[outlet].get_sediment_flux() << std::endl;
            this->chonk_network[outlet].add_to_sediment_flux(this->chonk_network[tnode].get_sediment_flux() * wws[cat],
             this->chonk_network[tnode].get_label_tracker(), 1.);
            ignore_some.emplace_back(ttnode);
            continue;
          }

          // if the node is the outlet, or not a 'y', I ignore it
          if(is_in_queue[ttnode] != 'y' || ttnode == outlet)
          {          
            ignore_some.emplace_back(ttnode);
            continue;
          }
          // I also ignore the nodes in the current lake (they have a 'y' signature)
          else if (this->node_in_lake[ttnode] >= 0)
          {
            ignore_some.emplace_back(ttnode);
            continue;
          }
        }
        // if(ignore_some.size() == this->chonk_network[tnode].get_chonk_receivers_copy().size())
          // std::cout << "LDKFJKLDFDSFHJKLDF" << std::endl;

        // # Ignore_some has the node i so not want
        // # So I transmit my fluxes to the nodes I do not ignore
        this->chonk_network[tnode].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);

      }
      else
      {
        std::cout << "rep " << tnode << "|";
        // # I am not a nodor:
        if(tnode == outlet && outflows)
        {
          this->is_processed[outlet] = false;
          // external_moving_prep(std::vector<int> rec,std::vector<double> wwf,std::vector<double> wws, std::vector<double> strec)
          std::cout << "Adding " << extra_wat << " to outlet and " << this->graph.depression_tree.volume_water[dep] << 
            " to lake which has a total volume of " << this->graph.depression_tree.volume_max_with_evaporation[dep] << " -> ev= " 
              << this->graph.depression_tree.actual_amount_of_evaporation[dep] << " volume: " <<  this->graph.depression_tree.volume[dep] 
                << " hw_max = " << this->graph.depression_tree.hw_max[dep] << " ztpoint = " << this->surface_elevation[this->graph.depression_tree.tippingnode[dep]] << std::endl; 


          this->chonk_network[outlet].add_to_water_flux(extra_wat);
          this->chonk_network[outlet].add_to_sediment_flux(extra_sed, representative_chonk[dep].get_label_tracker(), 1.);
          this->chonk_network[outlet].external_moving_prep({receiver_extern}, {1.}, {1.}, {(this->topography[outlet] - this->topography[receiver_extern])/this->dx});
          this->process_node_nolake_for_sure(tnode, this->is_processed, this->active_nodes, this->cellarea,this->topography, false, false);
        }
        else
        {
          // throw std::runtime_error("SafetyStoppah #12BC45f");
          this->process_node_nolake_for_sure(tnode, this->is_processed, this->active_nodes, this->cellarea,this->topography, true, true);
        }
        std::cout << "done|";
      }
    }
    std::cout << std::endl;


  }// end of for loop fgoing through the lakes

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
  this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep,  this->topography, 
        this->dx, this->dy, chonk_network);

  // int this_case = intcorrespondance[this->move_method];

  // std::vector<int> rec = this_chonk.get_chonk_receivers_copy();
  // switch(this_case)
  // {
  //   case 2:
  //     this_chonk.move_to_steepest_descent(this->graph, this->timestep,  this->topography, this->dx, this->dy, chonk_network);
  //     break;
  //   case 3:
  //     this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, 
  //  this->topography, this->dx, this->dy, chonk_network);
  //     break;
  //   case 4:
  //     this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep,  this->topography, 
  //       this->dx, this->dy, chonk_network);
  //     break;
      
  //   default:
  //     std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  // }

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


  double this_Kr;
  double this_Ks;
  double this_kappar;
  double this_kappas;
  double S_c;
  this->manage_K_kappa(label_id, this_chonk, this_Kr, this_Ks, this_kappar, this_kappas, S_c);

  if(this->CHARLIE_I)
  {

    // double this_Kr = this->labelz_list[label_id].Kr_modifyer * this->labelz_list[label_id].base_K;
    // double this_Ks = this->labelz_list[label_id].Ks_modifyer * this->labelz_list[label_id].base_K;
    // std::cout << 
    this_chonk.charlie_I(this->labelz_list[label_id].n, this->labelz_list[label_id].m, this_Kr, this_Ks,
    this->labelz_list[label_id].dimless_roughness, this->sed_height[index], 
    this->labelz_list[label_id].V, this->labelz_list[label_id].dstar, this->labelz_list[label_id].threshold_incision, 
    this->labelz_list[label_id].threshold_entrainment,label_id, these_sed_props, this->timestep,  this->dx, this->dy);
  }

  // Hillslope routine
  if(this->CIDRE_HS)
  {

    // double this_kappas = this->labelz_list[label_id].kappa_s_mod * this->labelz_list[label_id].kappa_base;
    // double this_kappar = this->labelz_list[label_id].kappa_r_mod * this->labelz_list[label_id].kappa_base;
    // std::cout << "kappe_r is " << this_kappar << " and kappa_s is " << this_kappas << " Sc = " << this->labelz_list[label_id].critical_slope << std::endl;

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
      // sumwat += this->io_double_array["lake_depth"][node] * this->dx * this->dy / this->timestep ;
    }
    // std::cout << "INHERITED WATER AT " << minnodor << " : " << this->graph.depression_tree.volume_water[tlake]/this->timestep << std::endl;
    this->chonk_network[minnodor].add_to_water_flux((this->graph.depression_tree.volume_water[tlake] - this->graph.depression_tree.actual_amount_of_evaporation[tlake])/this->timestep);
    this->inherited_water_added[minnodor] += (this->graph.depression_tree.volume_water[tlake] - this->graph.depression_tree.actual_amount_of_evaporation[tlake])/this->timestep;
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

    double totsed = 0;
    double totwat = 0;
    double ratio_of_dep = this->graph.depression_tree.volume_sed[i]/(this->graph.depression_tree.volume_water[i] - this->graph.depression_tree.actual_amount_of_evaporation[i]);
    if(this->graph.depression_tree.volume_water[i] - this->graph.depression_tree.actual_amount_of_evaporation[i] < this->graph.depression_tree.volume_sed[i])
    {
      
      if(this->graph.depression_tree.volume_sed[i] > 0)
      {
        std::cout << i << " Draping only sed "<< std::endl;
        this->drape_dep_only_sed(i);
      
      }

      continue;
    }
    std::cout << i << " Draping f_water "<< std::endl;

    std::cout << "Gougnge:" <<i << " rat : " << ratio_of_dep << std::endl;

    // NEED TO DEAL WITH THAT BOBO
    // if(ratio_of_dep > 1)
    // {
    //   std::cout << "POSSIBLY MISSING " << this->graph.depression_tree.volume_sed[i] * (ratio_of_dep - 1) << std::endl;
    //   ratio_of_dep = 1;
    // }

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
        continue;

      }
        // throw std::runtime_error("afshdffdaglui;regji;");

      if(this->topography[no] != this->graph.depression_tree.hw[i])
      {

        std::cout <<"DN::" << no << " [] " <<  this->node_in_lake[no] << " || " << this->topography[no] << "||" << this->graph.depression_tree.hw[i] << " || " << this->surface_elevation[no] << " is outlet? " << this->graph.depression_tree.is_outlet(i) << std::endl;
        // throw std::runtime_error("NOT THE RIGHT ELEV");
      }

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
      


      chonk_network[no].add_sediment_creation_flux(slangh);
      chonk_network[no].add_deposition_flux(slangh); // <--- This is solely for balance calculation
      chonk_network[no].set_label_tracker(this->graph.depression_tree.label_prop[i]);
      
      // chonk_network[no].reset_sed_fluxes();
    }


    DEBUG_totsed += totsed;
    // Seems fine here...
    // std::cout << "BALANCE LAKE = " << total << " out of " << this->graph.depression_tree.volume_sed[i] << std::endl;
    // if(double_equals(totsed , this->graph.depression_tree.volume_sed[i], 1) == false || double_equals(totwat + this->graph.depression_tree.actual_amount_of_evaporation[i] , this->graph.depression_tree.volume_water[i],1) == false)
    if(true)
    {
      std::cout << i << " TOT IN SED = " << totsed << " out of " << this->graph.depression_tree.volume_sed[i] << " AND VOL WAS " << this->graph.depression_tree.volume[i] << std::endl;
      std::cout << i << " TOT IN WATER = " << totwat + this->graph.depression_tree.actual_amount_of_evaporation[i] << " out of " << this->graph.depression_tree.volume_water[i] << " AND VOL WAS " << this->graph.depression_tree.volume[i] << std::endl;
      std::cout << i << " hw " << this->graph.depression_tree.hw[i] << " vs max " << this->graph.depression_tree.hw_max[i] << " pitelev " << this->surface_elevation[this->graph.depression_tree.pitnode[i]] << " outelev " << this->surface_elevation[this->graph.depression_tree.tippingnode[i]] << std::endl;
      double totvolmax = 0;
      for(auto no:this->graph.depression_tree.get_all_nodes(i) )
      {
        // std::cout << no << " in lake " << i << " -> " << this->node_in_lake[no] <<  " this->topography[no]" << this->topography[no] << std::endl;
        // std::cout << no << " in lake " << i << " -> " << this->node_in_lake[no] << std::endl;
        totvolmax += (this->topography[no] - this->surface_elevation[no]) * this->cellarea;
      }
      std::cout << "totvolmax is " << totvolmax << std::endl; 
      std::cout << std::endl;

    }
  }

  std::cout << "DEBUG::drape_lake_sed::tototsed=" << DEBUG_totsed << std::endl;

}


void ModelRunner::drape_dep_only_sed(int dep_ID)
{

  double Vs = this->graph.depression_tree.volume_sed[dep_ID];
  if(Vs == 0)
    return;

  // Depressions which have only sed need special treatment
  auto nodes = this->graph.depression_tree.get_all_nodes(dep_ID);

  std::priority_queue< PQ_helper<int,double>, std::vector<PQ_helper<int,double> >, std::greater<PQ_helper<int,double> > > filler;
  
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
      next_hs = this->surface_elevation[this->graph.depression_tree.tippingnode[dep_ID]];
    else
      next_hs = filler.top().score;

    double dz = next_hs - hs;
    double dV = nnodes * this->cellarea * dz;

    double ratio = Vs/dV;
    if(ratio > 1)
      ratio = 1;

    dV = ratio * dV;
    Vs -= dV;
    
    nodes2fill.emplace_back(node);

    hs += ratio * (next_hs - hs);
    // this->chonk_network[no].set_label_tracker(this->graph.depression_tree.label_prop[i]);
  }

  for(auto node:nodes2fill)
  {

    double slangh = (hs - this->surface_elevation[node])/this->timestep;

    if(slangh > 1e3)
      throw std::runtime_error("slangh is > 1e3 in lake draping onlysed");

    if(std::isfinite(slangh) == false)
      throw std::runtime_error("NAN in lake draping"); 

    this->chonk_network[node].add_sediment_creation_flux(slangh);
    this->chonk_network[node].add_deposition_flux(slangh);
    this->chonk_network[node].add_to_sediment_flux(slangh * this->cellarea * this->timestep,this->graph.depression_tree.label_prop[dep_ID], this->chonk_network[node].get_fluvialprop_sedflux());
    // std::cout << "Adding " << slangh << " to " << node << std::endl;
  }


  if(Vs > 0)
    std::cout << "EXTRA SED IN THE FILLING OF SED-ONLY DEP " << Vs << std::endl;

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




#endif

