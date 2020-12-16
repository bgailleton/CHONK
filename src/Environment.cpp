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


// Managing the comparison operators for the fill node, to let the queue know I wanna compare it by elevation values
// lhs/rhs : left hand side, right hand side
bool operator>( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation > rhs.elevation;
}
bool operator<( const nodium& lhs, const nodium& rhs )
{
  return lhs.elevation < rhs.elevation;
}

bool operator>( const node_to_reproc& lhs, const node_to_reproc& rhs )
{
  return lhs.id_in_mstack > rhs.id_in_mstack;
}
bool operator<( const node_to_reproc& lhs, const node_to_reproc& rhs )
{
  return lhs.id_in_mstack < rhs.id_in_mstack;
}

// Small function utilised into the debugging
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

// ######################################################################
// ######################################################################
// ######################################################################
// ###################### Model Runner ##################################
// ######################################################################
// ######################################################################

// Initialises the model object, actually Does not do much but is required.
void ModelRunner::create(double ttimestep, std::vector<std::string> tordered_flux_methods, std::string tmove_method)
{
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
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{

  std::cout << "initiating nodegraph..." <<std::endl;
  // Creating the nodegraph and preprocessing the depression nodes
  this->io_double_array["topography"] = xt::pytensor<double,1>(this->io_double_array["surface_elevation"]);


  // First jsut sorting out the active node array.
  // Needs cleaning and optimisation cause some routines with bool where somehow buggy
  xt::pytensor<bool,1> active_nodes = xt::zeros<bool>({this->io_int_array["active_nodes"].size()});
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];


  // Converting int to bool
  for(size_t i =0; i<inctive_nodes.size(); i++)
  {
    int B = inctive_nodes[i];
    if(B==1)
      active_nodes[i] = true;
    else
      active_nodes[i] = false;
  }

  //Dat is the real stuff:
  // Initialising the graph
  this->graph = NodeGraphV2(this->io_double_array["surface_elevation"], active_nodes,this->io_double["dx"], this->io_double["dy"],
this->io_int["n_rows"], this->io_int["n_cols"], this->lake_solver);

  // std::cout << "done, sorting few stuff around ..." << std::endl;
  
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

  // Stuff
  // I NEED TO WORK ON THAT PART YO
  // This add previous inherited water from previous lakes
  // Note that it "empties" the lake and reinitialise the depth. If there is still a reason to for the lake, it will form it
  // std::cout << "wat" << std::endl;
  this->process_inherited_water();
  // std::cout << "er" << std::endl;

  // Also initialising the Lake graph
  //# no lakes so far
  lake_network = std::vector<Lake>();
  //# incrementor reset to 0
  lake_incrementor = 0;
  this->lakes.clear();
  //# no nodes in lakes
  node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);

  std::cout << "done with setting up graph stuff" <<  std::endl;
}

void ModelRunner::run()
{
  // Alright now I need to loop from top to bottom
  std::cout << "Starting the run" << std::endl;

  this->lake_in_order = this->graph.get_Cordonnier_order();

  this->lake_status = std::vector<int>(this->io_int["n_elements"],-1);
  
  for(auto tn:lake_in_order)
  {
    this->lake_status[tn] = 0;
  }


  // Keeping track of which node is processed, for debugging and lake management
  is_processed = std::vector<bool>(io_int["n_elements"],false);

  // Aliases for efficiency
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];
  xt::pytensor<double,1>&surface_elevation =  this->io_double_array["surface_elevation"];
  // well area of a cell to gain time
  double cellarea = this->io_double["dx"] * this->io_double["dy"];
  // Debug checker
  int underfilled_lake = 0;
  std::cout << "starting iteration" << std::endl;
  // Iterating though all the nodes
  for(int i=0; i<io_int["n_elements"]; i++)
  {
    // Getting the current node in the Us->DS stack order
    int node = this->graph.get_MF_stack_at_i(i);
    if(this->graph.get_MF_receivers_at_node(node).size() == 0 && this->io_int_array["active_nodes"][node]>0 && this->graph.is_depression(node) == false)
      throw std::runtime_error("No rec Error");
    // Processing that node
    this->process_node(node, is_processed, lake_incrementor, underfilled_lake, inctive_nodes, cellarea, surface_elevation, true);   
    // std::cout << this->chonk_network[node].get_water_flux() << std::endl; 
    // Switching to the next node in line
  }
  std::cout << "First pass done" << std::endl;

  if(this->lake_solver)
  {
    std::cout << "Iterative lake pass..." << std::endl;
    this->iterative_lake_solver();
    std::cout << "Iterative lake pass... done" << std::endl;
  }

  // Calling the finalising function: it applies the changes in topography and I think will apply the lake sedimentation
  this->finalise();
  // Done
  // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  std::cout << "done" << std::endl;
}


void ModelRunner::iterative_lake_solver()
{
  // Right 
  // Initialising an empty queue of entry points, ie, points which will initiate a lake with a given sed and wat content
  std::queue<EntryPoint> iteralake;

  // Initialising the lake_status array: -1 = not a lake; 0 = to be processed and 1= processed at least once
  // setting all potential entry_points to 0
  for(auto starting_node : this->lake_in_order)
    this->lake_status[starting_node] = 0;

  // reinitialising the lakes
  this->lakes = std::vector<LakeLite>();
  this->has_been_outlet = std::vector<char>(this->io_int["n_elements"],'n');


  // Initialising the queue with the first lakes. Also I am preprocessing the flat surfaces
  for(auto starting_node : this->lake_in_order)
  {
    // std::cout << this->lake_status[starting_node] << " Starting node = " << starting_node << std::endl;
    // If this depression if already incorporated into another flat surface
    if(this->lake_status[starting_node] > 0)
      continue;

    // getting the full water and sed volumes to add to the lake
    double water_volume,sediment_volume; std::vector<double> label_prop;

    std::vector<int> these_nodes;
    // This function checks if there are flats around it and process the whole lake as a single flat
    this->original_gathering_of_water_and_sed_from_pixel_or_flat_area(starting_node, water_volume, sediment_volume, label_prop, these_nodes);
    // emplce the entry point and its characteristics into the lake
    iteralake.emplace(EntryPoint( water_volume,  sediment_volume,  starting_node, label_prop));
    // Also create an empty lake here
    this->lakes.push_back(LakeLite(this->lake_incrementor));
    this->lakes[lake_incrementor].nodes = these_nodes;
    for(auto tnode : these_nodes)
      this->node_in_lake[tnode] = this->lake_incrementor;

    this->lake_incrementor++;
  }


  std::cout << "DEBUG::Starting the iterative process..." << std::endl;
  // I am iterating while I still have some lakes to fill
  while(iteralake.empty() == false)
  {
    // this is a FIFO queue, First in, first out
    EntryPoint entry_point = iteralake.front();
    // removing the thingy
    iteralake.pop();
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


    // Function that fills and updates the topography, also checks for an outlet
    int current_lake = this->fill_mah_lake(entry_point, iteralake);

    // If there is an outlet detected in the current lake solver
    if(this->lakes[current_lake].outlet >= 0)
    {
      std::cout << "Lake " << current_lake << " -> " << this->lakes[current_lake].outlet;
      this->reprocess_nodes_from_lake_outlet(current_lake, this->lakes[current_lake].outlet, is_processed, iteralake, entry_point);
    }
  }
}

void ModelRunner::reprocess_nodes_from_lake_outlet(int current_lake, int outlet, std::vector<bool>& is_processed, std::queue<EntryPoint>& iteralake, EntryPoint& entry_point)
{
  xt::pytensor<double,1>& topography = this->io_double_array["topography"];
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];

  // gave_to_lake_water = std::vector<double>(this->io_int["n_elements"],0.);
  // gave_to_lake_sed = std::vector<double>(this->io_int["n_elements"],0.);


  // initialising a priority queue sorting the nodes to reprocess by their id in the Mstack. the smallest Ids should come first
  // Because I am only processing nodes in between 2 discrete lakes, it should not be a problem
  std::priority_queue< node_to_reproc, std::vector<node_to_reproc>, std::greater<node_to_reproc> > ORDEEEEEER;
  
  // Temporary queue gathering nodes to transmit
  std::queue<int> transec;
  
  // keeping track of which nodes are already in teh queue (or more generally to avoid)
  std::vector<char> is_in_queue(this->io_int["n_elements"],'n');

  //keeping track of the delta sed/all contributing to potential lakes to add
  std::vector<double> pre_sed(this->lakes.size(),0), pre_water(this->lakes.size(),0);
  std::vector<int> pre_entry_node(this->lakes.size(),-9999);
  std::vector<std::vector<double> > label_prop_of_pre(this->lakes.size(), std::vector<double>(this->n_labels,0.) );
  std::vector<double> delta_sed(this->lakes.size(),0), delta_water(this->lakes.size(),0);
  std::vector<std::vector<double> > label_prop_of_delta(this->lakes.size(), std::vector<double>(this->n_labels,0.) );

  // avoinding nodes in the lakes
  for(auto tnode:this->lakes[current_lake].nodes)
  {
    is_in_queue[tnode] = 'y';
  }


    // I will also reprocess a bunch of donors, but I cannot be sure they will be to reprocess before the end
  std::vector<int> node_to_deltaise_to_lake;

  // Initiating the transec with the outlet, but not putting it the 
  transec.emplace(outlet);
  is_in_queue[outlet] = 'y';
  if(this->node_in_lake[outlet] >= 0)
    node_to_deltaise_to_lake.push_back(outlet);


  while(transec.empty() == false)
  {
    int next_node = transec.front();
    transec.pop();

    // if this is not the outlet and if it is an active node, I add it in the PQ to reproc
    if(next_node != outlet )
    {
      ORDEEEEEER.emplace(node_to_reproc(next_node,this->graph.get_index_MF_stack_at_i(next_node)));
      is_in_queue[next_node] = 'y';
    }

    // otherwise going through neightbors
    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node, active_nodes, neightbors, dummy);

    // checking the state of the neightbor
    for (auto tnode : neightbors)
    {
      // If is already processed, in queue, or in the original lake, ingore it
      if(is_in_queue[tnode] == 'y')
        continue;

      // if(this->node_in_lake[tnode]>=0)
      //   if(motherlake(this->node_in_lake[tnode]) == current_lake)
      //     continue;
      
      // if it is below my current node
      if(topography[tnode] < topography[next_node] )
      {
        // If already in an existing lake, saving the node as lake potential
        if(this->node_in_lake[tnode] >= 0)
        {
          node_to_deltaise_to_lake.push_back(tnode);
          is_in_queue[tnode] = 'y';
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
        if(is_in_queue[tnode] == 'n')
        {
          is_in_queue[tnode] = 'd';
          continue;
        }
      }

    }

  }


  // Adding the donors to the PQ
  for(int i =0; i<this->io_int["n_elements"]; i++)
  {
    if (is_in_queue[i] == 'd')
    {

      // if(has_been_outlet[i] == 'y')
      //   throw std::runtime_error("UH?");

      ORDEEEEEER.emplace(node_to_reproc(i,this->graph.get_index_MF_stack_at_i(i)));
    }
  }

  // Formatting the iteratorer, whatever the name this thing is. not iterator. is something else. I guess. I think. why would you care anyway as I'd be surprised anyone ever reads this code.
  std::vector<int> local_mstack; local_mstack.reserve(ORDEEEEEER.size());
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

    // if not a donor, reset the MF node
    if(is_in_queue[tnode] != 'd')
    {
      this->cancel_fluxes_before_moving_prep(this->chonk_network[tnode], tnode);
      this->chonk_network[tnode].reset();
      this->chonk_network[tnode].set_other_attribute_array("label_tracker", std::vector<double>(this->n_labels,0));
    }
  }

  // Final size OK

  std::vector<int> neightbors; std::vector<double> dist ; graph.get_D8_neighbors(this->lakes[current_lake].outlet, active_nodes, neightbors, dist);
  std::vector<int> donors_to_outlet;
  double maxslope = 0;
  int ID = -9999;
  int __ = -1;
  for(auto neigh:neightbors)
  {
    __++;
    
    if(this->node_in_lake[neigh] > -1)
    if(motherlake(this->node_in_lake[neigh]) == current_lake)
      continue;

    double tslope = (topography[this->lakes[current_lake].outlet] - topography[neigh])/dist[__];
    if(tslope>maxslope)
    {
      ID = neigh;
      maxslope = tslope;
    }

    if(topography[this->lakes[current_lake].outlet] < topography[neigh])
      donors_to_outlet.push_back(neigh);
  }
  if(this->node_in_lake[ID] >= 0)
    node_to_deltaise_to_lake.push_back(ID);


  // Now deprocessing the receivers in potential lake while saving their contribution to lakes in order to calculate the delta
  for(auto tnode : local_mstack)
  {
    if (is_in_queue[tnode] == 'd')
      continue;

    std::vector<int> these_lakid; std::vector<double> twat; std::vector<double> tsed; std::vector<std::vector<double> > tlab; std::vector<int> bulug;
    this->check_what_gives_to_lake(tnode, these_lakid, twat, tsed, tlab, bulug, current_lake);

    for(int i = 0; i < int(these_lakid.size()); i++)
    {
      int tlakid = these_lakid[i];
      label_prop_of_pre[tlakid] = mix_two_proportions(pre_sed[tlakid],label_prop_of_pre[tlakid],tsed[i],tlab[i]);
      pre_sed[tlakid] += tsed[i];
      pre_water[tlakid] += twat[i];
      pre_entry_node[tlakid] = bulug[i];
    }

  }

  // Reprocessing the outlet here!
  chonk& tchonk = this->chonk_network[this->lakes[current_lake].outlet];
  // Saving the part of the fluxes going in the not-lake direction first
  
  if(this->has_been_outlet[this->lakes[current_lake].outlet] == 'n')
  {
    std::vector<int> ignore_some; 
    for(auto ttnode: tchonk.get_chonk_receivers_copy())
    {
      if(this->node_in_lake[ttnode] >= 0)
      if(motherlake(this->node_in_lake[ttnode]) == current_lake)
        ignore_some.push_back(ttnode);
    }

    tchonk.split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);
    has_been_outlet[this->lakes[current_lake].outlet] = 'y';

  }
  std::cout << " ---------> [" <<tchonk.get_water_flux() << "] ";

  if(true)
  {
    std::vector<int> these_lakid; std::vector<double> twat; std::vector<double> tsed; std::vector<std::vector<double> > tlab; std::vector<int> bulug;
    this->check_what_gives_to_lake(this->lakes[current_lake].outlet, these_lakid, twat, tsed, tlab, bulug,current_lake);

    for(int i = 0; i < int(these_lakid.size()); i++)
    {
      int tlakid = these_lakid[i];
      label_prop_of_pre[tlakid] = mix_two_proportions(pre_sed[tlakid],label_prop_of_pre[tlakid],tsed[i],tlab[i]);
      pre_sed[tlakid] += tsed[i];
      pre_water[tlakid] += twat[i];
      pre_entry_node[tlakid] = bulug[i];
    }
  }

  tchonk.reset();


  double water_rate = entry_point.volume_water / this->timestep + this->lakes[current_lake].sum_outrate;
  this->lakes[current_lake].sum_outrate += entry_point.volume_water / this->timestep;
  double sed_rate = entry_point.volume_sed;
  std::vector<double> labprop = entry_point.label_prop;
  tchonk.set_water_flux(water_rate);
  tchonk.set_sediment_flux(sed_rate, labprop);
  std::cout << ID << " [" << water_rate << "]" << std::endl;

  
  double cellarea = this->io_double["dx"] * this->io_double["dy"];

  // // Last step: reprocess donors to outlet, jsut for the outlet
  // for(auto tnode:donors_to_outlet)
  // {
  //   std::vector<int> ignore_some; 
  //   for(auto ttnode: this->chonk_network[tnode].get_chonk_receivers_copy())
  //   {
  //     if(ttnode != outlet)
  //       ignore_some.push_back(ttnode);
  //   }
  //   this->chonk_network[tnode].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);
  // }



  tchonk.external_moving_prep({ID},{1.},{1.},{maxslope});
  this->process_node_nolake_for_sure(this->lakes[current_lake].outlet, is_processed, active_nodes, 
      cellarea,topography, false, false);

  // I am now ready to reprocess all the node from upstream to downstream, and then rechek the delat for me lakes
  for(auto tnode:local_mstack)
  {

    if(is_in_queue[tnode] == 'd')
    {
      // this->chonk_network[tnode].reinitialise_moving_prep();
      // this->manage_move_prep(this->chonk_network[tnode]);
      std::vector<int> ignore_some; 
      for(auto ttnode: this->chonk_network[tnode].get_chonk_receivers_copy())
      {
        if(is_in_queue[ttnode] != 'y' || ttnode == this->lakes[current_lake].outlet)
        {
          ignore_some.push_back(ttnode);
          continue;
        }
        else if (this->node_in_lake[ttnode] >= 0)
          if(motherlake(this->node_in_lake[ttnode]) == current_lake)
          {
            ignore_some.push_back(ttnode);
            continue;
          }


      }

      this->chonk_network[tnode].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);

    }
    else
    {
      this->process_node_nolake_for_sure(tnode, is_processed, active_nodes, 
        cellarea,topography, true, true);
    }

  }

  // I have reprocessed my stuff, lets calculate the delta by lakes to add entry points to the queue

  // Now deprocessing the receivers in potential lake while saving their contribution to lakes in order to calculate the delta
  for(auto tnode : local_mstack)
  {

    if (is_in_queue[tnode] == 'd')
      continue;


    std::vector<int> these_lakid; std::vector<double> twat; std::vector<double> tsed; std::vector<std::vector<double> > tlab; std::vector<int> bulug;
    this->check_what_gives_to_lake(tnode, these_lakid, twat, tsed, tlab, bulug, current_lake);

    for(int i = 0; i < int(these_lakid.size()); i++)
    {
      int tlakid = these_lakid[i];
      label_prop_of_delta[tlakid] = mix_two_proportions(delta_sed[tlakid],label_prop_of_delta[tlakid],tsed[i],tlab[i]);
      delta_sed[tlakid] += tsed[i];
      delta_water[tlakid] += twat[i];
    }

  }

  if(true)
  {
    std::vector<int> these_lakid; std::vector<double> twat; std::vector<double> tsed; std::vector<std::vector<double> > tlab; std::vector<int> bulug;
    this->check_what_gives_to_lake(this->lakes[current_lake].outlet, these_lakid, twat, tsed, tlab, bulug, current_lake);

    for(int i = 0; i < int(these_lakid.size()); i++)
    {
      int tlakid = these_lakid[i];
      label_prop_of_pre[tlakid] = mix_two_proportions(delta_sed[tlakid],label_prop_of_delta[tlakid],tsed[i],tlab[i]);
      delta_sed[tlakid] += tsed[i];
      delta_water[tlakid] += twat[i];
    }
  }

  for(size_t i=0; i < this->lakes.size(); i++ )
  {
    double dwat = delta_water[i] - pre_water[i];
    double dsed = delta_sed[i] - pre_sed[i];
    
    if(dwat <= 0 && dsed <= 0)
      continue;

    std::cout << "DWAT = " << dwat  << " TO " << pre_entry_node[i] << std::endl;
    iteralake.emplace(EntryPoint(dwat * this->timestep, dsed, pre_entry_node[i], label_prop_of_delta[i]));
  }

  debugint = xt::zeros<int>({this->io_int["n_elements"]});
  for(int i = 0; i < this->io_int["n_elements"]; i++)
  {
    int val = -1;

    if(is_in_queue[i] == 'd')
      val = 0;
    if(is_in_queue[i] == 'y')
      val = 1;

    debugint[i] = val;
  }
  // Doen
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
    
    if(is_not_here)
    {
    
      index = int(these_lakid.size());
      these_lakid.push_back(this_lakid);
      twat.push_back(this->chonk_network[entry_node].get_water_flux() * WWC[idx_rec]);
      tsed.push_back(this->chonk_network[entry_node].get_sediment_flux() * WWS[idx_rec]);
      tlab.push_back(this->chonk_network[entry_node].get_other_attribute_array("label_tracker"));
      these_ET.push_back(recs[idx_rec]);

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

int ModelRunner::fill_mah_lake(EntryPoint& entry_point, std::queue<EntryPoint>& iteralake)
{
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;
  xt::pytensor<double,1>& topography = this->io_double_array["topography"];
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];

  depressionfiller.emplace(nodium(entry_point.node, topography[entry_point.node]));

  //DEBUG STATEMENT
  std::cout << "Filling lake at node " << entry_point.node << " with rate of  " << entry_point.volume_water/this->timestep << std::endl;
  double save_entry_water = entry_point.volume_water;

  int current_lake = this->lake_incrementor;
  this->lakes.push_back(LakeLite(this->lake_incrementor));
  this->lake_incrementor++;

  this->lakes[current_lake].water_elevation = topography[entry_point.node];

  int outlet = -9999;
  std::vector<char> is_in_queue(this->io_int["n_elements"],'n');
  is_in_queue[entry_point.node] = 'y';
  std::vector<char> is_in_lake(this->io_int["n_elements"],'n');

  if(active_nodes[entry_point.node] == 0)
  {
    outlet = entry_point.node;
  }

  while(entry_point.volume_water > 0 && outlet < 0)
  {
    // first get the node
    nodium next_node = depressionfiller.top();
    // then pop the next node
    depressionfiller.pop();

    // go through neighbours manually and either feed the queue or detect an outlet
    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node.node, active_nodes, neightbors, dummy);
    for(auto tnode:neightbors)
    {
      if(is_in_queue[tnode] == 'y')
        continue;

      if(topography[tnode] >= topography[next_node.node])
      {
        depressionfiller.emplace(nodium(tnode, topography[tnode]));
        is_in_queue[tnode] = 'y';
      }
      else
      {
        outlet = next_node.node;
        break;
      }
    }

    double area_component_of_volume = int(this->lakes[current_lake].nodes.size()) * this->io_double["dx"] * this->io_double["dy"];
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
        std::vector<int> fnodes = this->graph.get_all_flat_from_node(next_node.node, topography, active_nodes);
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

  std::cout << "DEUBUG::NODES IN LAKE  (watelev = " << this->lakes[current_lake].water_elevation << ")::";
  for (auto tnode:this->lakes[current_lake].nodes)
    std::cout << tnode << "|";
  std::cout << std::endl;


  std::cout << "STORED " << this->lakes[current_lake].volume_water/this->timestep << " IN THE LAKE (rate)" << std::endl;

  // if there is an outlet 
  if(outlet >= 0)
    this->lakes[current_lake].outlet = outlet;

  // Now merging with lakes below adn updating the topography
  for(auto tnode: this->lakes[current_lake].nodes)
  {
    // updating topography
    topography[tnode] = this->lakes[current_lake].water_elevation;
    // checking if the node belongs to a lake, in which case I ingest it
    if(this->node_in_lake[tnode] >= 0)
    {
      int tested = this->motherlake(this->node_in_lake[tnode]);
      int against = this->lakes[current_lake].id;

      if( tested != against)
      {
        std::cout << tested << " SHOULD BE DIFFERENT THAN " << against << " IF THIS MESSAGE IS DISPLAYED" << std::endl;
        this->drink_lake(this->lakes[current_lake].id, this->motherlake(this->node_in_lake[tnode]));
      }
    }
    this->node_in_lake[tnode] = current_lake;
  }

  // Finally adding the sediments
  this->lakes[current_lake].label_prop = mix_two_proportions(entry_point.volume_sed, entry_point.label_prop,
    this->lakes[current_lake].volume_sed, this->lakes[current_lake].label_prop);

  if(this->lakes[current_lake].volume_water <= entry_point.volume_sed )
  {
   
   this->lakes[current_lake].volume_sed = this->lakes[current_lake].volume_water;
   entry_point.volume_sed -= this->lakes[current_lake].volume_water;

  }
  else
  {
    this->lakes[current_lake].volume_sed = entry_point.volume_sed;
    entry_point.volume_sed = 0;

  }
  


  return current_lake;

}

/// function eating a lake from another
void ModelRunner::drink_lake(int id_eater, int id_edible)
{ 
  // Updating the id of the new lake
  this->lakes[id_edible].is_now = id_eater;
  // merging water volumes
  this->lakes[id_eater].volume_water += this->lakes[id_edible].volume_water;

  if (this->lakes[id_eater].outlet == this->lakes[id_edible].outlet)
  {
    std::cout << "LAKE TRANSMIT ITS SUMOUTRATE :: " << this->lakes[id_edible].sum_outrate << std::endl;
    this->lakes[id_eater].sum_outrate += this->lakes[id_edible].sum_outrate;
  }
  // Merging sediment volumes
  this->lakes[id_eater].label_prop = mix_two_proportions(this->lakes[id_eater].volume_sed,this->lakes[id_eater].label_prop,
    this->lakes[id_edible].volume_sed,this->lakes[id_edible].label_prop);
  this->lakes[id_eater].volume_sed += this->lakes[id_edible].volume_sed;
}

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
  
  // return;
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  xt::pytensor<double,1>& topography = this->io_double_array["topography"];

  std::queue<int> FIFO;
  std::vector<char> is_in_queue(this->io_int["n_elements"], 'n');
  is_in_queue[starting_node] = 'y';
  these_nodes.push_back(starting_node);
  FIFO.push(starting_node);
  
  while(FIFO.empty() == false)
  {
    int next_node = FIFO.front();
    FIFO.pop();

    std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node, active_nodes, neightbors, dummy);
    double telev = topography[next_node];

    for(auto tnode:neightbors)
    {
      if(active_nodes[tnode] == false)
        continue;
      if(is_in_queue[tnode] == 'y')
        continue;

      if(topography[tnode] == telev)
      {
        FIFO.push(tnode);
        is_in_queue[tnode] = 'y';

        if(this->lake_status[tnode] == 0)
        {
          auto this_label_prop = this->chonk_network[tnode].get_other_attribute_array("label_tracker");
          label_prop = mix_two_proportions(sediment_volume, label_prop, this->chonk_network[tnode].get_sediment_flux(), this_label_prop);
          water_volume +=  this->chonk_network[tnode].get_water_flux()  * this->timestep;
          sediment_volume += this->chonk_network[tnode].get_sediment_flux();
        }
        this->lake_status[tnode] = 1;
        these_nodes.push_back(tnode);
      
      }
    }
  }  

}

void ModelRunner::process_node(int& node, std::vector<bool>& is_processed, int& lake_incrementor, int& underfilled_lake,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep)
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
      if(inctive_nodes[node] == 1 && this->graph.is_depression(node))
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
        this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
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
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
}

void ModelRunner::process_node_nolake_for_sure(int& node, std::vector<bool>& is_processed,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move)
{
    is_processed[node] = true;
    if(need_flux_before_move)
      this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);
    
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    
    // if(this->chonk_network[node].get_chonk_receivers().size() == 0 && inctive_nodes[node] > 0)
    //   throw std::runtime_error("NoRecError::internal flux broken");
    
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
}

void ModelRunner::process_node_nolake_for_sure(int& node, std::vector<bool>& is_processed,
  xt::pytensor<int,1>& inctive_nodes, double& cellarea, xt::pytensor<double,1>& surface_elevation, bool need_move_prep, bool need_flux_before_move, std::vector<int>& ignore_some)
{
    is_processed[node] = true;
    if(need_flux_before_move)
      this->manage_fluxes_before_moving_prep(this->chonk_network[node], this->label_array[node]);
    // first step is to apply the right move method, to prepare the chonk to move
    if(need_move_prep)
      this->manage_move_prep(this->chonk_network[node]);
    
    this->manage_fluxes_after_moving_prep(this->chonk_network[node],this->label_array[node]);
    
    // if(this->chonk_network[node].get_chonk_receivers().size() == 0 && inctive_nodes[node] > 0)
    //   throw std::runtime_error("NoRecError::internal flux broken");
    
    this->chonk_network[node].split_and_merge_in_receiving_chonks_ignore_some(this->chonk_network, this->graph, this->timestep, ignore_some);
}

void ModelRunner::increment_new_lake(int& lakeid)
{
  this->lake_network.push_back(Lake(lake_incrementor));
  lakeid = lake_incrementor;
  lake_incrementor++;
}

void ModelRunner::finalise()
{
  // Finilising the timestep by applying the changes to the thingy
  // First gathering all the aliases 
  xt::pytensor<double,1>& surface_elevation_tp1 = this->io_double_array["surface_elevation_tp1"];
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<double,1>& topography = this->io_double_array["topography"];
  xt::pytensor<double,1>& sed_height_tp1 = this->io_double_array["sed_height_tp1"];
  // xt::pytensor<double,1> tlake_depth = xt::zeros<double>({size_t(this->io_int["n_elements"])});
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];

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

    if(active_nodes[i] == 0)
      continue;

    // Getting the current chonk
    chonk& tchonk = this->chonk_network[i];
    // getting the current composition of the sediment flux
    auto this_lab = tchonk.get_other_attribute_array("label_tracker");

    // NANINF DEBUG CHECKER
    for(auto LAB:this_lab)
      if(std::isfinite(LAB) == false)
        std::cout << LAB << " << naninf for sedflux" << std::endl;

    // // getting the lake ID and depth
    // if(node_in_lake[i]>=0)
    // {
    //   int lakeid = node_in_lake[i];
    //   if(this->lake_network[lakeid].get_parent_lake() > 0)
    //     lakeid = this->lake_network[lakeid].get_parent_lake();

    //   double tdepth = lake_network[lakeid].get_lake_depth_at_node(i,node_in_lake);
    //   tlake_depth[i] = tdepth;
    // }
    // else
    // {
    //   // if no lake, depth is 0 NSS
    //   tlake_depth[i] = 0;
    // }
    

    // First applying the bedrock-only erosion flux: decrease the overal surface elevation without affecting the sediment layer
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux_only_bedrock() * timestep;

    // Applying elevation changes from the sediments
    // Reminder: sediment creation flux is the absolute rate of removal/creation of sediments
    double sedcrea = tchonk.get_sediment_creation_flux() * timestep;

    // NANINF DEBUG CHECKER
    if(std::isfinite(sedcrea) == false)
      throw std::runtime_error("NAN sedcrea finalisation not possible yo");


    // std::cout << "0.2" << std::endl;
    // TEMP DEBUGGER TOO
    // AT TERM THIS SHOULD NOT HAPPEN???
    // if I end up with a negative sediment layer
    if(sedcrea + sed_height_tp1[i] < 0)
    {
      std::cout << "happens??" << sedcrea << "||" << sed_height_tp1[i] << "||" << this->node_in_lake[i] << std::endl;
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
      if(std::ceil(sed_height_tp1[i]/this->io_double["depths_res_sed_proportions"]) != int(sed_prop_by_label[i].size()))
      {
        std::cout << "called with " << i << "|" << sedcrea << "|" << sed_height_tp1[i] << std::endl;
        std::cout << std::ceil(sed_height_tp1[i]/this->io_double["depths_res_sed_proportions"]) << "||" << int(sed_prop_by_label[i].size()) << std::endl;
        throw std::runtime_error("COCKROACH");
      }
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

    if(sed_prop_by_label[i].size() == 0 && sed_height_tp1[i] > 0)
      throw std::runtime_error("Sediment but no tracking error");



  }

  auto tlake_depth = this->io_double_array["topography"] - this->io_double_array["surface_elevation"];
  // Calculating the water balance thingies
  double save_Ql_out = this->Ql_out;
  this->Ql_out = 0;
  for(int i=0; i<this->io_int["n_elements"]; i++)
  {
    this->Ql_out += (tlake_depth[i] - this->io_double_array["lake_depth"][i]) * this->io_double["dx"] * this->io_double["dy"] / this->timestep;
  }

  
  // Saving the new lake depth  
  this->io_double_array["lake_depth"] = tlake_depth;


  // calculating other water mass balance.
  // xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  for(int i =0; i<this->io_int["n_elements"]; i++)
  {
    if(active_nodes[i] == 0)
    {
      this->Qw_out += this->chonk_network[i].get_water_flux();
    }
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







void ModelRunner::find_nodes_to_reprocess(int start, std::vector<bool>& is_processed, std::vector<int>& nodes_to_reprocess, std::vector<int>& nodes_to_relake,
  std::vector<int>& nodes_to_recompute_neighbors_at_the_end, int lake_to_avoid)
{
  // Alright Now is the time to finally comment this function as it seems to be the reason of mass balance and sediment nan bug :(
  // I have been dreading this moment as this function has been written in one go alongside all the lake reprocessing routines

  // this was a debugging statement marking the start of the function to locate where it first segfaulted
  // std::cout << "start" << std::endl;

  // Some magic to keep a nice balance between cpu and memory optimisation: I initialise a vector assuming I will probably not need to reprocess > 1/4 of the nodes
  // if I eceed this size, I use push_back whihc is slightly slower
  int tsize = int(round(this->io_int["n_elements"]/4));
  // I am initialising various ID to insert and read nodes in this vector
  int insert_id_trav = 0,insert_id_lake = 0,reading_id = -1, insert_id_proc = 1;

  // traversal is my vector storing the nodes to reprocess in a BFS search-like algorithm. I think.
  std::vector<int> traversal;traversal.reserve(tsize);
  // the Q storing the nodes to process
  std::vector<int> tQ;tQ.reserve(tsize);
  // the lake IDs to (re)process
  std::vector<int> travlake;travlake.reserve(tsize);
  // Sets to keep track of nodes already processed in this function. Again, should be more efficient than a full vector as the number of nodes whshould be relatively low
  std::set<int> is_in_Q,is_in_lake;

  // Starting the processing with the starting node ofc
  tQ.emplace_back(start);
  is_in_Q.insert(start);

  // const bool is_in = container.find(element) != container.end();
  // Iterating til' I decide it
  while(true)
  {
    // Reading the next node in the Q
    // if first time, -1 +1 =0 so it works aye
    reading_id++;
    // if I reach the end of me Q, tis the end
    // It works because I reserve some space in it, so I increase its CAPACITY but not its SIZE which grows with emplace_back
    if(reading_id >= tQ.size())
      break;

    // getting the node
    int this_node = tQ[reading_id];
    bool recomputed_neight = false;
    // std::cout << "|Q_NODE:" << this_node;
    // if node is not start, I put it into the traversal, cause it has already been checked
    if(this_node != start)
    {  
      if(insert_id_trav < tsize)
      {
        traversal.emplace_back(this_node);
        insert_id_trav++;
      }
      else
        traversal.push_back(this_node);
    }

    std::vector<int> golog = this->graph.get_MF_receivers_at_node(this_node);
    // iterating through all his neighbors
    for(auto node:golog)
    {
     
      // if the neightbor has not been processed originally no reproc
      if(is_processed[node] == false)
        continue;

      // if the node is already in Q skip
      if(is_in_Q.find(node) != is_in_Q.end())
        continue;
     
      // getting the lake id
      int this_lake_id = node_in_lake[node];

      // If this node is in a lake, but not processed yet by this function and not the starting one
      if(this_lake_id >= 0 && is_in_lake.find(node) == is_in_lake.end()) // && this_node != start)
      {

        // Checking and avoiding dedundancy-> if is in the lake to avoid -> rework the receivers of the outlet and process it
        if(this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid)
        {
          std::vector<int> new_rec;
          std::vector<int> datiznogoud = this->graph.get_MF_receivers_at_node(this_node);
          for(auto recnode:datiznogoud)
          {
            if(recnode != node)
              new_rec.push_back(recnode);
          }
          // std::cout << "IT HAPPENS" << std::endl;
          this->graph.update_receivers_at_node(this_node, new_rec);
          recomputed_neight = true;
          continue;
        }

        // if it passes this test, then the node is in a lake to reprocess

        is_in_lake.insert(node);
        if(insert_id_lake < tsize )
        {

          travlake.emplace_back(node);
          insert_id_lake++;
        }
        else
          travlake.push_back(node);

        // I do not want to reprocess this node, so I am putting "in the Q" so that it gets ignored now. I know I have to reprocess it
        is_in_Q.insert(node);
        
        continue;
      }

      // if is in lake to avoid at start, I stop here
      if(node == start && (this_lake_id == lake_to_avoid || this->lake_network[this_lake_id].get_parent_lake() == lake_to_avoid) )
        continue;

      // otherwise, I'll need to check this node and put it in teh traversal yolo
      is_in_Q.insert(node);
      if(insert_id_proc < tsize )
      {
        tQ.emplace_back(node);
        insert_id_proc++;
      }
      else
        tQ.push_back(node);

    }
    if(recomputed_neight)
      nodes_to_recompute_neighbors_at_the_end.push_back(this_node);

  }


  // for (auto bugh:traversal)
  //   if(node_in_lake[bugh] >= 0)
  //     throw std::runtime_error("ERROAR! this nor should not be in this traversal");
  // if(travlake.size()>0)
  //   std::cout << "{" << travlake.size() << "}";

  nodes_to_reprocess = std::move(traversal);
  nodes_to_relake = std::move(travlake);




  // // std::cout << "end" << std::endl;
  // std::map<int,int> node_counter;
  // for(auto n:nodes_to_reprocess)
  // {
  //   node_counter[n] = 0;
  // }
  // for(auto n:nodes_to_reprocess)
  // {
  //   node_counter[n] += 1;
  //   if(node_counter[n] > 1)
  //     throw std::runtime_error("Doubled reprocessing error");
  // }

}



void ModelRunner::manage_fluxes_before_moving_prep(chonk& this_chonk, int label_id)
{

  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 5:
        this_chonk.inplace_only_drainage_area(this->io_double["dx"], this->io_double["dy"]);
        this->Qw_in += this->io_double["dx"]* this->io_double["dy"];
        break;
      case 6:
        this_chonk.inplace_precipitation_discharge(this->io_double["dx"], this->io_double["dy"],this->io_double_array["precipitation"]);
        this->Qw_in += this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
      case 7:
        this_chonk.inplace_infiltration(this->io_double["dx"], this->io_double["dy"], this->io_double_array["infiltration"]);
        this->Qw_out += this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
    }

  }
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
        this_chonk.cancel_inplace_only_drainage_area(this->io_double["dx"], this->io_double["dy"]);
        this->Qw_in -= this->io_double["dx"]* this->io_double["dy"];
        break;
      case 6:
        this_chonk.cancel_inplace_precipitation_discharge(this->io_double["dx"], this->io_double["dy"],this->io_double_array["precipitation"]);
        this->Qw_in -= this->io_double_array["precipitation"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
      case 7:
        this_chonk.cancel_inplace_infiltration(this->io_double["dx"], this->io_double["dy"], this->io_double_array["infiltration"]);
        this->Qw_out -= this->io_double_array["infiltration"][this_chonk.get_current_location()] * this->io_double["dx"]* this->io_double["dy"];
        break;
    }

  }
}



void ModelRunner::manage_move_prep(chonk& this_chonk)
{
  int this_case = intcorrespondance[this->move_method];

  std::vector<int> rec = this_chonk.get_chonk_receivers_copy();
  switch(this_case)
  {
    case 2:
      this_chonk.move_to_steepest_descent(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["topography"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 3:
      this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["topography"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 4:
      this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["topography"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
      
    default:
      std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  }
}

void ModelRunner::manage_fluxes_after_moving_prep(chonk& this_chonk, int label_id)
{
  bool has_moved = false;
  for(auto method:this->ordered_flux_methods)
  {
    int index = this_chonk.get_current_location();
    
    if(method == "move")
    {
       has_moved = true;
       continue;
    }

    if(has_moved == false)
      continue;

    int this_case = intcorrespondance[method];
    switch(this_case)
    {
      case 1:
        this_chonk.active_simple_SPL(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["SPIL_K"][label_id], this->timestep, this->io_double["dx"], this->io_double["dy"], label_id);
        break;
      case 8:
        std::vector<double> these_sed_props(this->n_labels,0.);
        if(is_there_sed_here[index] && this->sed_prop_by_label[index].size()>0) // I SHOULD NOT NEED THE SECOND THING, WHY DO I FUTURE BORIS????
          these_sed_props = this->sed_prop_by_label[index][this->sed_prop_by_label[index].size() - 1];
      // if(this->node_in_lake[index] >=0)
      // {
      //   std::cout << "NODEINLAKEYO? " << index << "|" << this->node_in_lake[index] << "|" << this-> lake_network[this->node_in_lake[index]].get_lake_outlet() << std::endl;
      //   throw std::runtime_error("node in lake yo should not charlicise");
      // }
        this_chonk.charlie_I(this->labelz_list_double["SPIL_n"][label_id], this->labelz_list_double["SPIL_m"][label_id], this->labelz_list_double["CHARLIE_I_Kr"][label_id], 
  this->labelz_list_double["CHARLIE_I_Ks"][label_id],
  this->labelz_list_double["CHARLIE_I_dimless_roughness"][label_id], this->io_double_array["sed_height"][index], 
  this->labelz_list_double["CHARLIE_I_V"][label_id], 
  this->labelz_list_double["CHARLIE_I_dstar"][label_id], this->labelz_list_double["CHARLIE_I_threshold_incision"][label_id], 
  this->labelz_list_double["CHARLIE_I_threshold_entrainment"][label_id],
  label_id, these_sed_props, this->timestep,  this->io_double["dx"], this->io_double["dy"]);
        
        break;
    }
  }
  return;
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
    xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];


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
      // sumwat += this->io_double_array["lake_depth"][node] * this->io_double["dx"] * this->io_double["dy"] / this->timestep ;
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
    xt::pytensor<double,1> temp = xt::zeros_like(io_double_array["surface_elevation"]);
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


//#################################################################################
//#################################################################################
//#################################################################################
//#################################################################################
// Functions managing lakes #######################################################
//#################################################################################
//#################################################################################
//#################################################################################


void Lake::ingest_other_lake(
   Lake& other_lake,
   std::vector<int>& node_in_lake, 
   std::vector<bool>& is_in_queue,
   std::vector<Lake>& lake_network,
   xt::pytensor<double,1>& topography
   )
{
  std::cout << "LAKE " << this->lake_id << " is ingesting lake " << other_lake.get_lake_id() << std::endl;
  // getting the attributes of the other lake
  std::vector<int>& these_nodes = other_lake.get_lake_nodes();
  std::vector<int>& these_node_in_queue = other_lake.get_lake_nodes_in_queue();
  std::unordered_map<int,double>& these_depths = other_lake.get_lake_depths();
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& this_PQ = other_lake.get_lake_priority_queue();

  // if(this->water_elevation != other_lake.get_water_elevation())
  // {
  //   std::cout << this->water_elevation << "||" << other_lake.get_water_elevation() << std::endl;
  //   throw std::runtime_error("IOIOIOIOIOIO");
  // }

  // merging them into this lake while node forgetting to label them as visited and everything like that
  for(auto node:these_nodes)
  {
    if(std::find(this->nodes.begin(), this->nodes.end(), node) == this->nodes.end())
      this->nodes.push_back(node);
    node_in_lake[node] = this->lake_id;
  }

  for(auto node:these_node_in_queue)
  {
    this->node_in_queue.push_back(node);
    is_in_queue[node] = true;
  }

  this->depths.insert(these_depths.begin(), these_depths.end());

  this->ngested_nodes += int(other_lake.get_n_nodes());
  // this->n_nodes += other_lake.get_n_nodes();

  // if(int(these_nodes.size()) !=other_lake.get_n_nodes() )
  //   throw std::runtime_error("TRRTRTTSTSK");

  // Transferring the PQ (not ideal but meh...)
  while(this_PQ.empty() == false)
  {
    nodium this_nodium = this_PQ.top();
    is_in_queue[this_nodium.node] = true;
    this->depressionfiller.emplace(nodium(this_nodium.node, topography[this_nodium.node] ));
    this_PQ.pop();
  }
  this->pour_sediment_into_lake(other_lake.get_volume_of_sediment(), other_lake.get_lake_lab_prop());

  this->volume += other_lake.get_lake_volume();

  // Deleting this lake and setting its parent lake
  int save_ID = other_lake.get_lake_id();


  Lake temp = Lake(save_ID);
  other_lake = temp;
  other_lake.set_parent_lake(this->lake_id);
  this->ingested_lakes.push_back(other_lake.get_lake_id());

  for(auto lid:other_lake.get_ingested_lakes())
    lake_network[lid].set_parent_lake(this->lake_id);


  return;
}

void Lake::pour_sediment_into_lake(double sediment_volume, std::vector<double> label_prop)
{
  if(this->volume_of_sediment<0)
    std::cout << "NEG BEFORE POURING SED" << std::endl;
  if(sediment_volume<0)
    std::cout << "NEG POURED" << std::endl;

  std::vector<double> coplaklab = this->lake_label_prop;
  std::vector<double> copaddlab = label_prop;
  
  if(this->lake_label_prop.size()>0)
  {
    // std::cout << "bo";
    for(auto lb:this->lake_label_prop)
      if(std::isfinite(lb) == false)
        std::cout << "NANINF in lake already" << std::endl;
    for(auto lb:label_prop)
      if(std::isfinite(lb) == false)
        std::cout << "NANINF coming in" << std::endl;
    if(this->volume_of_sediment > 0)
    {
      // std::cout << "go";
      this->lake_label_prop = mix_two_proportions(this->volume_of_sediment, this->lake_label_prop, sediment_volume, label_prop);
      // std::cout << "ris";
    }
    else
    {
      this->lake_label_prop = label_prop;
    }

      
    for(auto lb:this->lake_label_prop)
      if(std::isfinite(lb) == false)
      {
        std::cout << "NANINF in lake after::" << this->volume_of_sediment << "||" << sediment_volume << std::endl;
        for(auto yu:this->lake_label_prop)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:label_prop)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:copaddlab)
          std::cout << yu << "|";
        std::cout << std::endl;
        for(auto yu:coplaklab)
          std::cout << yu << "|";
        std::cout << std::endl;
        
        throw std::runtime_error("NANINF in lake sed pouring");
      }
    // std::cout << "ris";
  }
  else
    this->lake_label_prop = label_prop;

  this->volume_of_sediment += sediment_volume;
  if(this->volume_of_sediment<0)
  {
    std::cout << "NEG AFTER POURING SED " << sediment_volume << std::endl;
    // throw std::runtime_error("sedneg issue number 5");
  }

}

// Give the deposition fluxe from lakes to the 
void Lake::drape_deposition_flux_to_chonks(std::vector<chonk>& chonk_network, xt::pytensor<double,1>& surface_elevation, double timestep)
{

  if(this->volume == 0)
    return;

  double ratio_of_dep = this->volume_of_sediment/this->volume;

  // NEED TO DEAL WITH THAT BOBO
  if(ratio_of_dep>1)
    ratio_of_dep = 1;

  for(auto& no:this->nodes)
  {
    double pre = chonk_network[no].get_sediment_creation_flux();
    if(std::isfinite(pre) ==false)
     throw std::runtime_error("already naninf before draping");// std::cout <<  chonk_network[no].get_sediment_creation_flux() << "||";
    

    double slangh = ratio_of_dep * (this->water_elevation - surface_elevation[no]) / timestep;
    chonk_network[no].add_sediment_creation_flux(slangh);
    chonk_network[no].set_other_attribute_array("label_tracker", this->outlet_chonk.get_other_attribute_array("label_tracker"));

    pre = chonk_network[no].get_sediment_creation_flux();
    if(std::isfinite(pre) == false)
    {
      std::cout << ratio_of_dep << "||" << (this->water_elevation - surface_elevation[no]) << "||" << timestep << std::endl;
      throw std::runtime_error(" naninf after draping");
    }
  }
}

// Give the deposition fluxe from lakes to the 
void ModelRunner::drape_deposition_flux_to_chonks()
{

  xt::pytensor<double,1>& topography = this->io_double_array["topography"];
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];


  for(auto& loch:this->lakes)
  { 
    // Checking if this is a main lake
    if(loch.is_now >= 0)
      continue;
    if(loch.volume_sed == 0)
      continue;

    double ratio_of_dep = loch.volume_sed/loch.volume_water;

    // NEED TO DEAL WITH THAT BOBO
    if(ratio_of_dep>1)
      ratio_of_dep = 1;


    for(auto no:loch.nodes)
    {

      double slangh = ratio_of_dep * (topography[no] - surface_elevation[no]) / timestep;
      chonk_network[no].add_sediment_creation_flux(slangh);
      chonk_network[no].set_other_attribute_array("label_tracker", loch.label_prop);
      if(std::isfinite(slangh) == false)
      {
        std::cout << "ERROR::Cannot drape? " << slangh << " || " << ratio_of_dep << " || " << loch.volume_sed << " || " << loch.volume_water << std::endl;
        throw std::runtime_error("LakeDrapeError::Nan in the process");
      }

    }
  }

}

// Fill a lake with a certain amount of water for the first time of the run
void Lake::pour_water_in_lake(
  double water_volume,
  int originode,
  std::vector<int>& node_in_lake,
  std::vector<bool>& is_processed,
  xt::pytensor<int,1>& active_nodes,
  std::vector<Lake>& lake_network,
  xt::pytensor<double,1>& surface_elevation,
  xt::pytensor<double,1>& topography,
  NodeGraphV2& graph,
  double cellarea,
  double dt,
  std::vector<chonk>& chonk_network,
  double& Ql_out
  )
{ 

  std::cout << "Pouring " << water_volume << " water (rate = " << water_volume/dt << ") into " << this->lake_id << " from node " << originode << std::endl;


  std::cout << "Entering water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;


  if(water_volume < -1)
    throw std::runtime_error("NegWatPoured!!!");


  double save_entering_water = water_volume;
  double save_preexistingwater = this->volume;
  int n_labels = int(chonk_network[originode].get_other_attribute_array("label_tracker").size());

  // Some ongoing debugging
  // if(originode == 8371)
  //   std::cout << "processing the problem node W:" << chonk_network[originode].get_water_flux() << std::endl;

  // first cancelling the outlet to make sure I eventually find a new one, If I pour water into the lake I might merge with another one, etc
  this->outlet_node = -9999;

  double sum_this_fill = 0;

  // no matter if I am filling a new lake or an old one:
  // I am filling a vector of nodes already in teh system (Queue or lake)
  std::vector<bool> is_in_queue(node_in_lake.size(),false);
  for(auto nq:this->node_in_queue)
    is_in_queue[nq] = true;

  // First checking if this node is in a lake, if yes it means the lake has already been initialised and 
  // We are pouring water from another lake 
  if(node_in_lake[originode] == -1)
  {
    // NEW LAKE
    // Emplacing the node in the queue, It will be the first to be processed
    if(originode == 0)
      throw std::runtime_error("0 is originode...");

    depressionfiller.emplace( nodium( originode, surface_elevation[originode] ) );
    // This function fills an original lake, hence there is no lake depth yet:
    this->water_elevation = surface_elevation[originode];
    is_in_queue[originode] = true;
    // this->nodes.push_back(originode);
    // this->n_nodes ++;
  }



  // I am processing new nodes while I still have water OR still nodes upstream 
  // (if an outlet in encountered, the loop is breaked anually)
  // UPDATE_09_2020:: No idea what I meant by anually.
  // UPDATE_10_2020:: I meant manually, but with a typo.
  std::cout << "starting loop" << std::endl;
  while(depressionfiller.empty() == false && water_volume > 0 )
  {
    // std::cout << this->water_elevation << "||" << water_volume << std::endl;
    // Getting the next node and ...
    // (If this is the first time I fill a lake -> the first node is returned)
    // (If this is another node, it is jsut the next one the closest elevation to the water level)
    nodium next_node = this->depressionfiller.top();

    // ... removing it from the priority queue 
    this->depressionfiller.pop();


    // Initialising a dummy outlet, if this outlet becomes something I shall break the loop
    int outlet = -9999;

    // Adding the upstream neighbors to the queue and checking if there is an outlet node
    outlet = this->check_neighbors_for_outlet_or_existing_lakes(next_node, graph, node_in_lake, lake_network, surface_elevation,
     is_in_queue, active_nodes, chonk_network, topography);

    bool isinnodelist = std::find(this->nodes.begin(), this->nodes.end(), next_node.node) == this->nodes.end();
    // If I have an outlet, then the outlet node is positive
    if(outlet >= 0)
    {
      // this->water_elevation = next_node.elevation;

      // I therefore save it and break the loop
      // std::cout << "{" << outlet << "|" << this->outlet_node << "}";
      this->outlet_node = outlet;
      // and readding the node to the depression
      this->depressionfiller.emplace(next_node);


      // break;
    }



    //Decreasing water volume by filling teh lake
    double dV = this->n_nodes * cellarea * ( next_node.elevation - this->water_elevation );

    // if(isinnodelist == false)
    //   dV = 0;

    sum_this_fill += dV;
    // I SHOULD NOT HAVE TO DO THAT!!!! PROBABLY LINKED TO NUMERICAL UNSTABILITIES BUT STILL
    if(dV > - 1e-3 && dV < 0)
      dV = 0;

    if(dV<0)
    {
      std::cout << "Arg should not have neg filling::" << dV << std::endl;
      dV = 0;
      // std::string ljsdfld;
      // if(is_processed[next_node.node] == true)
      //   ljsdfld = "true";
      // else
      //   ljsdfld = "false";
      // std::cout << "DV::" << dV << " :: " << node_in_lake[next_node.node]  << " :: " << this->n_nodes << "::" << this->water_elevation << " :: " << next_node.elevation << " :: " << ljsdfld << "::" << outlet << std::endl;
      // throw std::runtime_error("negative dV lake filling");
    }

    water_volume -= dV;
    

    this->volume += dV;


    // The water elevation is the elevation of that Nodium object
    // (if 1st node -> elevation of the bottom of the depression)
    // (if other node -> lake water elevation)
    this->water_elevation = next_node.elevation;

    // Alright, what is hapenning here:
    // I sometimes ingest other lakes, but I do not consider the ingested ndoes in my n_nodes at ingestion
    // Why? because I would be rising a lot more nodes to water elevation than intended and artificially fill my lakes.
    // So I am adding tham after dealing wih dV
    if(this->ngested_nodes>0)
    {
      this->n_nodes += this->ngested_nodes;
      std::cout << "ADDED " <<  this->ngested_nodes << " AFTER DV AND INGESTING" << std::endl;
      this->ngested_nodes = 0;
    }

    if(outlet >= 0)
      break;
        // Otehr wise, I do not have an outlet and I can save this node as in depression
    if(isinnodelist)
    {
      this->nodes.push_back(next_node.node);
      
      this->n_nodes ++;
    }
    node_in_lake[next_node.node] = this->lake_id;
    
    // At this point I either have enough water to carry on or I stop the process
  }
  std::cout << "ending loop" << std::endl;


  double local_balance = (save_entering_water - water_volume)/dt -  sum_this_fill/dt;

  // if(save_entering_water = save_preexistingwater)

  std::cout << "LOCAL BALANCE SHOULD BE 0::" << local_balance << std::endl;
  std::cout << "After raw filling lake water volume is " << water_volume << " hence water flux is " <<  water_volume/dt << std::endl;
  std::cout << "Outletting in " << this->outlet_node << " AND I HAVE " << this->n_nodes << std::endl;;

  // if(this->outlet_node>=0)
  // {
  //   if(node_in_lake[this->outlet_node]>=0)
  //   {
  //     if(node_in_lake[this->outlet_node] == this->lake_id || node_in_lake[this->outlet_node] == this->get_parent_lake())
  //     {

  //       std::cout << "SSDF" << node_in_lake[this->outlet_node] << "||" << this->lake_id <<  "||" << this->nodes.size() << "||" << this->get_parent_lake() << std::endl;
  //       throw std::runtime_error("Fatal Error:: outletinlake");
  //     }
  //   }
  // }




  // checking that I did not overfilled my lake:
  if(water_volume < 0 && this->outlet_node <0)
  {
    double extra = abs(water_volume);
    // this->n_nodes -= 1;
    std::cout << this->nodes.size() - 1 << std::endl;
    int extra_node = this->nodes[this->nodes.size() - 1];
    std::cout << ":bulf" << std::endl;


    this->depressionfiller.emplace(nodium(extra_node,topography[extra_node]));
    this->nodes.erase(this->nodes.begin() + this->nodes.size() - 1);
    sum_this_fill -= extra;
    double dZ = extra / this->n_nodes / cellarea;


    this->water_elevation -= dZ;
    water_volume = 0;
    this->volume -= extra;
  }




  Ql_out += sum_this_fill/dt;

  // std::cout << "Water balance: " << this->volume - save_preexistingwater + water_volume << "should be equal to " << save_entering_water << std::endl;


  // Labelling the node in depression as belonging to this lake and saving their depth
  std::map<int,int> counting_nodes;
  for(auto Unot:this->nodes)
    counting_nodes[Unot] = 0;

  for(auto Unot:this->nodes)
  {
    counting_nodes[Unot]++;
    if(counting_nodes[Unot] >1)
      throw std::runtime_error("double_nodation_there");
    // std::cout << Unot << "|";
    double this_depth = this->water_elevation - surface_elevation[Unot];
    this->depths[Unot] = this_depth;
    topography[Unot] = surface_elevation[Unot] + this_depth;

    node_in_lake[Unot] = this->lake_id;
    double temp_watflux = chonk_network[Unot].get_water_flux();
    double temp_sedflux = chonk_network[Unot].get_sediment_flux();
    std::vector<double> oatlab = chonk_network[Unot].get_other_attribute_array("label_tracker");
    chonk_network[Unot].reset();
    chonk_network[Unot].set_water_flux(temp_watflux);
    chonk_network[Unot].set_sediment_flux(temp_sedflux,oatlab);
    chonk_network[Unot].initialise_local_label_tracker_in_sediment_flux(n_labels);
    // std::cout <<  chonk_network[Unot].get_sediment_creation_flux() << "||";
  }
    // std::cout << " in " << this->lake_id << std::endl;

  // std::cout << "Water volume left: " << water_volume << std::endl;
  // Transmitting the water flux to the SS receiver not in the lake
  if(water_volume > 0 && this->outlet_node >= 0)
  {

    // If the node is inactive, ie if its code is 0, the fluxes can escape the system and we stop it here
  
    // Otherwise: calculating the outflux: water_volume_remaining divided by the time step
    double out_water_rate = water_volume/(dt);

    // Getting all the receivers and the length to the oulet
    std::vector<int> receivers = graph.get_MF_receivers_at_node(this->outlet_node);
    std::vector<double> length = graph.get_MF_lengths_at_node(this->outlet_node);
    // And finding the steepest slope 
    int SS_ID = -9999; 
    double SS = 0; // hmmmm I may need to change this name
    for(size_t i=0; i<receivers.size(); i++)
    {
      int nodelakeid = node_in_lake[receivers[i]];

      if(nodelakeid > -1)
      {
        if( nodelakeid == this->lake_id  || lake_network[nodelakeid].get_parent_lake() == this->lake_id)
          continue;
      } 

      double elevA = surface_elevation[this->outlet_node];
      double elevB = surface_elevation[receivers[i]];
      int testlake = node_in_lake[this->outlet_node];
      if( testlake >= 0)
      {
        if(lake_network[testlake].get_parent_lake() >=0)
          testlake = lake_network[testlake].get_parent_lake();

        elevA += lake_network[testlake].get_lake_depth_at_node(this->outlet_node, node_in_lake);

      }
      testlake = node_in_lake[receivers[i]] ;

      if( testlake >= 0)
      {
        if(lake_network[testlake].get_parent_lake() >= 0)
          testlake = lake_network[testlake].get_parent_lake();

        elevB += lake_network[testlake].get_lake_depth_at_node(receivers[i], node_in_lake);

      }

      double this_slope = (elevA - elevB)/length[i];


      if(this_slope >= SS )
      {
        SS = this_slope;
        SS_ID = receivers[i];
        // std::cout << "HURE::" << SS_ID << "||" << nodelakeid << "||" << node_in_lake[SS_ID]  << std::endl;

      }
    }

    if(SS_ID < 0)
    {
      int sr = graph.get_Srec(this->outlet_node);
      int lsr = node_in_lake[sr];
      if(sr != this->outlet_node && lsr != this->lake_id && lake_network[lsr].get_parent_lake() != this->lake_id)
      {
        // std::cout << "HERE" << std::endl;
        SS_ID = sr;
        SS = 0;
      }
      // std::cout << "Warning::lake outlet is itself a lake bottom? is it normal?" << std::endl;
      // yes it can be: flat surfaces
      else
      {
        // std::cout << "HARE" << std::endl;
        if(node_in_lake[SS_ID] < 0)
          throw std::runtime_error("Outlet potential ambiguity");

        SS_ID = this->outlet_node;
        SS = 0.;
      }
    }

    // here I am checking if my receiver is directly a lake, in whihc case I put my outlet directly in this lake to trigger merge. It will be simpler that way
    int SSlid = node_in_lake[SS_ID];
    bool SSlid_happened = false;
    if(SSlid >= 0)
    {
      if(lake_network[SSlid].get_parent_lake() >= 0)
        SSlid = lake_network[SSlid].get_parent_lake();

      SSlid_happened = true;
      // and I add the outlet of this thingy to the lake
      this->n_nodes ++;
      // removing it from the depressionfiller queue while making sure I readd others (the node should be right at the top of the queue so it will not empty and refill the whole queue)
      std::vector<nodium> toreadd;
      nodium dat = this->depressionfiller.top();
      this->depressionfiller.pop();
      while(dat.node != this->outlet_node)
      {
        toreadd.push_back(dat);
        dat = this->depressionfiller.top();
        this->depressionfiller.pop();
      }
      for(auto dut:toreadd)
        this->depressionfiller.push(dut);

      // Formerly add the node to the lake
      if(std::find(this->nodes.begin(), this->nodes.end(), this->outlet_node) == this->nodes.end())
        this->nodes.push_back(this->outlet_node);

      this->depths[this->outlet_node] = this->water_elevation - surface_elevation[this->outlet_node]; // should be 0 here yo
      node_in_lake[this->outlet_node] = this->lake_id;

      this->outlet_node = SS_ID;

      double temp_watflux = chonk_network[this->outlet_node].get_water_flux();
      double temp_sedflux = chonk_network[this->outlet_node].get_sediment_flux();
      std::vector<double> oatlab = chonk_network[this->outlet_node].get_other_attribute_array("label_tracker");
      chonk_network[this->outlet_node].reset();
      chonk_network[this->outlet_node].set_water_flux(temp_watflux);
      chonk_network[this->outlet_node].set_sediment_flux(temp_sedflux,oatlab);
      chonk_network[this->outlet_node].initialise_local_label_tracker_in_sediment_flux(n_labels);
    }

    // resetting the outlet CHONK
    // if(is_processed[this->outlet_node])
    // {
    //   chonk_network[this->outlet_node].cancel_split_and_merge_in_receiving_chonks(chonk_network,graph,dt);
    // }

    this->outlet_chonk = chonk(-1, -1, false); //  this is creating a "fake" chonk so its id is -1
    this->outlet_chonk.reinitialise_moving_prep();
    this->outlet_chonk.initialise_local_label_tracker_in_sediment_flux( n_labels );
    // forcing the new water flux

    this->outlet_chonk.set_water_flux(out_water_rate);

    // Forcing receivers
    std::cout << "SS ID for routletting is " << SS_ID << " and should receive " << out_water_rate << std::endl;
    std::vector<int> rec = {SS_ID};
    std::vector<double> wwf = {1.};
    std::vector<double> wws = {1.};
    std::vector<double> Strec = {SS};

    // if(SS_ID == this->outlet_node)
    //   throw std::runtime_error("LakeReroutingError::Lake outlet is itself");


    this->outlet_chonk.external_moving_prep(rec,wwf,wws,Strec);
    if(chonk_utilities::has_duplicates(rec))
      throw std::runtime_error("DUPLICATES FOUND HERE #3");

    if(this->volume_of_sediment > this->volume)
    {
      double outsed = this->volume_of_sediment - this->volume;
      this->volume_of_sediment -= outsed;
      this->outlet_chonk.set_sediment_flux(0., this->lake_label_prop);
      this->outlet_chonk.add_to_sediment_flux(outsed, this->lake_label_prop);
    }
    else
    {
      std::vector<double> baluf_2 (chonk_network[originode].get_other_attribute_array("label_tracker").size(),0.);
      this->outlet_chonk.set_sediment_flux(0.,baluf_2);
    }



    if(active_nodes[this->outlet_node] == 0)
    {
      chonk_network[this->outlet_node] = this->outlet_chonk;
      chonk_network[this->outlet_node].set_current_location(this->outlet_node);
      chonk_network[this->outlet_node].reinitialise_moving_prep();
    }

    // ready for re calculation, but it needs to be in the env object
  

  }
  std::cout << "done with lake " << this->lake_id << " outlet is " << this->outlet_node << std::endl;


  return;
}

// This function checks all the neighbours of a pixel node and return -9999 if there is no receivers
// And the node index if there is a receiver
// 
int Lake::check_neighbors_for_outlet_or_existing_lakes(
  nodium& next_node, 
  NodeGraphV2& graph, 
  std::vector<int>& node_in_lake, 
  std::vector<Lake>& lake_network,
  xt::pytensor<double,1>& surface_elevation,
  std::vector<bool>& is_in_queue,
  xt::pytensor<int,1>& active_nodes,
  std::vector<chonk>& chonk_network,
  xt::pytensor<double,1>& topography
  )
{

  // Getting all neighbors: receivers AND donors
  // Ignore dummy, is just cause I is too lazy to overlaod my functions correctly
  // TODO::Overload your function correctly
  std::vector<int> neightbors; std::vector<double> dummy ; graph.get_D8_neighbors(next_node.node, active_nodes, neightbors, dummy);

  // std::cout << "IS THIS EVEN CALLED???:: " << next_node.node << std::endl;

  // No outlet so far
  int outlet = -9999;
  bool has_eaten = false;

  std::vector<int> rec_of_node = chonk_network[next_node.node].get_chonk_receivers_copy();

  // Checking all neighbours
  for(auto node : neightbors)
  {
    // std::cout << "From node " << next_node.node << " checking " << node << std::endl;
    // First checking if the node is already in the queue
    // If it is, well I do not need it right?
    // Right.
    if(is_in_queue[node]  || node  == next_node.node)
      continue;

    // Check if the neighbour is a lake, if it is, I am gathering the ID and the depths
    int lake_index = -1;
    if(node_in_lake[node] > -1)
    {
      lake_index = node_in_lake[node];
      // If my node is not in the queue but in the same lake (this happens when refilling the lake with more water)
      if(lake_index == this->lake_id || lake_network[lake_index].get_parent_lake() == this->lake_id )
        continue;
    }


    
    // lake depth
    double this_depth = 0.;
    // getting potentially inherited lake depth
    if(lake_index >= 0)
    {
      if(lake_network[lake_index].get_parent_lake()>=0)
        lake_index = lake_network[lake_index].get_parent_lake();
      this_depth = lake_network[lake_index].get_lake_depth_at_node(node, node_in_lake);
    }

    // It gives me the elevation to be considered
    double tested_elevation = surface_elevation[node] + this_depth;

    // However if there is another lake, and that his elevation is mine I am ingesting it
    // if the lake has a lower elevation, I am outletting in it
    // if the lake has greater elevation, I am considering this node as a potential donor
    if(lake_index > -1 && tested_elevation == next_node.elevation)
    {
      // Well, before drinking it I need to make sure that I did not already ddid it
      if(lake_network[lake_index].get_parent_lake() == this->lake_id)
        continue;

      // OK let's try to drink it 
      this->ingest_other_lake(lake_network[lake_index], node_in_lake, is_in_queue,lake_network,topography);
      has_eaten = true; 
      continue;
    }


    
    // If the node is at higher (or same) elevation than me water surface, I set it in the queue
    else if(tested_elevation >= next_node.elevation)
    {
      // std::cout << "depressionfiller ingests " << node << " from " << next_node.node << std::endl;
      this->depressionfiller.emplace(nodium(node,tested_elevation));
      // Making sure I mark it as queued
      is_in_queue[node] = true;
      // Adding the node to the list of nodes in me queue
      this->node_in_queue.push_back(node);
    }

    // Else, if not in queue and has lower elevation, then the current mother node IS an outlet
    else
    {

      int outlake = node_in_lake[next_node.node] ;
      // std::cout << "{" << outlake << "}";
       // In some rare cases the outlet is already labelled as in this lake: for example if I already processed the lake before and readd water or more convoluted situations where I had a single pixeld lake
      if(outlake < 0)
      {
        // std::cout <<"gabul1  " << next_node.node << std::endl;
        outlet = next_node.node;
      }
      else if(outlake != this->lake_id && lake_network[outlake].get_parent_lake() != this->lake_id)
      {
        // std::cout <<"gabul2 " << next_node.node << std::endl;
        outlet = next_node.node;
      }


      // IMPORTANT::not breaking the loop: I want to get all myneighbors in the queue for potential repouring water in the thingy
    }

    // Moving to the next neighbour
  }

  // if a lake ahs been ingested, it makes the outlet ambiguous and need reprocessing of this node
  if(has_eaten)
  {
    this->depressionfiller.emplace(next_node);
    outlet = -9999;
  }

  // // outlet is >= 0 -> tehre is an outlet
  // if(outlet>=0)
  //   std::cout << "POCHTRAC::" << outlet << "||" << node_in_lake[outlet] << "||" << this->lake_id << std::endl;

  if(active_nodes[next_node.node] == false)
  {
    outlet = next_node.node;
    
  }

  return outlet;
}

void  ModelRunner::find_underfilled_lakes_already_processed_and_give_water(int SS_ID, std::vector<bool>& is_processed )
{

  // Legaciatisation
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  double& dt = timestep;
  double cellarea = this->io_double["dx"] * this->io_double["dy"] ;

  int& n_elements = this->io_int["n_elements"];
  // Using an already processed vector to not readd
  std::vector<bool> to_reprocessed(n_elements,false);
  std::vector<int> traversal(n_elements,-9999); // -9999 is nodata
  std::unordered_map<int,double> to_add_in_lakes, to_add_in_lakes_sed_edition;
  std::unordered_map<int, std::vector<double> >to_add_in_lakes_sed_edition_but_label_tracker;
  traversal[0] = SS_ID;
  to_reprocessed[SS_ID] = true;
  // basically here I wneed to reprocess all nodes dowstream of that one!
  // if processed and not in lake -> reprocess
  // else: stop
  // First: graph traversal
  int n_reprocessed = 0;
  int next_test = traversal[0];
  int reading_ID = 1, writing_ID = 1;
  while(next_test != -9999)
  {
    // feeding the queues with the receivers
    std::vector<int> recs = graph.get_MF_receivers_at_node_no_rerouting(next_test);
    for (auto node:recs)
    {
      // if not preprocessed globally or jsut already in the queue yet: boom
      if(to_reprocessed[node] || is_processed[node] == false || node_in_lake[node] >= 0)
        continue;

      to_reprocessed[node] = true;
      traversal[writing_ID] = node;
      writing_ID++;
    }

    // Reading the next node in list
    next_test = traversal[reading_ID];
    reading_ID++;
    // will break here if next node is -9999
  }
  // std::cout << writing_ID << "||";


  //Reinitialising all the concerned nodes but the outlet to repropagate correctly the thingies 
  for(size_t i=1; i<writing_ID; i++)
    chonk_network[traversal[i]] = chonk(traversal[i],traversal[i],false);


  // reprocessing nodes
  for(int i=0; i<n_elements; i++)
  {

 
    int node = graph.get_MF_stack_at_i(i);

    if(to_reprocessed[node] == false)
      continue;


    this->manage_fluxes_before_moving_prep(chonk_network[node], this->label_array[node] );


    this->manage_move_prep(chonk_network[node]);
    // std::cout << "call from reproc" << std::endl;
    // std::cout << "call from reproc" << std::endl;
    this->manage_fluxes_after_moving_prep(chonk_network[node], this->label_array[node]);
    // std::cout << "called from reproc" << std::endl;

    // need to check if some nodes give in lake
    std::vector<int> to_ignore;
    std::vector<int>& crec = chonk_network[node].get_chonk_receivers();
    std::vector<double>& cwawe = chonk_network[node].get_chonk_water_weight();
    std::vector<double>& csedw = chonk_network[node].get_chonk_sediment_weight();
    for(size_t i=0; i<crec.size(); i++)
    {
      if(node_in_lake[crec[i]]>=0)
      {
        to_ignore.push_back(crec[i]);
        int lake_to_consider = node_in_lake[crec[i]];
        to_add_in_lakes[lake_to_consider] += cwawe[i] * chonk_network[crec[i]].get_water_flux() * dt;
        to_add_in_lakes_sed_edition[lake_to_consider] += chonk_network[crec[i]].get_sediment_flux() * csedw[i];
        if(to_add_in_lakes_sed_edition_but_label_tracker.count(lake_to_consider) == 0)
        {
          to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider] = chonk_network[crec[i]].get_other_attribute_array("label_tracker");
        }
        else
        {
    std::cout << "ka";
          to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider] = mix_two_proportions(to_add_in_lakes_sed_edition[lake_to_consider],
            to_add_in_lakes_sed_edition_but_label_tracker[lake_to_consider]
            ,chonk_network[crec[i]].get_sediment_flux() * csedw[i], chonk_network[crec[i]].get_other_attribute_array("label_tracker"));
    std::cout << "ren";
        }
      }
    }

    chonk_network[node].split_and_merge_in_receiving_chonks_ignore_some(chonk_network, graph, timestep, to_ignore);
  }

  for(auto x:to_add_in_lakes)
  {
    int lake_to_fill = x.first;
    if(lake_network[lake_to_fill].get_parent_lake()>=0)
      lake_to_fill = lake_network[lake_to_fill].get_parent_lake();
    double water_volume = x.second;
    std::cout << "dar";

    this->lake_network[lake_to_fill].pour_sediment_into_lake(to_add_in_lakes_sed_edition[lake_to_fill], to_add_in_lakes_sed_edition_but_label_tracker[lake_to_fill]);
    std::cout << "de";

    this->lake_network[lake_to_fill].pour_water_in_lake(water_volume,lake_network[lake_to_fill].get_lake_nodes()[0], // pouring water in a random nodfe in the lake, it does not matter
  node_in_lake, is_processed, active_nodes,lake_network, surface_elevation, this->io_double_array["topography"],graph, cellarea, timestep,chonk_network,this->Ql_out);
  
  }

}


double Lake::get_lake_depth_at_node(int node, std::vector<int>& node_in_lake)
{
  return this->depths[node];
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
  xt::pytensor<int,1> output = xt::zeros<int>({this->io_int["n_elements"]}) -1;

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    if(node_in_lake[i] >= 0)
    {
      if(this->lake_network[node_in_lake[i]].get_parent_lake() >= 0)
        output[i] = this->lake_network[node_in_lake[i]].get_parent_lake();
    }
  }
  return output;
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



//////////////////////////////////////////
/////////////////////////////////////////





#endif

