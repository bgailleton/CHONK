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


// ######################################################################
// ######################################################################
// ######################################################################
// ###################### Model Runner ##################################
// ######################################################################
// ######################################################################

// Initialises the model object, actually Does not do much but is required.
void ModelRunner::create(double ttimestep,double tstart_time,std::vector<std::string> tordered_flux_methods, std::string tmove_method)
{
  // Getting all the attributes
  this->timestep = ttimestep;
  this->start_time = tstart_time;
  this->ordered_flux_methods = tordered_flux_methods;
  this->move_method = tmove_method;
  this->lake_solver = true;
}

// initialising the node graph and the chonk network
void ModelRunner::initiate_nodegraph()
{
  std::cout << "initiating nodegraph..." <<std::endl;
  // creating the nodegraph and preprocessing the depression nodes
  // this->graph = NodeGraph(this->io_int_array["pre_stack"],this->io_int_array["pre_rec"],this->io_int_array["post_rec"],this->io_int_array["post_stack"] , this->io_int_array["m_stack"], this->io_int_array2d["m_rec"],this->io_int_array2d["m_don"], 
  //   this->io_double_array["surface_elevation"], this->io_double_array2d["length"], this->io_double["x_min"], this->io_double["x_max"], this->io_double["y_min"], 
  //   this->io_double["y_max"], this->io_double["dx"], this->io_double["dy"], this->io_int["n_rows"], this->io_int["n_cols"], this->io_int["no_data"]);
  xt::pytensor<bool,1> active_nodes = xt::zeros<bool>({this->io_int_array["active_nodes"].size()});
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];

  for(size_t i =0; i<inctive_nodes.size(); i++)
  {
    int B = inctive_nodes[i];
    if(B==1)
      active_nodes[i] = true;
    else
      active_nodes[i] = false;
  }
  this->graph = NodeGraphV2(this->io_double_array["surface_elevation"], active_nodes,this->io_double["dx"], this->io_double["dy"],
this->io_int["n_rows"], this->io_int["n_cols"]);

  std::cout << "done, sorting few stuff around ..." << std::endl;
   // if(node == 2074)
    // {
      // std::vector<int> receivers = graph.get_MF_receivers_at_node(2074);
      // std::cout << "FLUBR::" << receivers.size() << std::endl;

      // for(auto trec:receivers)
      // {
      //   std::cout << "2074REC is " << trec << std::endl;
       // }
    // }

  // Chonkification
  if(this->chonk_network.size()>0)
  {
    this->chonk_network.clear();
  }
  this->chonk_network = std::vector<chonk>();
  this->chonk_network.reserve(size_t(this->io_int["n_elements"]));

  // filling my network with empty chonks
  for(size_t i=0; i<size_t(this->io_int["n_elements"]); i++)
  {
    this->chonk_network.emplace_back(chonk(int(i), int(i), false));
  }

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
  //# no nodes in lakes
  node_in_lake = std::vector<int>(this->io_int["n_elements"], -1);


  std::cout << "done with setting up graph stuff" <<  std::endl;
}

void ModelRunner::run()
{

  // Alright now I need to loop from top to bottom
  std::cout << "Starting the run" << std::endl;
  // Keeping track of which node is processed, for debugging and lake management
  std::vector<bool> is_processed(io_int["n_elements"],false);

  // Aliases
  xt::pytensor<int,1>& inctive_nodes = this->io_int_array["active_nodes"];
  xt::pytensor<double,1>&surface_elevation =  this->io_double_array["surface_elevation"];
  double cellarea = this->io_double["dx"] * this->io_double["dy"];
  for(int i=0; i<io_int["n_elements"]; i++)
  {

 
    int node = this->graph.get_MF_stack_at_i(i);
    is_processed[node] = true;

    this->manage_fluxes_before_moving_prep(this->chonk_network[node]);

    if(this->lake_solver)
    {
      if(this->graph.is_depression(node))
      {
        // incrementing a new lake
        this->lake_network.push_back(Lake(lake_incrementor));
        
        // Getting the total volume of water
        double water_volume = this->chonk_network[node].get_water_flux() * timestep;
        double sedvol  = this->chonk_network[node].get_sediment_flux();

        this->lake_network[lake_incrementor].pour_sediment_into_lake(sedvol);
        
        // adding the water into the lake
        this->lake_network[lake_incrementor].pour_water_in_lake(water_volume,node, 
        node_in_lake, is_processed, inctive_nodes,lake_network, surface_elevation,graph, cellarea, timestep,chonk_network);
    
        // getting potential outlet
        int outlet = this->lake_network[lake_incrementor].get_lake_outlet();

        if(outlet >= 0)
        {
          // then is outlet
          // I first need to reprocess outlet. Outlets are special nodes, I forced them to have a single outlet, because a lake will not spread water all around
          //#1: reprocess lake outlet, note that it has already be reconditionned in the lake functions
          this->manage_fluxes_after_moving_prep(this->chonk_network[outlet]);
          this->chonk_network[outlet].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
          //#2 Is special node, has only one receiver, need to get the checking from dat
          int node_to_check = this->chonk_network[outlet].get_chonk_receivers()[0];
          if(is_processed[node_to_check ])
          {
            // This node_to_check  has already been processed
            this->find_underfilled_lakes_already_processed_and_give_water(node_to_check , is_processed);
          }
        }

        // Done with lake management
        // Incrementing the lake ID
        lake_incrementor++;
        // And skip the end of the run
        continue;

      }
    }
    else
    {
      if(inctive_nodes[node])
      {
        int next_node = this->graph.get_Srec(node);
        if(graph.is_depression(next_node) == false)
          next_node = this->graph.get_Srec(next_node);

        this->chonk_network[next_node].add_to_water_flux(this->chonk_network[node].get_water_flux());
        if(is_processed[next_node] == true )
          throw std::runtime_error("FATAL_ERROR::NG24, node " + std::to_string(node) + " gives water to " + std::to_string(next_node) + " but is processed already");
      }
      continue;
    }

    // first step is to apply the right move method, to prepare the chonk to move
    this->manage_move_prep(this->chonk_network[node]);
    // std::cout << "prep" << std::endl;
    this->manage_fluxes_after_moving_prep(this->chonk_network[node]);
    // std::cout << "bite" << std::endl;
    this->chonk_network[node].split_and_merge_in_receiving_chonks(this->chonk_network, this->graph, this->io_double_array["surface_elevation_tp1"], io_double_array["sed_height_tp1"], this->timestep);
    // std::cout << "garg" << std::endl;
    
  }
  std::cout << "Ending the run" << std::endl;
  this->finalise(); //TODO

  


}


void ModelRunner::finalise()
{
  xt::pytensor<double,1>& surface_elevation_tp1 = this->io_double_array["surface_elevation_tp1"];
  xt::pytensor<double,1>& sed_height_tp1 = this->io_double_array["sed_height_tp1"];
  xt::pytensor<double,1> tlake_depth = xt::zeros<double>({size_t(this->io_int["n_elements"])});

  for(int i=0; i< this->io_int["n_elements"]; i++)
  {
    chonk& tchonk = this->chonk_network[i];
    surface_elevation_tp1[i] -= tchonk.get_erosion_flux() * timestep;
    surface_elevation_tp1[i] += tchonk.get_deposition_flux() * timestep;
    sed_height_tp1[i] -= tchonk.get_erosion_flux() * timestep;
    if(sed_height_tp1[i]<0)
      sed_height_tp1[i] = 0;
    sed_height_tp1[i] += tchonk.get_deposition_flux() * timestep;

    if(node_in_lake[i]>=0)
    {
      int lakeid = node_in_lake[i];
      double tdepth = lake_network[lakeid].get_lake_depth_at_node(i,node_in_lake);
      tlake_depth[i] = tdepth;
    }
    else
    {
      tlake_depth[i] = 0;
    }
  }

  this->io_double_array["lake_depth"] = tlake_depth;
}



void ModelRunner::manage_fluxes_before_moving_prep(chonk& this_chonk)
{

  std::map<std::string,int> intcorrespondance;
  intcorrespondance["drainage_area"] = 1;
  intcorrespondance["precipitation_discharge"] = 2;
  intcorrespondance["infiltration_discharge"] = 3;

  for(auto method:this->ordered_flux_methods)
  {
    if(method == "move")
      break;
    int this_case = intcorrespondance[method];

    switch(this_case)
    {
      case 1:
        this_chonk.inplace_only_drainage_area(this->io_double["dx"], this->io_double["dy"]);
        break;
      case 2:
        this_chonk.inplace_precipitation_discharge(this->io_double["dx"], this->io_double["dy"],this->io_double_array["precipitation"]);
        break;
      case 3:
        this_chonk.inplace_infiltration(this->io_double["dx"], this->io_double["dy"], this->io_double_array["infiltration"]);
        break;
    }

  }
}

void ModelRunner::manage_move_prep(chonk& this_chonk)
{

  std::map<std::string,int> intcorrespondance;
  intcorrespondance["D8"] = 1;
  intcorrespondance["D8_nodeps"] = 2;
  intcorrespondance["MF_fastscapelib"] = 3;
  intcorrespondance["MF_fastscapelib_threshold_SF"] = 4;
  int this_case = intcorrespondance[this->move_method];

  switch(this_case)
  {
    case 1:
      this_chonk.move_to_steepest_descent(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 2:
      this_chonk.move_to_steepest_descent_nodepression(this->graph, this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;

    case 3:
      this_chonk.move_MF_from_fastscapelib(this->graph, this->io_double_array2d["external_weigths_water"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
    case 4:
      this_chonk.move_MF_from_fastscapelib_threshold_SF(this->graph, this->io_double["threshold_single_flow"], this->timestep, this->io_double_array["sed_height"], this->io_double_array["sed_height_tp1"], 
   this->io_double_array["surface_elevation"],  this->io_double_array["surface_elevation_tp1"], this->io_double["dx"], this->io_double["dy"], chonk_network);
      break;
      
    default:
      std::cout << "WARNING::move method name unrecognised, not sure what will happen now, probably crash" << std::endl;
  }
}

void ModelRunner::manage_fluxes_after_moving_prep(chonk& this_chonk)
{
  bool has_moved = false;
  std::map<std::string,int> intcorrespondance;
  intcorrespondance["basic_SPIL"] = 1;
  for(auto method:this->ordered_flux_methods)
  {
    
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
        this_chonk.active_simple_SPL(this->io_double["SPIL_n"], this->io_double["SPIL_m"], this->io_double_array["erodibility_K"], this->timestep, this->io_double["dx"], this->io_double["dy"]);
        break;
    }
  }
  return;
}


//#################################################
//#################################################
//#################################################
//######### Solving depression ####################
//#################################################
//#################################################
//#################################################



// This function is the main algorithm managing depression solving
int ModelRunner::solve_depressionv2(int node)
{
  std::cout << "solving depression at node " << node << std::endl;
  // Gathering the depression rerouter, I'll need it to check when I reach another pit vs true base level
  // xt::pytensor<int,1>& depressions = this->io_int_array["depression_to_reroute"];
  xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];
  xt::pytensor<double,1>& lake_depth = this->io_double_array["lake_depth"];
  xt::pytensor<int,1>& active_nodes = this->io_int_array["active_nodes"];
  // I am making these aliases to avoid the cost of accessing the map element each nodes

  // Getting the total volume of water arriving in this depression: Q * dt
  double water_volume = this->chonk_network[node].get_water_flux() * timestep;
  // My first node to be processed is the pit
  nodium working_node( node, surface_elevation[node] + lake_depth[node]);

  // Initialising the priority queue I will be using to processed my nodes
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> > depressionfiller;
  // Adding my first node in it
  depressionfiller.push(working_node);
  
  // I'll be keeping track of how many and which nodes are in this depression
  int n_nodes_underwater = 0;
  std::vector<int> underwater_nodes;
  std::vector<bool> is_underwater(this->io_int["n_elements"], false);
  std::vector<bool> is_in_queue(this->io_int["n_elements"], false);
  is_in_queue[working_node.node] = true;

  // My current water level is the elevation of this node + eventually preexisting lake water from THIS timestep
  double current_water_level = surface_elevation[node] + lake_depth[node];
  // I will track my outlet status
  int potential_outlet = -9999;
  // Temp boolean I need to break the loop in some specific cases (e.g. an edge is reach and my water escapes)
  bool break_main_loop = false;
  // Now filling the depression, doing it while I have enough water to do so
  while(water_volume>0 && depressionfiller.empty() == false)
  {
    // At the start of this loop, my working node is in the depression
    n_nodes_underwater++;
    working_node =  depressionfiller.top();
    depressionfiller.pop();
    underwater_nodes.push_back(working_node.node);
    is_underwater[working_node.node] = true;

    if(active_nodes[working_node.node] == 0)
      break;

    // HEre, checking if this node outlets the depression
    // I am getting all the receivers of the current working node, i.e. the downslope neighbors of my current one
    std::vector<int> recnodes = graph.get_MF_receivers_at_node(working_node.node);
    std::vector<int> donodes = graph.get_MF_donors_at_node(working_node.node);
    recnodes.insert(recnodes.end(), donodes.begin(), donodes.end());
    for(auto nenode:recnodes)
    {
      double this_elev = surface_elevation[nenode] + lake_depth[nenode];
      if(this_elev>= current_water_level || lake_depth[nenode] > 0)
      {
        nodium next_node( nenode, this_elev);
        depressionfiller.push(next_node);
      }

    }
    
    if(break_main_loop)
    {
      current_water_level = surface_elevation[potential_outlet] + lake_depth[potential_outlet];

      break;
    }

    // for(auto don:donodes)
    // {
      
    // }

    if(depressionfiller.empty())
    {
      break;
    }

    nodium next_node = depressionfiller.top();

    water_volume -= n_nodes_underwater * this->io_double["dx"] * this->io_double["dy"] * (next_node.elevation + lake_depth[next_node.node] - current_water_level);

    current_water_level = next_node.elevation + lake_depth[next_node.node];

  }

  // // DEAL WITH EXTRA WATER HERE
  // if(potential_outlet>=0)
  // {

  // }


  for(auto unode:underwater_nodes)
  {
    lake_depth[unode] = current_water_level -  surface_elevation[unode];
  }

  return potential_outlet;

}


// This function processes every existing lakes and check if the new topography has an outlet for them
// if it does , it empty what needs to be emptied at the outlet
// whatever remains after all of this is added to each nodes separatedly which simulates mass-balance
// POSSIBLE OPTIMISATION::Recreating lake objects, but I would need a bit of refactoring the Lake object. Will seee if this is very slow
void ModelRunner::process_inherited_water()
{
  for(auto& tlake:this->lake_network)
  {
    // getting node underwater
    std::vector<int>& unodes = tlake.get_lake_nodes();
    if(unodes.size() == 0)
      continue;

    // going through all the nodes and gathering the potenial outlets
    // With the new topography, we can imagine a rare case whenre several breaches are opened in the lake within one timestep
    // It is unlikely that they would happened at the same time in real life so I just decide to drain it to the steepest descent one assuming it would arrive first
    // I have to say that this might be controversial, but also would only happen in very rare cases, even never to be fair
    std::set<int> outlets;

    xt::pytensor<double,1>& surface_elevation = this->io_double_array["surface_elevation"];

    for(auto node:unodes)
    {
        std::vector<int>& recs = graph.get_MF_receivers_at_node_no_rerouting(node);
        std::vector<int>& dons = graph.get_MF_donors_at_node(node);
        std::vector<int> neightbors;
        neightbors.reserve(neightbors.size() + dons.size());

        for (auto r:recs)
          neightbors.emplace_back(r);
        for (auto d:dons)
          neightbors.emplace_back(d);

      double elenode = surface_elevation[node] + tlake.get_lake_depth_at_node(node, node_in_lake);


      for(auto nenode:neightbors)
      {
        int this_lake_id = tlake.get_lake_id();
        if(this->node_in_lake[nenode] == this_lake_id)
          continue;
        double this_depth = 0.;

        if(this_lake_id >= 0)
          this_depth = this->lake_network[this_lake_id].get_lake_depth_at_node(nenode, node_in_lake);

        double tselev = this_depth + surface_elevation[nenode];
        if(tselev < elenode)
        {
          if(outlets.find(nenode) != outlets.end())
            continue;
          outlets.insert(nenode);
        }
      }
    }

    if(outlets.size() > 0)
    {
      double min_elev = std::numeric_limits<double>::max();
      int toutlet = -9999;
      for(auto node:outlets)
      {
        double this_elev = surface_elevation[node];
        // if(this->node_in_lake[nenode]>=0)
        //   this_elev += this->lake_network[this[node_in_lake[node]]].get_lake_depth_at_node(node, node_in_lake);
        if(this_elev < min_elev)
        {
          toutlet = node;
          min_elev = this_elev;
        }
      }
      // Now I need to remove the water from the lake
      double volume_to_transfer = 0;
      for(auto node:unodes)
      {
        double this_elevation  = surface_elevation[node] + tlake.get_lake_depth_at_node(node, node_in_lake);
        double delta_elevation = this_elevation - min_elev;
        double tvolume = delta_elevation * this->io_double["dx"] * this->io_double["dy"];
        tlake.set_lake_depth_at_node(node, min_elev);
        volume_to_transfer += tvolume;
      }

      tlake.set_lake_volume(tlake.get_lake_volume() - volume_to_transfer);
      this->chonk_network[toutlet].add_to_water_flux(volume_to_transfer/timestep);
    }

    // And finally refeeding the water to the rest
    for(auto node:unodes)
      this->chonk_network[node].add_to_water_flux( tlake.get_lake_depth_at_node(node, this->node_in_lake) * this->io_double["dx"] * this->io_double["dy"]);

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
    output[tchonk.get_current_location()] = tchonk.get_erosion_flux();
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
   std::vector<bool>& is_in_queue
   )
{
  // getting the attributes of the other lake
  std::vector<int>& these_nodes = other_lake.get_lake_nodes();
  std::vector<int>& these_node_in_queue = other_lake.get_lake_nodes_in_queue();
  std::unordered_map<int,double>& these_depths = other_lake.get_lake_depths();
  std::priority_queue< nodium, std::vector<nodium>, std::greater<nodium> >& this_PQ = other_lake.get_lake_priority_queue();
  // merging them into this lake while node forgetting to label them as visited and everything like that
  for(auto node:these_nodes)
  {
    this->nodes.push_back(node);
    node_in_lake[node] = this->lake_id;
  }

  for(auto node:these_node_in_queue)
  {
    this->node_in_queue.push_back(node);
    is_in_queue[node] = true;
  }

  this->depths.insert(these_depths.begin(), these_depths.end());

  this->n_nodes += other_lake.get_n_nodes();

  // Transferring the PQ (not ideal but meh...)
  while(this_PQ.empty() == false)
  {
    nodium this_nodium = this_PQ.top();
    is_in_queue[this_nodium.node] = true;
    this->depressionfiller.emplace(this_nodium);
    this_PQ.pop();
  }

  this->volume_of_sediment = other_lake.get_volume_of_sediment();

  // Deleting this lake and setting its parent lake
  int save_ID = other_lake.get_lake_id();
  Lake temp = Lake(save_ID);
  other_lake = temp;
  other_lake.set_parent_lake(this->lake_id);

  return;
}

void Lake::pour_sediment_into_lake(double sediment_volume)
{
  this->volume_of_sediment += sediment_volume;
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
  NodeGraphV2& graph,
  double cellarea,
  double dt,
  std::vector<chonk>& chonk_network
  )
{
  
    // no matter if I am filling a new lake or an old one:
  // I am filling a vector of nodes already in teh system (Queue or lake)
  std::vector<bool> is_in_queue(node_in_lake.size(),false);
  for(auto nq:this->node_in_queue)
    is_in_queue[nq] = true;

  // First checking if this node is in a lake, if yes it means the lake has already been initialised and 
  // We are pouring water from another lake 

  if(node_in_lake[originode] == -1)
  {
    // Emplacing the node in the queue, It will be the first to be processed
    depressionfiller.emplace( nodium( originode,surface_elevation[originode] ) );

    // this function fills an original lake, hence there is no lake depth yet:
    this->water_elevation = surface_elevation[originode];
    is_in_queue[originode] = true;

  }



  // I am processing new nodes while I still have water OR still nodes upstream (if an outlet in encountered, the loop is breaked anually)
  while(depressionfiller.empty() == false && water_volume > 0 )
  {
    // Getting the next node and ...
    nodium next_node = this->depressionfiller.top();
    this->water_elevation = next_node.elevation;
    // ... removing it from the priority queue 
    this->depressionfiller.pop();
    // Initialising a dummy outlet
    int outlet = -9999;

    // Adding the upstream neighbors to the queue and checking if there is an outlet node
    outlet = this->check_neighbors_for_outlet_or_existing_lakes(next_node, graph, node_in_lake, lake_network, surface_elevation, is_in_queue);

    // If I have an outlet, then the outlet node is positive
    if(outlet>=0)
    {
      // I therefore save it and break the loop
      this->outlet_node = outlet;
      // std::cout <<"OUTLET FOUND::" << this->outlet_node << std::endl;
      break;
    }

    // Otehr wise, I do not have an outlet and I can save this node as in depression
    this->nodes.push_back(next_node.node);
    this->n_nodes ++;

    //Decreasing water volume by filling teh lake
    double dV = this->n_nodes * cellarea * (this->water_elevation - next_node.elevation );
    water_volume -= dV;
    this->volume += dV;
    // At this point I either have enough water to carry on or I stop the process
  }

  // Labelling the node in depression as belonging to this lake and saving their depth
  for(auto Unot:nodes)
  {
    this->depths[Unot] = this->water_elevation - surface_elevation[Unot];
    node_in_lake[Unot] = this->lake_id;
  }


  // Transmitting the water flux to the SS receiver not in the lake
  if(water_volume > 0 && this->outlet_node >= 0)
  {
    // If the node is inactive, ie if its code is 0, the fluxes can escape the system and we stop it here
    if(active_nodes[this->outlet_node] > 0)
    {
      // Otherwise: calculating the outflux: water_volume_remaining divided by the time step
      double out_water_rate = water_volume/dt;
      // Getting all the receivers and the length to the oulet
      std::vector<int>& receivers = graph.get_MF_receivers_at_node_no_rerouting(this->outlet_node);
      std::vector<double>& length = graph.get_MF_lengths_at_node(this->outlet_node);
      // And finding the steepest slope 
      int SS_ID = -9999; 
      double SS = -9999; // hmmmm I may need to change this name
      for(size_t i=0; i<receivers.size(); i++)
      {

        double this_slope = (surface_elevation[this->outlet_node] - surface_elevation[receivers[i]])/length[i];
        if(this_slope>SS)
        {
          SS = this_slope;
          SS_ID = receivers[i];
        }
      }

      // temporary check, I shall delete it when I'll be sure of it
      if(SS_ID<0)
      {
        // std::cout << "Warning::lake outlet is itself a lake bottom? is it normal?" << std::endl;
        // yes it can be
        SS_ID = this->outlet_node;
        // throw std::runtime_error(" The lake has an outlet with no downlslope neighbors ??? This is not possible, check Lake::initial_lake_fill or warn Boris that it happened");
      }

      // resetting the outlet CHONK
      chonk_network[this->outlet_node].reinitialise_moving_prep();
      // forcing the new water flux
      chonk_network[this->outlet_node].set_water_flux(out_water_rate);
      // Forcing receivers
      std::vector<int> rec = {SS_ID};
      std::vector<double> wwf = {1.};
      std::vector<double> wws = {1.};
      std::vector<double> Strec = {SS};
      chonk_network[this->outlet_node].external_moving_prep(rec,wwf,wws,Strec);
      if(this->volume_of_sediment > this->volume)
      {
        double outsed = this->volume_of_sediment - this->volume;
        this->volume_of_sediment -= outsed;
        chonk_network[this->outlet_node].set_sediment_flux(outsed);
      }
      else
        chonk_network[this->outlet_node].set_sediment_flux(0.);

      // ready for re calculation, but it needs to be in the env object


    }

  }


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
  std::vector<bool>& is_in_queue
  )
{

  // Getting all neighbors: receivers AND donors
  std::vector<int>& recs = graph.get_MF_receivers_at_node_no_rerouting(next_node.node);
  std::vector<int>& dons = graph.get_MF_donors_at_node(next_node.node);
  std::vector<int> neightbors;
  neightbors.reserve(neightbors.size() + dons.size());

  for (auto r:recs)
    neightbors.emplace_back(r);
  for (auto d:dons)
    neightbors.emplace_back(d);

  int outlet = -9999;
  // checking all neighbors
  for(auto node : neightbors)
  {
    // Check if the neighbour is a lake, if it is, I am gathering the ID and the depths
    int lake_index = -1;
    if(node_in_lake[node] > -1)
      lake_index = node_in_lake[node];

    if(lake_index == this->lake_id || is_in_queue[node])
      continue;

    double this_depth = 0.;
    

    if(lake_index >= 0)
      this_depth = lake_network[lake_index].get_lake_depth_at_node(node, node_in_lake);


    // it gives me the elevation to be considered
    double tested_elevation = surface_elevation[node] + this_depth;

    // however if there is another lake, I am ingesting it
    if(lake_index> -1 )
    {
      this->ingest_other_lake(lake_network[lake_index], node_in_lake, is_in_queue);
      continue;
    }



    // If my neighbor is not in queue and at a higher elevation: I ingest it in the system
    if(tested_elevation >= this->water_elevation && is_in_queue[node] == false)
    {
      this->depressionfiller.emplace(nodium(node,tested_elevation));
      is_in_queue[node] = true;
      this->node_in_queue.push_back(node);
    }
    // Else, if not in queue and has lower elevation, then the current mother node IS an outlet
    else if(is_in_queue[node] == false)
    {
      this->outlet_node = next_node.node;
      // break; // not breaking the loop: I want to get all myneighbors in the queue for potential repouring water in the thingy
    }

    // Moving to the next neighbour
  }
  // outlet is >= 0 -> tehre is an outlet

  return this->outlet_node;
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
  traversal[0] = SS_ID;
  to_reprocessed[SS_ID] = true;
  // basically here I wneed to reprocess all nodes dowstream of that one!
  // if processed and not in lake -> reprocess
  // else: stop
  // First: graph traversal
  int next_test = traversal[0];
  int reading_ID = 1, writing_ID = 1;
  while(next_test != -9999)
  {
    // feeding the queues with the receivers
    std::vector<int>& recs = graph.get_MF_receivers_at_node_no_rerouting(next_test);
    for (auto node:recs)
    {
      // if not preprocessed globally or jsut already in the queue yet: boom
      if(to_reprocessed[node] || is_processed[node] == false || node_in_lake[node] >=0)
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


  //Reinitialising all the concerned nodes but the outlet to repropagate correctly the thingies 
  for(size_t i=1; i<writing_ID; i++)
    chonk_network[traversal[i]] = chonk(traversal[i],traversal[i],false);


  // reprocessing nodes
  for(int i=0; i<n_elements; i++)
  {

 
    int node = graph.get_MF_stack_at_i(i);

    if(to_reprocessed[node] == false)
      continue;


    this->manage_fluxes_before_moving_prep(chonk_network[node]);


    this->manage_move_prep(chonk_network[node]);
    this->manage_fluxes_after_moving_prep(chonk_network[node]);

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

    this->lake_network[lake_to_fill].pour_sediment_into_lake(to_add_in_lakes_sed_edition[lake_to_fill]);

    this->lake_network[lake_to_fill].pour_water_in_lake(water_volume,lake_network[lake_to_fill].get_lake_nodes()[0], // pouring water in a random nodfe in the lake, it does not matter
  node_in_lake, is_processed, active_nodes,lake_network, surface_elevation,graph, cellarea, timestep,chonk_network);
  
  }

}


double Lake::get_lake_depth_at_node(int node, std::vector<int>& node_in_lake)
{
  if(node_in_lake[node] == this->lake_id)
  {
    return depths[node];
  }
  return 0.;
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





#endif

