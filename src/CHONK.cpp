#ifndef CHONK_CPP
#define CHONK_CPP

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

#include "CHONK.hpp"
#include "cppintail.hpp"


//####################################################
//####################################################
//############ ADMIN FUCTIONs ########################
//####################################################
//####################################################
//

void chonk::create()
{
  this->reset();
}

// This empty constructor is just there to have a default one.
void chonk::create(int tchonkID, int tcurrent_node, bool tmemory_saver)
{
  // Initialising the fluxes to 0 and other admin detail
  this->is_empty = false;
  this->erosion_flux = 0;
  this->water_flux = 0;
  this->sediment_flux = 0;
  this->deposition_flux = 0;
  this->other_attributes["lake_depth"] = 0;

  // required params
  this->chonkID = tchonkID;
  this->current_node = tcurrent_node;
  this->depression_solved_at_this_timestep = false;
  this->memory_saver = tmemory_saver;
}

// This empty my chonk to reduce it's memory usage
void chonk::reset()
{
  this->is_empty = true;
  this->depression_solved_at_this_timestep = false;
  this->memory_saver = 0;
  this->water_flux = 0;
  this->erosion_flux = 0;
  this->sediment_flux = 0;
  this->other_attributes["lake_depth"] = 0;

  this->receivers.clear();
  this->weigth_water_fluxes.clear();
  this->weigth_sediment_fluxes.clear();
  this->slope_to_rec.clear();
}

void chonk::finalise(NodeGraph& graph, xt::pytensor<double,1>& surface_elevation_tp1, xt::pytensor<double,1>& sed_height_tp1, double dt)
{
  // registering the erosion for the pit
  if(graph.get_pits_ID_at_node(this->current_node)>=0)
  {
    graph.add_erosion_flux_at_node(this->current_node,this->erosion_flux);
    graph.add_deposition_flux_at_node(this->current_node,this->deposition_flux);
  }

  double before =  surface_elevation_tp1[this->current_node];

  surface_elevation_tp1[this->current_node] -= this->erosion_flux * dt;
  surface_elevation_tp1[this->current_node] += this->deposition_flux * dt;
  // if(surface_elevation_tp1[current_node]<0)
  // {
  //   std::cout << "BEFORE:" << before << " after:" << surface_elevation_tp1[this->current_node] << "err: " << this->erosion_flux * dt << " dep:" << this->deposition_flux * dt << " slope:" << slope_to_rec[0] << std::endl;
  //   exit(EXIT_SUCCESS);
  // }

  // if(isnan(surface_elevation_tp1[this->current_node]))
  // {
  //   std::cout<< "NAN DETECTED::errate:" << this->erosion_flux << " Dep:" << this->deposition_flux << std::endl;
  // }

  sed_height_tp1[this->current_node] -= this->erosion_flux * dt;
  if(sed_height_tp1[this->current_node] <0)
    sed_height_tp1[this->current_node] = 0;
  sed_height_tp1[this->current_node] += this->deposition_flux * dt;
}




//####################################################
//####################################################
//############ Split and Merge FUCTIONs ##############
//####################################################
//####################################################
//

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraph& graph, xt::pytensor<double,1>& surface_elevation_tp1, xt::pytensor<double,1>& sed_height_tp1, double dt)
{
  // Iterating through the receivers
  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];
    // Adding the fluxes*modifyer
    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i]);

  }

  // finaliseing the chonk at the end of its duty
  this->finalise(graph, surface_elevation_tp1, sed_height_tp1,dt);

  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();
}





//####################################################
//####################################################
//############ MOVE FUCTIONs ##########################
//####################################################
//####################################################
//


// Simplest function we can think of: move the thingy to 
void chonk::move_to_steepest_descent(NodeGraph& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network)
{
  // Find the steepest descent node first
  // Initialising the checkers to minimum
  int steepest_rec = -9999;
  double steepest_S = -std::numeric_limits<double>::max();
  // std::cout << "Start" << std::endl;
  // I need the receicing neighbours and the distance to them
  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);


  bool all_minus_1 = true;
  // std::cout << "1" << std::endl;
  // looping through neighbors
  for(size_t i=0; i<8; i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor >= int(surface_elevation.size()) || this_neightbor == this->current_node)
      continue;

    all_minus_1 = false;

    // getting the slope, dz/dx
    double this_slope = 0;
    if((these_lengths[i] >= Xres) || (these_lengths[i] >= Yres))
      this_slope = (surface_elevation[this->current_node] - surface_elevation[this_neightbor]) / these_lengths[i];
    else
      this_slope = 0;
    // NEED TO CHECK WHY IT DDOES THAT!!
    if(this_slope<0)
    {
      this_slope = 0;
    }


    // if(this_slope<0)
    // {
    //   std::cout << "Z:" << surface_elevation[this->current_node] << " Zrec:" << surface_elevation[this_neightbor] << " L: " << these_lengths[i] << std::endl;
    // }

    // checking if the slope is higher and recording the receiver
    if(this_slope>steepest_S)
    {
      steepest_rec = this_neightbor;
      steepest_S = this_slope;
    }
    // Mover to the next step
  }

  // std::cout << "2" << std::endl;


  // Base level! i am stopping the code there and treating it as a depression already solved to inhibit all the process
  if(steepest_rec == this->current_node || all_minus_1 == true)
  {
    this->depression_solved_at_this_timestep = true;
    // std::cout << "BL reached" << std::endl;
    return;
  }

  // If there is no neighbors: True base-level and nothing happens
  if(steepest_rec == -9999)
  {
    // stoping the function by calling the return statement
    return; 
  }
  // std::cout << "3" << std::endl;

  // This is the part where I deal with topographic depression now
  int pit_id = graph.get_pits_ID_at_node(current_node);
  // std::cout << "3.1::" << pit_id << std::endl;

  // Is this a pit?
  if(pit_id>=0)
  {
    
    // std::cout << "pit" << std::endl;
    // Apparently so, let's check if I am at the bottom
    int pit_bottom = graph.get_pits_bottom_at_pit_ID(pit_id);
    // std::cout << "3.2" << std::endl;
    if(current_node == pit_bottom)
    {
      // std::cout << "solving depression" << std::endl;
      // Need to deal with depressions here!!!!
      this->solve_depression_simple(graph,  dt, sed_height, sed_height_tp1, surface_elevation, surface_elevation_tp1, Xres, Yres, chonk_network);
      // # I WANT TO STOP HERE IF THE DEPRESSION IS SOLVED!!!
      // # The depression solving routine takes care of the receivers and all
      // std::cout << "done" << std::endl;
      return;
    }
  }
  // std::cout << "4" << std::endl;

  // std::cout << "done" << std::endl;
  // There is a non-pit neighbor, let's save it with its attributes
  this->receivers.push_back(steepest_rec);
  this->weigth_water_fluxes.push_back(1.);
  this->weigth_sediment_fluxes.push_back(1.);
  this->slope_to_rec.push_back(steepest_S); 
}


// Simplest function we can think of: move the thingy to 
void chonk::move_to_steepest_descent_nodepression(NodeGraph& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network)
{
  // Find the steepest descent node first
  // Initialising the checkers to minimum
  int steepest_rec = -9999;
  double steepest_S = -std::numeric_limits<double>::max();

  // I need the receicing neighbours and the distance to them
  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);
  bool all_minus_1 = true;
  // looping through neighbors
  for(size_t i=0; i<8; i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor >= int(surface_elevation.size()) || this_neightbor == this->current_node)
      continue;

    all_minus_1 = false;
    // getting the slope, dz/dx
    // std::cout << this_neightbor  << "||" << surface_elevation[this->current_node]<< std::endl;
    
    double this_slope = 0;
    if(these_lengths[i] >= Xres)
      this_slope = (surface_elevation[this->current_node] - surface_elevation[this_neightbor]) / these_lengths[i];
    else
      this_slope = 0;
    // NEED TO CHECK WHY IT DDOES THAT!!
    if(this_slope<0)
    {
      this_slope = 0;
    }

    // if(this_slope>10)
    // {
    //    std::cout << "Z:" << surface_elevation[this->current_node] << " Zrec:" << surface_elevation[this_neightbor] << " L: " << these_lengths[i] << std::endl;
    //    exit(EXIT_SUCCESS);
    // }

    // if(this_slope<0)
    // {
    //   std::cout << "Z:" << surface_elevation[this->current_node] << " Zrec:" << surface_elevation[this_neightbor] << " L: " << these_lengths[i] << std::endl;
    // }

    // checking if the slope is higher and recording the receiver
    if(this_slope>steepest_S)
    {
      steepest_rec = this_neightbor;
      steepest_S = this_slope;
    }
    // Mover to the next step
  }
  // std::cout << "2" << std::endl;

  // Base level! i am stopping the code there and treating it as a depression already solved to inhibit all the process
  if(steepest_rec == this->current_node || all_minus_1 == true)
  {
    this->depression_solved_at_this_timestep = true;
    return;
  }

  // There is a non-pit neighbor, let's save it with its attributes
  this->receivers.push_back(steepest_rec);
  this->weigth_water_fluxes.push_back(1.);
  this->weigth_sediment_fluxes.push_back(1.);
  this->slope_to_rec.push_back(steepest_S); 
}





//########################################################################################
//########################################################################################
//############ Fluxes appliers in places FUNCTIONs #######################################
//########################################################################################
//########################################################################################
// These functions apply modifications on fluxes in place, i.e before splitting and moving
// This includes infiltration, precipitation, drainage area, soil production ...
// As opposed to the modification of fluxes linked to the motions of the chonk which needs to be treated later (e,d,...) 

// Simplest scenario possible, the chonk accumulates the drainage area
void chonk::inplace_only_drainage_area(double Xres, double Yres){this->water_flux += Xres * Yres;};

// Calculate discharge by adding simple precipitation modulator 
void chonk::inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += Xres * Yres * precipitation[current_node];};

// Reduce the waterflux by infiltrating some water
void chonk::inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration){this->water_flux -= Xres * Yres * infiltration[this->current_node];};



//########################################################################################
//########################################################################################
//############ Fluxes appliers in motion #################################################
//########################################################################################
//########################################################################################
// These funtions apply the fluxes modification while moving 
// this includes erosion, deposition, ...
// they need to take care of the motion

// Simplest Stream power incision formulation, Howard and Kerby 1984
void chonk::active_simple_SPL(double n, double m, xt::pytensor<double,1>& K, double dt, double Xres, double Yres)
{

  // I am recording the current sediment fluxes in the model distributed for each receivers
  // std::cout << "1" << std::endl;
  std::vector<double> pre_sedfluxes;pre_sedfluxes.reserve(this->weigth_sediment_fluxes.size());
  for(auto v:this->weigth_sediment_fluxes)
  {
    pre_sedfluxes.emplace_back(v*this->sediment_flux);
  }
  // std::cout << "2" << std::endl;

  // Calculation current fluxes
  for(size_t i=0; i<this->receivers.size(); i++)
  {

    // calculating the flux E = K s^n A^m
    double this_eflux = std::pow(this->water_flux * this->weigth_water_fluxes[i],m) * std::pow(this->slope_to_rec[i],n) * K[this->current_node];

    if(isinf(this_eflux))
      std::cout << this->water_flux << "||" << this->weigth_water_fluxes[i] << "||" << this->slope_to_rec[i] << std::endl;

    // if(this_eflux<0)
    // {
    //   std::cout << "DEBUG::errate<0, WF:" << this->water_flux << " wWF:" << this->weigth_water_fluxes[i] << " S:" << this->slope_to_rec[i] << std::endl;
    // }

    // std::cout << "this_eflux::" << this_eflux << std::endl;
    // stacking the erosion flux
    this->erosion_flux += this_eflux;


    // What has been eroded moves into the sediment flux (which needs to be converted into a volume)
    this->sediment_flux += this_eflux * Xres * Yres * dt;

    // recording the current flux 
    pre_sedfluxes[i] += this_eflux * Xres * Yres * dt;

  }
  // std::cout << "3" << std::endl;


  // Now I need to recalculate the sediment fluxes weights to each receivers
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    if(this->sediment_flux>0)
      this->weigth_sediment_fluxes[i] = pre_sedfluxes[i]/this->sediment_flux;
  }

  // std::cout << "4" << std::endl;

  // Done
  return;
}



//########################################################################################
//########################################################################################
//############ Depression solver #########################################################
//########################################################################################
//########################################################################################

// This is an internal c++ structure I use to sort my nodes into a priority queue
struct FillNode
{
  /// @brief Elevation data.
  double elevation;
  /// @brief Row index value.
  int node;
};

// Managing the comparison operators for the fill node, to let the queue know I wanna compare it by elevation values
// lhs/rhs : left hand side, right hand side
bool operator>( const FillNode& lhs, const FillNode& rhs )
{
  return lhs.elevation > rhs.elevation;
}
bool operator<( const FillNode& lhs, const FillNode& rhs )
{
  return lhs.elevation < rhs.elevation;
}


// This is the first version of my depression solver. It fills the water ensuring mass balance
void chonk::solve_depression_simple(NodeGraph& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation,xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network)
{
  // std::cout << "start the depression stuff" << std::endl;
  // identifying my pit attributes
  // std::cout << "dep1" << std::endl;
  int pit_id = graph.get_pits_ID_at_node(current_node);
  int pit_outlet = graph.get_pits_outlet_at_pit_ID(pit_id);
  double volume = graph.get_pits_volume_at_pit_ID(pit_id);
  std::vector<int> nodes_in_pit = graph.get_pits_pixels_at_pit_ID(pit_id);
  double elevation_bottom = surface_elevation[current_node];
  double elevation_outlet = surface_elevation[pit_outlet];
  // std::cout << "dep2" << std::endl;

  // My receiver will be the pit outlet
  this->receivers.push_back(pit_outlet);
  this->slope_to_rec.push_back(0.);

  // std::cout << "Zout:" << elevation_bottom << " Zbottom:" << elevation_bottom << " Nnodes:" << nodes_in_pit.size() << " vol:" << volume << " node_bot:" << this->current_node << " node_out:" << pit_outlet << std::endl;

  // Dealing first with the water fluxes
  // # Volume of water available
  double total_water = this->water_flux * dt;
  // # lake water depths
  double current_water_elev_from_pit_bottom = 0; 
  // # lake water absolute elevation
  double current_water_elev = elevation_bottom;
  // # this will store the nodes that are part of the lake if the depression is not full 
  std::vector<int> underwater_nodes;

  // std::cout << "dep3" << std::endl;


  // Initialising the water depth in lake /!\ DOES NOT CALCULATE THE WATER DEPTH FOR RIVER! JUST WITHIN DEPRESSIONS
  // std::vector<double> lake_depths(nodes_in_pit.size(),0); 
  // std::cout << "Initialised" << std::endl;
  // Alright Let's go 
  // checker
  double last_total_water = total_water;
  int n_repetition = -1;
  if(total_water<volume && n_repetition<20)
  { 
    if(last_total_water == total_water)
      n_repetition++;

    last_total_water = total_water;
    // std::cout << "deptot<vol" << std::endl;
    // My depression is not filled with water, I need to calculate the extent of the lake and which nodes need to be treated as lake pixel
    // This is important: erosion/deposition/sediment fluxes for such nodes need to be back-calculated as riverr do not erode in a lake
    // This also needs to be optimised: I cannot afford highly iterative methods as depressions can be very large (e.g. endoreic basins)

    // std::cout << "not enough water, calculating the water volume" << std::endl;
    // # This hash table check if a node is already in the filling queue. if depression is incomplete
    std::map<int,bool> is_in_queue_lake_depth;
    for(auto node:nodes_in_pit) 
    {
      is_in_queue_lake_depth[node] = false;
    }

    // std::cout << "dep3.1" << std::endl;
    // First I am initialising a priority queue Data structure: PQ are a bit slower to sort data for small datasets, but it ensure that I only deal with the nodes I need and I believe this is faster (to test against presorted vectors of pit pixels)
    // # |Initialising a priority Queue here
    std::priority_queue< FillNode, std::vector<FillNode>, std::greater<FillNode> > PriorityQueue;
    // # My first working node will be the  bottom pit, ie this node
    FillNode working_node; working_node.node = this->current_node; working_node.elevation = surface_elevation[this->current_node];
    is_in_queue_lake_depth[working_node.node] = true;
    
    // If I have less water in total than my pit can handle, then it is 0
    this->weigth_water_fluxes.push_back(0.);

    // Next Step: calculating which nodes are under water
    // # I'll "use" my volume of water until it reaches 0
    double temp_total_water = total_water;
    // # To avoid any extra iteration, I save up the number of pixel currently under water to  gradually fill the lake 
    int n_units = 0;
    // std::cout << "dep3.2" << std::endl;
    // Starting the loop: filling the lake as long as I wstill have water to offer
    while(temp_total_water > 0)
    {
      // std::cout << temp_total_water << "||" << PriorityQueue.empty() << "||" << current_water_elev_from_pit_bottom << std::endl;
      // At this point the working node will be underwater whatev happens
      underwater_nodes.push_back(working_node.node);
      n_units++;

      // std::cout << "dep3.2.1" << std::endl;
      // First I am pushing all the donor nodes in the queue
      std::vector<int> nenodes = graph.get_MF_donors_at_node(working_node.node);
      for(auto node:nenodes)
      {
        // std::cout << "nodonor:" << node << std::endl;  
        if(is_in_queue_lake_depth[node] == true || graph.get_pits_ID_at_node(current_node) != pit_id || node <0 || node >= surface_elevation.size())
          continue;
        // std::cout << "nodonor:" << node << std::endl;  
        FillNode this_nenode; this_nenode.node = node; this_nenode.elevation = surface_elevation[node];
        PriorityQueue.push(this_nenode);
        is_in_queue_lake_depth[node] = true;
      }
// 
      // std::cout << "dep3.2.2" << std::endl;

      // Now dealing with the lake depth in taht node: the highest priority Fillnode
      // # This get the "top" priority node: i.e. the one with the closest higer elevation to all the nodes already in water
      if(PriorityQueue.empty())
        break;


      FillNode lowest_neighbour = PriorityQueue.top();
      PriorityQueue.pop();

      // std::cout << lowest_neighbour.elevation << std::endl;
      // if(n_units == 10)
      //   exit(EXIT_SUCCESS);
      // # Getting the relative elevation of the neighbor from the bottom of the repression
      double elev_neighbour_relative = lowest_neighbour.elevation - elevation_bottom;

      // # this is the important bit: I am adding JUST a layer of water between the current water level and the lowest higher neighbor 
      temp_total_water -= n_units * Xres * Yres * (elev_neighbour_relative - current_water_elev_from_pit_bottom);
      // # updating the current water level relative to the bottom
      current_water_elev_from_pit_bottom = elev_neighbour_relative;

      // std::cout << "dep3.2.3" << std::endl;

      // I need a last check before moving to the next node:
      // Have I used too much water
      if(temp_total_water<0)
      {
        // Whoops A bit too much water poured in, need to regulate
        // # Calculating the extra volume
        double volume_over = -1 * temp_total_water;
        // # converting it in elevation
        double elev_regulator = volume_over/(n_units * Xres * Yres);
        // # Correcting the water elevation
        current_water_elev_from_pit_bottom -= elev_regulator;
      }
      // std::cout << current_water_elev_from_pit_bottom<< std::endl;
      // std::cout << "dep3.2.4" << std::endl;
      // Preparing the next node to be (eventually) processed
      working_node = lowest_neighbour;
      // Removing the new working node from the list
      // Done, moving to the next if I still have water to spare
    }
    // std::cout << "dep3.3" << std::endl;

    // My current water level was the pit bottom's one
    current_water_elev += current_water_elev_from_pit_bottom;

  }
  else
  {
    // std::cout << "enough water, filling everything" << std::endl;
    // that's where it gets funny: I need to fall back to waht average discharge myt outlet will get
    this->water_flux = (total_water - volume)/dt;
    this->weigth_water_fluxes.push_back(1.);
    underwater_nodes = nodes_in_pit;
    current_water_elev = elevation_outlet;
    // meh actually that should do the trick 
  }

  // std::cout << "dep4" << std::endl;
    

  // std::cout << "cancelling erosion" << std::endl;
  // Moving to the next important step: I now know which nodes are under water
  // I need to "cancel" the erosion and deposition that happened on these nodes
  // as the lake process overwrite them
  for(auto node : underwater_nodes)
  {
    // i saved this information in the nodegraph, specifically for depressions
    double this_efux = graph.get_erosion_flux_at_node(node);
    double this_depflux = graph.get_deposition_flux_at_node(node);

    chonk_network[node].set_erosion_flux(0.);
    chonk_network[node].set_other_attribute("lake_depth",current_water_elev - surface_elevation[node]);

    // "Removing" the eroded sediments from the chonks at that node
    this->sediment_flux -= this_efux * Xres * Yres * dt;
    // "Re-adding" the sediments deposited from the chonks at that node
    this->sediment_flux += this_depflux * Xres * Yres * dt;
    // Correcting the t+1 elevations

    // if(surface_elevation_tp1[current_node]<0)
      // std::cout << "1" << std::endl;

    surface_elevation_tp1[node] = surface_elevation[node];
    // if(surface_elevation_tp1[current_node]<0)
      // std::cout << "2" << std::endl;
    
    // std::cout << "ELEV CORRECTED:" << surface_elevation_tp1[node] << std::endl; 
    // falling back to the previous amount of sediments
    sed_height_tp1[node] = sed_height[node];
  }
  // done with the back-calculation

  if(this->sediment_flux<0)
  {
    // std::cout << "YOU NEED TO SORT THAT BUG BORIS::ID7845BCD" << std::endl;
    this->sediment_flux = 0;
  }

  // std::cout << "Dealing with sediments" << std::endl;
  // now dealing with sediment flux
  if(volume < this->sediment_flux)
  {
    this->sediment_flux -= volume;
    this->weigth_sediment_fluxes.push_back(1.);
    // Filling the whole depression
    for(auto node:underwater_nodes)
    {
      sed_height_tp1[node] += current_water_elev_from_pit_bottom;
  
      // if(surface_elevation_tp1[current_node]<0)
        // std::cout << "3" << std::endl;
      surface_elevation_tp1[node] += current_water_elev_from_pit_bottom;
      // if(surface_elevation_tp1[current_node]<0)
        // std::cout << "4" << std::endl;

      

      // chonk_network[node].set_deposition_flux(current_water_elev_from_pit_bottom * Xres * Yres / dt);
    }
  }
  else
  {
    // getting the proportion of the depression I can fill
    double prop_fill = this->sediment_flux/volume;
    for(auto node:underwater_nodes)
    {
      // calculating the prop of sediment to add
      double this_addsed = (current_water_elev - surface_elevation[node]) * prop_fill;
      sed_height_tp1[node] += this_addsed;

      // if(surface_elevation_tp1[current_node]<0)
        // std::cout << "5" << std::endl;

      // surface_elevation_tp1[node] += this_addsed;
      // std::cout << "adsed:" << this_addsed << std::endl;

      // if(surface_elevation_tp1[current_node]<0)
        // std::cout << "6" << std::endl;

      // if(isnan(surface_elevation_tp1[node]))
      //   std::cout << "PROBLEM NAN AFTER sediment: 1:" << this_addsed << " 2:" << this->sediment_flux << " 2.5: " << volume << " 3:" << prop_fill << std::endl;

      // chonk_network[node].set_deposition_flux(this_addsed * Xres * Yres / dt);


    }
    this->sediment_flux = 0;
    this->weigth_sediment_fluxes.push_back(0);
  }
  // std::cout << "dep5" << std::endl;


  // I want to keep record of that as I don't wat to erode if I did the depression
  this->depression_solved_at_this_timestep = true; 
  // std::cout << current_water_elev_from_pit_bottom << std::endl;


}

#endif
