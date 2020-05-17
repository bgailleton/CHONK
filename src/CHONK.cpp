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

// This empty constructor is just there to have a default one.
void chonk::create(int tchonkID, int tcurrent_node, bool tmemory_saver)
{
  // Initialising the fluxes to 0 and other admin detail
  this->is_empty = true;
  this->erosion_flux = 0;
  this->water_flux = 0;
  this->sediment_flux = 0;
  this->lake_depth = 0;

  // required params
  this->chonkID = tchonkID;
  this->current_node = tcurrent_node;
  this->depression_solved_at_this_timestep = false;
  this->memory_saver = tmemory_saver;
}

// This empty my chonk to reduce it's memory usage
void chonk::reset()
{
  chonkID = -9999;
  is_empty = true;
  current_node = -9999;
  depression_solved_at_this_timestep = false;
  memory_saver = -9999;
  water_flux = -9999;
  erosion_flux = -9999;
  sediment_flux = -9999;

  receivers.clear();
  weigth_water_fluxes.clear();
  weigth_sediment_fluxes.clear();
  slope_to_rec.clear();
}

void chonk::finalise(NodeGraph graph)
{
    // registering the erosion for the pit
  if(graph.get_pits_ID_at_node(this->current_node)>=0)
  {
    graph.add_erosion_flux_at_node(this->current_node,this->erosion_flux);
    graph.add_deposition_flux_at_node(this->current_node,this->deposition_flux);
  }
}




//####################################################
//####################################################
//############ Split and Merge FUCTIONs ##############
//####################################################
//####################################################
//

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraph graph)
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
  this->finalise(graph);

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
void chonk::move_to_steepest_descent(xt::pytensor<double,1>& elevation, NodeGraph& graph, double dt, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres)
{
//   get_MF_stack_at_node
// get_MF_receivers_at_node
// get_MF_lengths_at_node
  // find the steepest descent node first
  //Initialising the checkers to minimum
  int steepest_rec = -9999;
  double steepest_S = -std::numeric_limits<double>::max();

  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);
  // looping through neighbors
  for(size_t i=0; i<8; i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor
    if(this_neightbor == -1)
      continue;

    // getting the slope
    double this_slope = these_lengths[i];
    // checking if the slope is higher
    if(this_slope>steepest_S)
    {
      steepest_rec = this_neightbor;
      steepest_S = this_slope;
    }
    // Mover to the next step
  }

  // If there is no neighbors: base-level and nothing happens
  if(steepest_rec == -9999)
  {
    return;
  }

  int pit_id = graph.get_pits_ID_at_node(current_node);

  if(pit_id>=0)
  {
    int pit_bottom = graph.get_pits_bottom_at_pit_ID(pit_id);
    if(current_node == pit_bottom)
    {
      // Need to deal with depressions here!!!!
      this->solve_depression_simple(graph,  dt, sed_height_tp1, surface_elevation, surface_elevation_tp1, Xres, Yres);
      // # I WANT TO STOP HERE IF THE DEPRESSION IS SOLVED!!!
      return;
    }
  }
  // There is a neighbor, let's save it
  receivers.push_back(steepest_rec);
  weigth_water_fluxes.push_back(1.);
  weigth_sediment_fluxes.push_back(1.);
  slope_to_rec.push_back(steepest_S); 
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
inline void chonk::inplace_only_drainage_area(double Xres, double Yres){this->water_flux += Xres * Yres;};

// Calculate discharge by adding simple precipitation modulator 
inline void chonk::inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += Xres * Yres * precipitation[current_node];};

// Reduce the waterflux by infiltrating some water
inline void chonk::inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration){this->water_flux -= Xres * Yres * infiltration[this->current_node];};



//########################################################################################
//########################################################################################
//############ Fluxes appliers in motion #################################################
//########################################################################################
//########################################################################################
// These funtions apply the fluxes modification while moving 
// this includes erosion, deposition, ...
// they need to take care of the motion

void chonk::active_simple_SPL(double n, double m, xt::pytensor<double,1>& K, double dt, double Xres, double Yres)
{
  // reinitialising erosion flux as this flux only depends on current conditions
  erosion_flux = 0;

  // saving the current sediment flux
  double presedflux = this->sediment_flux;
  std::vector<double> pre_sedfluxes;pre_sedfluxes.reserve(this->weigth_sediment_fluxes.size());
  for(auto v:this->weigth_sediment_fluxes)
    pre_sedfluxes.emplace_back(v*this->sediment_flux);

  // Calculation current fluxes
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    // calculating the flux
    double this_eflux = std::pow(this->water_flux * this->weigth_water_fluxes[i],m) * std::pow(this->slope_to_rec[i],n) * K[this->current_node] * dt;
    // stacking the erosion flux
    this->erosion_flux += this_eflux;
    // What has been eroded moves into the sediment flux
    this->sediment_flux += this_eflux * Xres * Yres;
    // recording the current flux
    pre_sedfluxes[i] += this_eflux * Xres * Yres;
  }

  // Now I need to recalcculate the sediment fluxes weights to each receivers
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    this->weigth_sediment_fluxes[i] = pre_sedfluxes[i]/this->sediment_flux;
  }

  // Done
  return;
}



//########################################################################################
//########################################################################################
//############ Depression solver #########################################################
//########################################################################################
//########################################################################################

struct FillNode
{
  /// @brief Elevation data.
  double elevation;
  /// @brief Row index value.
  int node;
};

//Overload the less than and greater than operators to consider Zeta data only
//N.B. Fill only needs greater than but less than useful for mdflow routing
//(I've coded this but not yet added to LSDRaster, it's only faster than presorting
//when applied to pretty large datasets).
bool operator>( const FillNode& lhs, const FillNode& rhs )
{
  return lhs.elevation > rhs.elevation;
}
bool operator<( const FillNode& lhs, const FillNode& rhs )
{
  return lhs.elevation < rhs.elevation;
}

void chonk::solve_depression_simple(NodeGraph graph, double dt, xt::pytensor<double,1>& sed_height_tp1, xt::pytensor<double,1>& surface_elevation,xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres)
{
  // identifying my pit attributes
  int pit_id = graph.get_pits_ID_at_node(current_node);
  int pit_outlet = graph.get_pits_outlet_at_pit_ID(pit_id);
  double volume = graph.get_pits_volume_at_pit_ID(pit_id);
  std::vector<int> nodes_in_pit = graph.get_pits_pixels_at_pit_ID(pit_id);
  double elevation_bottom = surface_elevation[current_node];
  double elevation_outlet = surface_elevation[pit_outlet];
  std::vector<int> underwater_nodes;
  


  // My receiver will be the pit outlet
  this->receivers.push_back(pit_outlet);

  // Dealing first with the water flux
  double total_water = this->water_flux * dt;
  double current_water_elev_from_pit_bottom = 0; 
  double current_water_elev = elevation_bottom; 

  // Initialising the water depth in lake /!\ DOES NOT CALCULATE THE WATER DEPTH FOR RIVER! JUST WITHIN DEPRESSIONS
  std::vector<double> lake_depths(nodes_in_pit.size(),0); 
  std::map<int,bool> is_in_queue_lake_depth;
  for(auto node:nodes_in_pit)
   is_in_queue_lake_depth[node] = false;

  // Laright Let's go 
  if(total_water<volume)
  {
    // If I have less water in total than my pit can handle, then it is 0
    weigth_water_fluxes.push_back(0.);
    // I need to calculate which nodes are under water here
    // # |Initialising a priority Queue here
    std::priority_queue< FillNode, std::vector<FillNode>, std::greater<FillNode> > PriorityQueue;
    FillNode working_node; working_node.node = this->current_node; working_node.elevation = surface_elevation[this->current_node];
    is_in_queue_lake_depth[working_node.node] = true;

    double temp_total_water = total_water;
    int n_units = 0;
    
    while(temp_total_water > 0)
    {
      // At this point the working node will be underwater whatev happens
      underwater_nodes.push_back(working_node.node);

      // Firs I am pushing all the neighboring nodes in the queue
      std::vector<int> nenodes = graph.get_MF_donors_at_node(working_node.node);
      for(auto node:nenodes)
      {
        if(is_in_queue_lake_depth[node] == true)
          continue;
        FillNode this_nenode; this_nenode.node = node; this_nenode.elevation = surface_elevation[node];
        PriorityQueue.push(this_nenode);
      }

      // Now dealing with the lake depth in taht node: the highest priority Fillnode
      FillNode lowest_neighbour = PriorityQueue.top();
      double elev_neighbour_relative = lowest_neighbour.elevation - elevation_bottom;
      temp_total_water -= n_units * Xres * Yres * (elev_neighbour_relative - current_water_elev_from_pit_bottom);
      current_water_elev_from_pit_bottom = elev_neighbour_relative;
      if(temp_total_water<0)
      {
        // Whoops A bit too much water poured in, need to regulate
        // # Calculating the extra volume
        double volume_over = -1 * temp_total_water;
        // # converting it in elevation
        double elev_regulator = volume_over/n_units * Xres * Yres;
        // Correcting the water elevation
        current_water_elev_from_pit_bottom -= elev_regulator;
      }

      // Preparing the next node to be (eventually) processed
      working_node = PriorityQueue.top();
      // Removing the new working node from the list
      PriorityQueue.pop();
    }
    current_water_elev += current_water_elev_from_pit_bottom;


  }
  else
  {
    // that's where it gets funny: I need to fall back to waht average discharge myt outlet will get
    this->water_flux = (total_water - volume)/dt;
    this->weigth_water_fluxes.push_back(1.);
    underwater_nodes = nodes_in_pit;
    current_water_elev = elevation_outlet;
    // meh actually that should do the trick 
  }

  // Now cancelling the erosion/deposition that happened in the lake
  for(auto node : underwater_nodes)
  {
    double this_efux = graph.get_erosion_flux_at_node(node);
    double this_depflux = graph.get_deposition_flux_at_node(node);
    // "Removing" the eroded sediments from the chonks at that node
    this->sediment_flux -= this_efux * Xres * Yres * dt;
    // "Re-adding" the sediments deposited from the chonks at that node
    this->sediment_flux += this_depflux * Xres * Yres * dt;
  }


  // now dealing with sediment flux
  if(volume < this->sediment_flux)
  {
    this->sediment_flux -= volume;
    this->weigth_sediment_fluxes.push_back(1.);
    // Filling the whole depression
    for(auto node:underwater_nodes)
    {
      sed_height_tp1[node] += current_water_elev_from_pit_bottom;
      surface_elevation_tp1[node] += current_water_elev_from_pit_bottom;
    }
  }
  else
  {
    // getting the proportion of the depression I can fill
    double prop_fill = this->sediment_flux/volume;
    for(auto node:underwater_nodes)
    {
      sed_height_tp1[node] += current_water_elev_from_pit_bottom * prop_fill;
      surface_elevation_tp1[node] += current_water_elev_from_pit_bottom * prop_fill;
    }
    this->sediment_flux = 0;
    this->weigth_sediment_fluxes.push_back(0);
  }

  // I want to keep record of that as I don't wat to erode if I did the depression
  this->depression_solved_at_this_timestep = true; 

}

#endif