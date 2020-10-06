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
#include "nodegraph.hpp"


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
  this->other_attributes["height_lake_sediments_tp1"] = 0;

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
  this->other_attributes["height_lake_sediments_tp1"] = 0;

  this->receivers.clear();
  this->weigth_water_fluxes.clear();
  this->weigth_sediment_fluxes.clear();
  this->slope_to_rec.clear();
}



//####################################################
//####################################################
//############ Split and Merge FUCTIONs ##############
//####################################################
//####################################################
//

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, xt::pytensor<double,1>& surface_elevation_tp1, xt::pytensor<double,1>& sed_height_tp1, double dt)
{
  // KEEPING FOR LEGACY COMPATIBILITY
  this->split_and_merge_in_receiving_chonks(chonkscape, graph,  dt);
}

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt)
{
  // Iterating through the receivers
  std::vector<double> oatalab = other_attributes_arrays["label_tracker"];
  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];

    // Adding the fluxes*modifyer
    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);

    // So far the tracker gives equal proportion of its tracking downstream
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab);
    // std::cout << "SEDFLUXDEBUG::" << this->sediment_flux << "||" << this->weigth_sediment_fluxes[i] << "||water::" << this->weigth_water_fluxes[i] << std::endl;
  }

  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();
}

void chonk::cancel_split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt)
{
  // Iterating through the receivers
  std::vector<double> oatalab = other_attributes_arrays["label_tracker"];
  for(auto& oat:oatalab)
    oat = -1 * oat;

  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];

    // Adding the fluxes*modifye
    other_chonk.add_to_water_flux( -1 * this->water_flux * this->weigth_water_fluxes[i]);
    // std::cout << "Node " << this->chonkID << " removes " << -1 * this->water_flux * this->weigth_water_fluxes[i] << " from " << this->receivers[i] << " leaving " << other_chonk.get_water_flux() << std::endl;;
    // if(other_chonk.get_water_flux()<0)
    // {
    //   std::cout << "warning test:: watf<0" << std::endl;
    //   other_chonk.set_water_flux(0.);
    // }

    other_chonk.add_to_sediment_flux( -1 * this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab);
    // std::cout << "SEDFLUXDEBUG::" << this->sediment_flux << "||" << this->weigth_sediment_fluxes[i] << "||water::" << this->weigth_water_fluxes[i] << std::endl;
  }

  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();
}

void chonk::split_and_merge_in_receiving_chonks_ignore_some(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt, std::vector<int>& to_ignore)
{
  
  std::vector<double> oatalab = other_attributes_arrays["label_tracker"];

  // Iterating through the receivers
  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // if this is in the ignoring list then
    if(std::find(to_ignore.begin(), to_ignore.end(), this->receivers[i])!= to_ignore.end())
      continue;
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];
    // Adding the fluxes*modifyer
    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab);
  }

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
void chonk::move_to_steepest_descent(NodeGraphV2& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
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
  for(size_t i=0; i<these_neighbors.size(); i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor >= int(surface_elevation.size()) || this_neightbor == this->current_node)
    {
      continue;
    }


    all_minus_1 = false;

    // getting the slope, dz/dx
    double this_slope = 0;
    if((these_lengths[i] >= Xres) || (these_lengths[i] >= Yres))
      this_slope = (surface_elevation[this->current_node] - surface_elevation[this_neightbor]) / these_lengths[i];
    else
    {
      this_slope = 0;
    }


    // NEED TO CHECK WHY IT DDOES THAT!!
    if(this_slope<0)
    {
      this_slope = 0;
    }

    // checking if the slope is higher and recording the receiver
    if(this_slope>steepest_S)
    {
      steepest_rec = this_neightbor;
      steepest_S = this_slope;
    }
    // Mover to the next step
  }


  // Base level! i am stopping the code there and treating it as a depression already solved to inhibit all the process
  if(steepest_rec == this->current_node || all_minus_1 == true)
  {
    this->depression_solved_at_this_timestep = true;
    return;
  }

  // If there is no neighbors: True base-level and nothing happens
  if(steepest_rec == -9999)
  {
    // stoping the function by calling the return statement
    return; 
  }

  // // This is the part where I deal with topographic depression now
  // int pit_id = graph.get_pits_ID_at_node(current_node);

  // // Is this a pit?
  // if(pit_id>=0)
  // {
    
  //   // Apparently so, let's check if I am at the bottom
  //   int pit_bottom = graph.get_pits_bottom_at_pit_ID(pit_id);

  //   if(current_node == pit_bottom)
  //   {
  //     // Need to deal with depressions here!!!!
  //     this->solve_depression_simple(graph,  dt, sed_height, sed_height_tp1, surface_elevation, surface_elevation_tp1, Xres, Yres, chonk_network);
  //     // # I WANT TO STOP HERE IF THE DEPRESSION IS SOLVED!!!
  //     // # The depression solving routine takes care of the receivers and all
  //     return;
  //   }
  // }

  // There is a non-pit neighbor, let's save it with its attributes
  this->receivers.push_back(steepest_rec);
  this->weigth_water_fluxes.push_back(1.);
  this->weigth_sediment_fluxes.push_back(1.);
  this->slope_to_rec.push_back(steepest_S); 
}


// Function where the weights of splitting the water comes from fastscapelib p method
void chonk::move_MF_from_fastscapelib(NodeGraphV2& graph, xt::pytensor<double,2>& external_weigth_water_fluxes, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network)
{ 
  // std::cout << "1.1" << std::endl;

  // I need the receicing neighbours and the distance to them
  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);
  // std::cout << "1.2" << std::endl;
  if(these_neighbors.size() == 0)
    return;

  std::vector<double> waterweigths(these_neighbors.size());
  std::vector<double> powerslope(these_neighbors.size());
  double sumslopes = 0;
  if(these_neighbors.size() > 1)
  {
    for(size_t i=0; i< these_neighbors.size(); i++)
    {
      double this_slope = (surface_elevation[this->current_node] -  surface_elevation[these_neighbors[i]])/these_lengths[i];
      // std::cout << this_slope << std::endl;
      powerslope[i] = std::pow(this_slope,(0.5 + 0.6 * this_slope));
      sumslopes += powerslope[i];
    }
    for(size_t i=0; i< these_neighbors.size(); i++)
    {
      waterweigths[i] = powerslope[i]/sumslopes;
    }
  }
  else
  {
    waterweigths[0] = 1;

  }


  bool all_minus_1 = true;
  // looping through neighbors
  for(size_t i=0; i<these_neighbors.size(); i++)
  {
    // std::cout << "1.4" << std::endl;
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor == this->current_node)
    {
      continue;
    }

    // std::cout << "1.5" << this_neightbor << std::endl;

    all_minus_1 = false;

    // getting the slope, dz/dx
    double this_slope = 0;
    if((these_lengths[i] >= Xres) || (these_lengths[i] >= Yres))
      this_slope = (surface_elevation[this->current_node] - surface_elevation[this_neightbor]) / these_lengths[i];
    else
    {
      // update:: should not happen anymore
      this_slope = 0;
    }
    // std::cout << "1.7" << std::endl;



    // NEED TO CHECK WHY IT DDOES THAT!!
    // update:: should not happen anymore
    if(this_slope<0)
    {
      this_slope = 0;
    }


    // DEPRECATED
    // //IF I REACH PIT BOTTOM I WANNA STOP THE FUNCTION
    // // This is the part where I deal with topographic depression now
    // int pit_id = graph.get_pits_ID_at_node(current_node);

    // // Is this a pit?
    // if(pit_id>=0)
    // {
      
    //   // Apparently so, let's check if I am at the bottom
    //   int pit_bottom = graph.get_pits_bottom_at_pit_ID(pit_id);

    //   if(current_node == pit_bottom)
    //   {
    //     // Need to deal with depressions here!!!!
    //     this->solve_depression_simple(graph,  dt, sed_height, sed_height_tp1, surface_elevation, surface_elevation_tp1, Xres, Yres, chonk_network);
    //     // # I WANT TO STOP HERE IF THE DEPRESSION IS SOLVED!!!
    //     // # The depression solving routine takes care of the receivers and all
    //     return;
    //   }
    // }
    // std::cout << "1.8" << std::endl;


    // There is a non-pit neighbor, let's save it with its attributes
    double weight = waterweigths[i];
    // std::cout << "WATER WEIGHT " << weight << std::endl;
    this->receivers.push_back(this_neightbor);
    this->weigth_water_fluxes.push_back(weight);
    this->weigth_sediment_fluxes.push_back(weight);
    this->slope_to_rec.push_back(this_slope); 


    // Mover to the next step
  }

}


// Function where the weights of splitting the water comes from fastscapelib p method
void chonk::move_MF_from_fastscapelib_threshold_SF(NodeGraphV2& graph, double threshold_Q, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
  xt::pytensor<double,1>& surface_elevation, xt::pytensor<double,1>& surface_elevation_tp1, double Xres, double Yres, std::vector<chonk>& chonk_network)
{ 

  // I need the receicing neighbours and the distance to them
  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);
  if(these_neighbors.size() == 0)
    return;


  std::vector<double> waterweigths(these_neighbors.size());
  std::vector<double> powerslope(these_neighbors.size());
  double sumslopes = 0;
  double maxslope = -9999;
  int id_max_slope = 0;

  if(these_neighbors.size() > 1)
  {
    for(size_t i=0; i< these_neighbors.size(); i++)
    {

      double this_slope = (surface_elevation[this->current_node] -  surface_elevation[these_neighbors[i]])/these_lengths[i];
      if(this_slope>maxslope)
      {
        maxslope = this_slope;
        id_max_slope = i;
      }
      powerslope[i] = std::pow(this_slope,(0.5 + 0.6 * this_slope));
      sumslopes += powerslope[i];
    }
    for(size_t i=0; i< these_neighbors.size(); i++)
    {
      waterweigths[i] = powerslope[i]/sumslopes;
    }
  }
  else
  {
    waterweigths[0] = 1;
  }

  if(this->water_flux >= threshold_Q)
  {
    for(size_t i=0; i< waterweigths.size(); i++)
    {
      if(i==id_max_slope)
       waterweigths[i] = 1;
      else
        waterweigths[i] = 0;
    }
  }


  bool all_minus_1 = true;
  // looping through neighbors
  for(size_t i=0; i<these_neighbors.size(); i++)
  {
    // std::cout << "1.4" << std::endl;
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor == this->current_node)
    {
      continue;
    }

    // std::cout << "1.5" << this_neightbor << std::endl;

    all_minus_1 = false;

    // getting the slope, dz/dx
    double this_slope = 0;
    // std::cout << these_lengths.size() << std::endl;
    if((these_lengths[i] >= Xres) || (these_lengths[i] >= Yres))
      this_slope = (surface_elevation[this->current_node] - surface_elevation[this_neightbor]) / these_lengths[i];
    else
    {
      // update:: should not happen anymore
      this_slope = 0;
    }
    // std::cout << "1.7" << std::endl;



    // NEED TO CHECK WHY IT DDOES THAT!!
    // update:: should not happen anymore
    if(this_slope<0)
    {
      this_slope = 0;
    }


    // DEPRECATED
    // //IF I REACH PIT BOTTOM I WANNA STOP THE FUNCTION
    // // This is the part where I deal with topographic depression now
    // int pit_id = graph.get_pits_ID_at_node(current_node);

    // // Is this a pit?
    // if(pit_id>=0)
    // {
      
    //   // Apparently so, let's check if I am at the bottom
    //   int pit_bottom = graph.get_pits_bottom_at_pit_ID(pit_id);

    //   if(current_node == pit_bottom)
    //   {
    //     // Need to deal with depressions here!!!!
    //     this->solve_depression_simple(graph,  dt, sed_height, sed_height_tp1, surface_elevation, surface_elevation_tp1, Xres, Yres, chonk_network);
    //     // # I WANT TO STOP HERE IF THE DEPRESSION IS SOLVED!!!
    //     // # The depression solving routine takes care of the receivers and all
    //     return;
    //   }
    // }
    // std::cout << "1.8" << std::endl;


    // There is a non-pit neighbor, let's save it with its attributes
    double weight = waterweigths[i];
    // std::cout << "WATER WEIGHT " << weight << std::endl;
    this->receivers.push_back(this_neightbor);
    this->weigth_water_fluxes.push_back(weight);
    this->weigth_sediment_fluxes.push_back(weight);
    this->slope_to_rec.push_back(this_slope); 


    // Mover to the next step
  }


}




// Simplest function we can think of: move the thingy to 
void chonk::move_to_steepest_descent_nodepression(NodeGraphV2& graph, double dt, xt::pytensor<double,1>& sed_height, xt::pytensor<double,1>& sed_height_tp1, 
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
  for(size_t i=0; i<these_neighbors.size(); i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor >= int(surface_elevation.size()) || this_neightbor == this->current_node)
      continue;

    all_minus_1 = false;
    // getting the slope, dz/dx
    
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

    // checking if the slope is higher and recording the receiver
    if(this_slope>steepest_S)
    {
      steepest_rec = this_neightbor;
      steepest_S = this_slope;
    }
    // Mover to the next step
  }

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
      
void chonk::inplace_only_drainage_area(double Xres, double Yres){this->water_flux += Xres * Yres; }//; std::cout << this->water_flux << "||";};

// Calculate discharge by adding simple precipitation modulator 
void chonk::inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += Xres * Yres * precipitation[current_node];};

// Reduce the waterflux by infiltrating some water
void chonk::inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration){this->water_flux -= Xres * Yres * infiltration[this->current_node];};


void chonk::cancel_inplace_only_drainage_area(double Xres, double Yres){this->water_flux += -1 *(Xres * Yres); }//; std::cout << this->water_flux << "||";};

// Calculate discharge by adding simple precipitation modulator 
void chonk::cancel_inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += -1 *(Xres * Yres * precipitation[current_node]);};

// Reduce the waterflux by infiltrating some water
void chonk::cancel_inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration){this->water_flux -= -1 *(Xres * Yres * infiltration[this->current_node]);};

//########################################################################################
//########################################################################################
//############ Traking management: clever flux modifyers##################################
//########################################################################################
//########################################################################################
// manages the sediment fluxes

// Returns the proportion of sediment fluxes sent to the receivers
std::vector<double> chonk::get_preexisting_sediment_flux_by_receivers()
{
  std::vector<double> pre_sedfluxes;pre_sedfluxes.reserve(this->weigth_sediment_fluxes.size());
  for(auto& v:this->weigth_sediment_fluxes)
  {
    pre_sedfluxes.emplace_back(v*this->sediment_flux);
  }
  return pre_sedfluxes;
}

// Set the total sediment flux manually, with given proportions for each labels
// Index of the label array is the label, and the sum of the proportions should be 1
void chonk::set_sediment_flux(double value, std::vector<double> label_proportions)
{
  this->sediment_flux = value;
  std::vector<double>& oatalab = other_attributes_arrays["label_tracker"];
  for(int i=0; i< int(label_proportions.size()); i++)
  {  
    oatalab[i] = label_proportions[i];
  }
}

// Add a certain amount to the sediment flux
void chonk::add_to_sediment_flux(double value, std::vector<double> label_proportions)
{

  // First I am getting the proportions of stuff I wanna add
  double total_proportion = 1;
  // getting current proportions
  std::vector<double>& oatalab = other_attributes_arrays["label_tracker"];

  double sum_test_pre = 0.;
  for(auto val:oatalab)
    sum_test_pre += val;

  // If I already have sediments, then normalising the new proportions to add to the thingy
  if(this->sediment_flux>0)
  {    
    for(auto& prop:label_proportions)
    {
      double temp = prop;
      prop = temp * value / this->sediment_flux;
    }
  }
  else
  {
    // Otherwise these proportions shall be the new truth
    total_proportion = sum_test_pre;
  }

  // Applying it
  for(int i=0; i< int(label_proportions.size()); i++)
  {
    total_proportion += label_proportions[i];
    oatalab[i] += label_proportions[i];
  }

  for(int i=0; i< int(label_proportions.size()); i++)
  {
    if(total_proportion!=0)
      oatalab[i] = oatalab[i]/total_proportion; 
  }

  // Finally adding the full amount


  // keeping this debugging here in case
  // double sum_test = 0.;
  // for(auto val:oatalab)
  //   sum_test += val;

  // if(sum_test>1.1)
  // {
  //   double sum_acc = 0.;
  //   for(auto val:label_proportions)
  //     sum_acc += val;

  //   std::cout << sum_acc << "||" << sum_test << "||" << sum_test_pre << "||" << this->sediment_flux << std::endl;
  //   throw std::runtime_error("unfitted_proportion");
  // }
  this->sediment_flux += value;



}
// next step:: propagte proportions to split and merge





//########################################################################################
//########################################################################################
//############ Fluxes appliers in motion #################################################
//########################################################################################
//########################################################################################
// These funtions apply the fluxes modification while moving 
// this includes erosion, deposition, ...
// they need to take care of the motion

// Simplest Stream power incision formulation, Howard and Kerby 1984
void chonk::active_simple_SPL(double n, double m, double K, double dt, double Xres, double Yres, int label)
{

  // I am recording the current sediment fluxes in the model distributed for each receivers
  std::vector<double> pre_sedfluxes = get_preexisting_sediment_flux_by_receivers();
  // Calculation current fluxes
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    // calculating the flux E = K s^n A^m
    double this_eflux = std::pow(this->water_flux * this->weigth_water_fluxes[i],m) * std::pow(this->slope_to_rec[i],n) * K;
  
    // stacking the erosion flux
    this->erosion_flux += this_eflux;

    // What has been eroded moves into the sediment flux (which needs to be converted into a volume)
    std::vector<double> buluf(this->other_attributes_arrays["label_tracker"].size(), 0.);
    buluf[label] = 1.;
    this->add_to_sediment_flux(this_eflux * Xres * Yres * dt, buluf);
    // recording the current flux 
    pre_sedfluxes[i] += this_eflux * Xres * Yres * dt;

  }


  // Now I need to recalculate the sediment fluxes weights to each receivers
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    if(this->sediment_flux>0)
      this->weigth_sediment_fluxes[i] = pre_sedfluxes[i]/this->sediment_flux;
  }

  // Done
  return;
}


void chonk::charlie_I(double n, double m, double K_r, double K_s,
  double dimless_roughness, double this_sed_height, double V_param, 
  double d_star, double threshold_incision, double threshold_sed_entrainment,
  int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres)
{
   // I am recording the current sediment fluxes in the model distributed for each receivers
  std::vector<double> pre_sedfluxes = get_preexisting_sediment_flux_by_receivers();

  double Er_tot = 0;
  double Es_tot = 0;
  double Ds_tot = 0;

  double exp_sed_height_roughness =  std::exp(- this_sed_height / dimless_roughness);

  // Calculation current fluxes
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    double this_qw = this->water_flux * this->weigth_water_fluxes[i];
    double current_stream_power = std::pow(this_qw,m) * std::pow(this->slope_to_rec[i],n);
    // calculating the flux E = K s^n A^m
    double Er = (current_stream_power * K_r 
        * (1 - std::exp(- current_stream_power + threshold_incision) ) )
        * exp_sed_height_roughness;

    double Es = (current_stream_power * K_s
        * (1 - std::exp(- current_stream_power + threshold_sed_entrainment) ) )
        * (1 - exp_sed_height_roughness);
    
    double Ds = V_param * d_star * this->sediment_flux * this->weigth_sediment_fluxes[i];
    

    Er_tot += Er;
    Es_tot += Es;
    Ds_tot += Ds;



    // // stacking the erosion flux
    // this->erosion_flux += this_eflux;

    // // What has been eroded moves into the sediment flux (which needs to be converted into a volume)
    // std::vector<double> buluf(this->other_attributes_arrays["label_tracker"].size(), 0.);
    // buluf[label] = 1.;
    // this->add_to_sediment_flux(this_eflux * Xres * Yres * dt, buluf);
    // // recording the current flux 
    pre_sedfluxes[i] += (Er + Es - Ds) * Xres * Yres * dt;

  }

  // Now I need to recalculate the sediment fluxes weights to each receivers
  for(size_t i=0; i<this->receivers.size(); i++)
  {
    if(this->sediment_flux>0)
      this->weigth_sediment_fluxes[i] = pre_sedfluxes[i]/this->sediment_flux;
  }

  // Adding the eroded bedrock to the sediment flux
  std::vector<double> buluf(this->other_attributes_arrays["label_tracker"].size(), 0.);
  buluf[zone_label] = 1.;
  this->add_to_sediment_flux(Er_tot * Xres * Yres * dt, buluf);

  // Adding the sediment entrained into the sedimetn flux
  this->add_to_sediment_flux(Es_tot * Xres * Yres * dt, sed_label_prop);

  // removing the deposition from sediment flux
  this->add_to_sediment_flux(-1 * Ds_tot * Xres * Yres * dt, this->other_attributes_arrays["label_tracker"]);

  // Applying to the global fluxes
  this->erosion_flux += Er_tot + Es_tot;
  this->deposition_flux += Ds_tot;

  // Done
  return;
}





// //           .     .  .      +     .      .          .
// //      .       .      .     #       .           .
// //         .      .         ###            .      .      .
// //       .      .   "#:. .:##"##:. .:#"  .      .
// //           .      . "####"###"####"  .
// //        .     "#:.    .:#"###"#:.    .:#"  .        .       .
// //   .             "#########"#########"        .        .
// //         .    "#:.  "####"###"####"  .:#"   .       .
// //      .     .  "#######""##"##""#######"                  .
// //                 ."##"#####"#####"##"           .      .
// //     .   "#:. ...  .:##"###"###"##:.  ... .:#"     .
// //       .     "#######"##"#####"##"#######"      .     .
// //     .    .     "#####""#######""#####"    .      .
// //             .     "      000      "    .     .
// //        .         .   .   000     .        .       .
// // .. .. ..................O000O........................ ...... ...
// // Christmas tree because why not


#endif
