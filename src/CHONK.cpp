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

bool double_equals(double a, double b, double epsilon)
{
    return std::abs(a - b) < epsilon;
}



void chonk::create()
{
  this->reset();
}

// This empty constructor is just there to have a default one.
void chonk::create(int tchonkID, int tcurrent_node, bool tmemory_saver)
{
  // Initialising the fluxes to 0 and other admin detail
  this->is_empty = false;
  this->erosion_flux_undifferentiated = 0;
  this->erosion_flux_only_bedrock = 0;
  this->erosion_flux_only_sediments = 0;
  this->water_flux = 0;
  this->sediment_flux = 0;
  this->deposition_flux = 0;
  this->sediment_creation_flux = 0;
  this->fluvialprop_sedflux = 0;

  // required params
  this->chonkID = tchonkID;
  this->current_node = tcurrent_node;
  this->depression_solved_at_this_timestep = false;
  this->memory_saver = tmemory_saver;
  this->receivers.reserve(8);
  this->weigth_water_fluxes.reserve(8);
  this->weigth_sediment_fluxes.reserve(8);
  this->slope_to_rec.reserve(8);
}

// This empty my chonk to reduce it's memory usage
void chonk::reset()
{
  this->is_empty = true;
  this->depression_solved_at_this_timestep = false;
  this->memory_saver = 0;
  this->water_flux = 0;
  this->erosion_flux_undifferentiated = 0;
  this->erosion_flux_only_bedrock = 0;
  this->erosion_flux_only_sediments = 0;
  this->deposition_flux = 0;
  this->sediment_creation_flux = 0;
  this->sediment_flux = 0;
  this->fluvialprop_sedflux = 0;

  this->receivers.clear();
  this->weigth_water_fluxes.clear();
  this->weigth_sediment_fluxes.clear();
  this->slope_to_rec.clear();

  this->receivers.reserve(8);
  this->weigth_water_fluxes.reserve(8);
  this->weigth_sediment_fluxes.reserve(8);
  this->slope_to_rec.reserve(8);
}

void chonk::reset_sed_fluxes()
{
  this->erosion_flux_undifferentiated = 0;
  this->erosion_flux_only_bedrock = 0;
  this->erosion_flux_only_sediments = 0;
  this->deposition_flux = 0;
  this->sediment_creation_flux = 0;
  this->sediment_flux = 0;
  this->fluvialprop_sedflux = 0;
  this->weigth_sediment_fluxes = std::vector<double> (this->weigth_water_fluxes.size(),0.);
}

//####################################################
//####################################################
//############ Split and Merge FUCTIONs ##############
//####################################################
//####################################################
//

// void check_sediment_weights()
// {
//   double sumsedweights = 0;

// }

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, xt::pytensor<double,1>& surface_elevation_tp1, xt::pytensor<double,1>& sed_height_tp1, double dt)
{
  // KEEPING FOR LEGACY COMPATIBILITY
  this->split_and_merge_in_receiving_chonks(chonkscape, graph,  dt);
}

void chonk::split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt)
{
  // Iterating through the receivers
  std::vector<double> oatalab = this->label_tracker;
  double sum_weight_sed = 0;
  double sum_outwat = 0;

  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];

    // Adding the fluxes*modifyer
    // if(this->chonkID == 466)
    // {
    //   std::cout << "466 giving to " << this->receivers[i] << "||" << this->water_flux * this->weigth_water_fluxes[i] << std::endl;
    // }


    // std::cout << this->current_node << "GIVING " << this->water_flux * this->weigth_water_fluxes[i] << " to " << receivers[i] << std::endl;

    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);
    sum_outwat += this->water_flux * this->weigth_water_fluxes[i];
    // So far the tracker gives equal proportion of its tracking downstream
    // std::cout << "SP::" << this->sediment_flux  << " * " <<  this->weigth_sediment_fluxes[i] << std::endl;
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab, this->fluvialprop_sedflux);
    sum_weight_sed += this->weigth_sediment_fluxes[i] * this->sediment_flux ;
    // std::cout << "SEDFLUXDEBUG::" << this->sediment_flux << "||" << this->weigth_sediment_fluxes[i] << "||water::" << this->weigth_water_fluxes[i] << std::endl;
    // if(other_chonk.get_sediment_flux()/dt * other_chonk.get_fluvialprop_sedflux() > other_chonk.get_water_flux() )
    // {
    //   std::cout << "Splitting overwhelm::" << this->sediment_flux * this->weigth_sediment_fluxes[i] * this->fluvialprop_sedflux << " was given to other which now has " << \
    //   other_chonk.get_sediment_flux() * other_chonk.get_fluvialprop_sedflux() << " vs " << other_chonk.get_water_flux() * dt << std::endl;;
    // }

  }

  if(double_equals(sum_weight_sed,this->sediment_flux, 1e-3) == false && graph.is_border[this->current_node] == 'n')
    std::cout << "WARNING::Sediment balance problem : " << sum_weight_sed << "||" << this->sediment_flux << "||" << this->receivers.size() << std::endl;



  if(double_equals(this->water_flux, sum_outwat, 1e-3) == false && graph.is_border[this->current_node] == 'n')
  {
    // std::cout << "WARNING::OOOOOOOOOOOOOOOOOOOOOOOO " << this->water_flux<< " to start with, but " << sum_outwat << " got out. NodeID == " << this->chonkID << std::endl;
    // std::cout << "I had " <<  this->receivers.size() << " receivers:" << std::endl;
    // for (auto rec : this->receivers)
    //   std::cout << rec << "!";
    // std::cout << std::endl;
    // this->print_status();
    if(this->receivers.size() > 0)
      throw std::runtime_error("WaterFluxError::Some water is lost in the splitting process: " + std::to_string(sum_outwat) + " vs " + std::to_string(this->water_flux) );
    // std::cout << "GULUUUUUUUUUB::::::::::" << this->water_flux - sum_outwat << std::endl;
  }

  // if(graph.is_border[this->current_node] == 'n' && this->receivers.size() == 0)
  // {
  //   throw std::runtime_error("NoRecError::No receivers in splitting");
  // }

  for (auto rec : this->receivers)
  {
    if(this->current_node == rec)
      throw std::runtime_error("RecError::Node giving to itself yo!");
  }
  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();
}

void chonk::get_what_was_given_to(int to, double& water, double& sed, std::vector<double>& lapprop, double& proflu)
{
  water = 0;
  sed = 0;
  lapprop = this->label_tracker;
  proflu = this->fluvialprop_sedflux;
  for(size_t i = 0; i < this->receivers.size(); i++)
  {
    int r = this->receivers[i];
    if(r == to)
    {
      water += this->get_water_flux() * this->weigth_water_fluxes[i];
      sed += this->get_sediment_flux() * this->weigth_sediment_fluxes[i];
    }
  }

}

void chonk::cancel_split_and_merge_in_receiving_chonks(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt)
{
  // Iterating through the receivers
  std::vector<double> oatalab = this->label_tracker;
  // for(auto& oat:oatalab)
  //   oat = -1 * oat;

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

    // std::cout << "PALUF";
    other_chonk.add_to_sediment_flux( -1 * this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab, this->fluvialprop_sedflux);
    // std::cout << "FIN";


    // TO DO:: SORT THIS SHIT IS NOT NORMAL
    // if(other_chonk.get_sediment_flux() < 0)
    //   other_chonk.set_sediment_flux(0., oatalab);


    // std::cout << "SEDFLUXDEBUG::" << this->sediment_flux << "||" << this->weigth_sediment_fluxes[i] << "||water::" << this->weigth_water_fluxes[i] << std::endl;
  }

  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();
}

void chonk::split_and_merge_in_receiving_chonks_ignore_some(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt, std::vector<int>& to_ignore)
{
  
  std::vector<double> oatalab = this->label_tracker;

  // Iterating through the receivers
  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // if this is in the ignoring list then
    if(std::find(to_ignore.begin(), to_ignore.end(), this->receivers[i])!= to_ignore.end())
      continue;
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];
    // Adding the fluxes*modifyer
    // std::cout << this->chonkID << " gives to " << this->receivers[i] << " " << this->water_flux * this->weigth_water_fluxes[i] << " it had before " << chonkscape[this->receivers[i]].get_water_flux() << std::endl; 
    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);


    // std::cout << "COR";
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab, this->fluvialprop_sedflux);
    // std::cout << "kar";
  }

  // and kill this chonk is memory saving is activated
  if(memory_saver)
    this->reset();

}

void chonk::split_and_merge_in_receiving_chonks_ignore_but_one(std::vector<chonk>& chonkscape, NodeGraphV2& graph, double dt, int the_one)
{
  
  std::vector<double> oatalab = this->label_tracker;

  // Iterating through the receivers
  for(size_t i=0; i < this->receivers.size(); i++)
  {
    // if this is in the ignoring list then
    if(this->receivers[i] != the_one)
      continue;
    // Adressing the chonk
    chonk& other_chonk = chonkscape[this->receivers[i]];
    // Adding the fluxes*modifyer
    // std::cout << this->chonkID << " gives to " << this->receivers[i] << " " << this->water_flux * this->weigth_water_fluxes[i] << " it had before " << chonkscape[this->receivers[i]].get_water_flux() << std::endl; 
    other_chonk.add_to_water_flux(this->water_flux * this->weigth_water_fluxes[i]);


    // std::cout << "COR";
    other_chonk.add_to_sediment_flux(this->sediment_flux * this->weigth_sediment_fluxes[i], oatalab, this->fluvialprop_sedflux);
    // std::cout << "kar";
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



// Simplest function we can think of: move the thingy to 
void chonk::move_to_steepest_descent(NodeGraphV2& graph, double dt, xt::pytensor<double,1>& surface_elevation, double Xres, double Yres, std::vector<chonk>& chonk_network)
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


  if(all_minus_1 && graph.get_Srec(this->current_node) != this->current_node)
  {
    // steepest_rec = graph.get_Srec(this->current_node);
    // steepest_S = 0.;
    return;
  }


  // Base level! i am stopping the code there and treating it as a depression already solved to inhibit all the process
  if(steepest_rec == this->current_node)
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

  this->receivers.emplace_back(steepest_rec);
  this->weigth_water_fluxes.emplace_back(1.);
  this->weigth_sediment_fluxes.emplace_back(0.);
  this->slope_to_rec.emplace_back(steepest_S); 
}


// Function where the weights of splitting the water comes from fastscapelib p method
void chonk::move_MF_from_fastscapelib(NodeGraphV2& graph, xt::pytensor<double,2>& external_weigth_water_fluxes, double dt,  
  xt::pytensor<double,1>& surface_elevation, double Xres, double Yres, std::vector<chonk>& chonk_network)
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
    this->receivers.emplace_back(this_neightbor);
    this->weigth_water_fluxes.emplace_back(weight);
    this->weigth_sediment_fluxes.emplace_back( 0. );
    this->slope_to_rec.emplace_back(this_slope); 


    // Mover to the next step
  }

}


// Function where the weights of splitting the water comes from fastscapelib p method
void chonk::move_MF_from_fastscapelib_threshold_SF(NodeGraphV2& graph, double threshold_Q, double dt,
  xt::pytensor<double,1>& surface_elevation, double Xres, double Yres, std::vector<chonk>& chonk_network)
{ 

  // I need the receicing neighbours and the distance to them
  std::vector<int> these_neighbors = graph.get_MF_receivers_at_node(this->current_node);
  std::vector<double> these_lengths = graph.get_MF_lengths_at_node(this->current_node);
  
  // No receiver? No prep
  if(these_neighbors.size() == 0)
    return;

  // Temporary vectors for the flow partitionning
  std::vector<double> waterweigths(these_neighbors.size());
  std::vector<double> powerslope(these_neighbors.size());

  // Sum, max and ID of the slope of interest
  double sumslopes = 0;
  int avger = 0;
  double maxslope = -9999;
  int id_max_slope = 0;

  // If I only have one receiver -> weight is 1
  if(these_neighbors.size() > 1)
  {
    // Calculating the slope for each rec
    for(size_t i=0; i< these_neighbors.size(); i++)
    {

      // Local slope
      double this_slope = (surface_elevation[this->current_node] -  surface_elevation[these_neighbors[i]])/these_lengths[i];

      // Saving the max slope and its ID
      if(this_slope>maxslope)
      {
        maxslope = this_slope;
        id_max_slope = i;
      }

      // Important checker to avoid slope == 0
      if( this_slope < 1e-6)
        this_slope = 1e-6;

      // Powerslope starts by being the slope
      powerslope[i] = this_slope;
      // umming the slope
      sumslopes += this_slope;
      // n slopes ++
      avger++;
    }

    // Getting the average slope
    double avgslope = sumslopes/avger;

    // Reinitialising the summer for other purposes
    sumslopes = 0;

    // if My average slope is 0 I skip (not possible anymore ??!!)
    if(avgslope > 0)
    {
      // Iterating through the rec and ...
      for(size_t i=0; i< these_neighbors.size(); i++)
      {
        // ... transforming the power into the power of itself to 0.5 + 0.6 * average slope -> see Jean Braun for explanation
        powerslope[i] = std::pow(powerslope[i],(0.5 + 0.6 * avgslope));
        // Summing the powerslopes
        sumslopes += powerslope[i];
      }
    }

    // DEBUGGING VARIABLE to catch few exceptions. Will delete after a while
    double sumcheique = 0;

    // If all the powers are not 0
    if(sumslopes >0)
    {
      // iterating through the rec to finally get the weight
      for(size_t i=0; i< these_neighbors.size(); i++)
      {
        waterweigths[i] = powerslope[i]/sumslopes;
        sumcheique += waterweigths[i];
      }
    }
    else
    {
      // if all is 0: equal partitionning though the receivers
      for(size_t i=0; i< these_neighbors.size(); i++)
      {
        waterweigths[i] = 1/int(these_neighbors.size());
        sumcheique += waterweigths[i];
      }
    }

    // DEBUG CHECKER --  I'll delete after a bit of time to make sure the algorithm is stable
    if(double_equals(sumcheique,1., 1e-4) == false)
    {
      std::cout << this->chonkID << "||" << surface_elevation[this->chonkID] << " Gulug::!!!!!" << these_neighbors.size() << std::endl;;
      for(size_t i=0; i< these_neighbors.size(); i++)
      {
        std::cout << "gaft::" << these_neighbors[i] << std::endl;
        std::cout << "Z::" << surface_elevation[these_neighbors[i]] << std::endl;
        std::cout << "WWW:::" << waterweigths[i] << std::endl;
        std::cout << "POWERSLOPE:::" << powerslope[i] << std::endl;
      }
      // throw std::runtime_error("SUMCHECK not 1???::" + std::to_string(sumcheique));
    }
    //-------------------------------------
  }
  else
  {
    // 1 rec -> weight is 1 
    waterweigths[0] = 1;
  }

  // Checking if I am above the flow threshold or not, to switch to single flow
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


  // Finally giving the info to the CHONK
  bool all_minus_1 = true;
  // looping through neighbors
  for(size_t i=0; i<these_neighbors.size(); i++)
  {
    int this_neightbor = these_neighbors[i];
    // checking if this is a neighbor, nodata will be -1 (fastscapelib standards)
    if(this_neightbor < 0 || this_neightbor == this->current_node)
    {
      continue;
    }

    all_minus_1 = false;

    double weight = waterweigths[i];

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

    // update:: should not happen anymore
    if(this_slope<0)
    {
      this_slope = 0;
    }




    this->receivers.emplace_back(this_neightbor);
    this->weigth_water_fluxes.emplace_back(weight);
    this->weigth_sediment_fluxes.emplace_back( 0 );
    this->slope_to_rec.emplace_back(this_slope); 


    // Mover to the next step
  }

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
double chonk::inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += Xres * Yres * precipitation[current_node]; return Xres * Yres * precipitation[current_node];};

// Reduce the waterflux by infiltrating some water
void chonk::inplace_infiltration(double Xres, double Yres, xt::pytensor<double,1>& infiltration){this->water_flux -= Xres * Yres * infiltration[this->current_node];};


void chonk::cancel_inplace_only_drainage_area(double Xres, double Yres){this->water_flux += -1 *(Xres * Yres); }//; std::cout << this->water_flux << "||";};

// Calculate discharge by adding simple precipitation modulator 
double chonk::cancel_inplace_precipitation_discharge(double Xres, double Yres, xt::pytensor<double,1>& precipitation){this->water_flux += -1 *(Xres * Yres * precipitation[current_node]); return Xres * Yres * precipitation[current_node];};

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

// Returns the proportion of fluvial sediment fluxes sent to the receivers
std::vector<double> chonk::get_preexisting_sediment_flux_by_receivers_fluvial()
{
  std::vector<double> pre_sedfluxes; pre_sedfluxes.reserve(this->weigth_sediment_fluxes.size());
  for(auto& v:this->weigth_sediment_fluxes)
  {
    pre_sedfluxes.emplace_back(v*this->sediment_flux * this->fluvialprop_sedflux);
  }
  return pre_sedfluxes;
}

// Returns the proportion of hillslopes sediment fluxes sent to the receivers
std::vector<double> chonk::get_preexisting_sediment_flux_by_receivers_hillslopes()
{
  std::vector<double> pre_sedfluxes; pre_sedfluxes.reserve(this->weigth_sediment_fluxes.size());
  for(auto& v:this->weigth_sediment_fluxes)
  {
    pre_sedfluxes.emplace_back(v * this->sediment_flux * (1 - this->fluvialprop_sedflux));
  }
  return pre_sedfluxes;
}


// Set the total sediment flux manually, with given proportions for each labels
// Index of the label array is the label, and the sum of the proportions should be 1
void chonk::set_sediment_flux(double value, std::vector<double> label_proportions, double prop_fluvial)
{
  if(value< 0)
  {
    std::cout << " WARNING:negsed:" <<  value << std::endl;
    value = 0;
    // throw std::runtime_error("SETTER sed to neg>>");
  }

  this->sediment_flux = value;
  std::vector<double>& oatalab = this->label_tracker;
  for(int i=0; i< int(label_proportions.size()); i++)
  {  
    oatalab[i] = label_proportions[i];
  }
  this->fluvialprop_sedflux = prop_fluvial;
}

void chonk::add_to_sediment_flux(double value, double prop_fluvial)
{
  this->add_to_sediment_flux(value, this->label_tracker, prop_fluvial );
}

// Add a certain amount to the sediment flux
void chonk::add_to_sediment_flux(double value, std::vector<double> label_proportions, double prop_fluvial)
{
  // if(this->sediment_flux <0)
  //   std::cout << "WARNING:: IN Sediment flux <0 " << this->sediment_flux << std::endl;

  // if I am adding no sediment: do nothing
  if(double_equals(value,0, 1e-8))
  {
    return;
  }


  // If my sediment flux is 0: bunk
  if(double_equals(this->sediment_flux,0))
  {
    this->sediment_flux += value;
    this->label_tracker = label_proportions;
    this->fluvialprop_sedflux = prop_fluvial;

    // if(this->sediment_flux <0)
    // {
    //   std::cout << this->fluvialprop_sedflux  << "|" << this->sediment_flux << "|" << prop_fluvial << "|value:" << value << std::endl;

    //   std::cout << "WARNING:: OUT(nobeef) Sediment flux <0 " << this->sediment_flux << std::endl;
    // }

    return;
  }


  // std::vector<double> newlabprop = mix_two_proportions(this->sediment_flux, this->label_tracker, value, label_proportions);
  this->label_tracker = mix_two_proportions(this->sediment_flux, this->label_tracker, value, label_proportions);;
  
  double fluvialsedfluxtot = value * prop_fluvial + this->fluvialprop_sedflux * this->sediment_flux;
  this->sediment_flux += value;
  if(double_equals(this->sediment_flux,0,1e-6))
  {
    this->sediment_flux = 0;
    return;
  }

  this->fluvialprop_sedflux = fluvialsedfluxtot/ this->sediment_flux;

  if(double_equals(this->fluvialprop_sedflux - 1, 0, 1e-8))
    this->fluvialprop_sedflux = 1;
  
  if(double_equals(this->fluvialprop_sedflux, 0, 1e-8))
    this->fluvialprop_sedflux = 0;


  if(this->fluvialprop_sedflux > 1)
  {
    std::cout << this->fluvialprop_sedflux << "|" << fluvialsedfluxtot << "|" << this->sediment_flux << "|" << prop_fluvial << "|value:" << value << std::endl;
    throw std::runtime_error("fluvprop>1");
  }
  if(this->sediment_flux <0)
  {
    std::cout << this->fluvialprop_sedflux << "|" << fluvialsedfluxtot << "|" << this->sediment_flux << "|" << prop_fluvial << "|value:" << value << std::endl;
    std::cout << "WARNING:: OUT Sediment flux <0 " << this->sediment_flux << std::endl;
  }

}



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

  throw std::runtime_error("The simplest stream power law is currently broken. Please use Charlie_I with 0 deposition instead.");

  // // I am recording the current sediment fluxes in the model distributed for each receivers
  // std::vector<double> pre_sedfluxes = get_preexisting_sediment_flux_by_receivers();
  // // Calculation current fluxes
  // for(size_t i=0; i<this->receivers.size(); i++)
  // {
  //   // calculating the flux E = K s^n A^m
  //   double this_eflux = std::pow(this->water_flux * this->weigth_water_fluxes[i],m) * std::pow(this->slope_to_rec[i],n) * K;
  
  //   // stacking the erosion flux
  //   this->erosion_flux_undifferentiated += this_eflux;

  //   // What has been eroded moves into the sediment flux (which needs to be converted into a volume)
  //   std::vector<double> buluf(this->label_tracker.size(), 0.);
  //   buluf[label] = 1.;
  //   this->add_to_sediment_flux(this_eflux * Xres * Yres * dt, buluf);
  //   // recording the current flux 
  //   pre_sedfluxes[i] += this_eflux * Xres * Yres * dt;

  // }


  // // Now I need to recalculate the sediment fluxes weights to each receivers
  // for(size_t i=0; i<this->receivers.size(); i++)
  // {
  //   if(this->sediment_flux>0)
  //     this->weigth_sediment_fluxes[i] = pre_sedfluxes[i]/this->sediment_flux;
  // }

  // Done
  return;
}

void chonk::charlie_I_K_fQs(double n, double m, double K_r, double K_s,
  double dimless_roughness, double this_sed_height, double V_param, 
  double d_star, double threshold_incision, double threshold_sed_entrainment,
  int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres,
  std::vector<double> Krmodifyer)
{
  // Very trivial modifyer to modulate Kr and simulate a tool effect
  double mod = 0;
  for(size_t i=0; i<sed_label_prop.size(); i++)
  {
    mod += this->label_tracker[i] * Krmodifyer[i];
  }

  if(mod <= 0)
    mod = Krmodifyer[zone_label];
  K_r = mod * K_r;


  // Calling good old CHARLIE_I
  this->charlie_I( n,  m,  K_r,  K_s, dimless_roughness,  this_sed_height,  V_param,  d_star,  threshold_incision,  
    threshold_sed_entrainment, zone_label,  sed_label_prop,  dt,  Xres,  Yres);
}

void chonk::charlie_I(double n, double m, double K_r, double K_s,
  double dimless_roughness, double this_sed_height, double V_param, 
  double d_star, double threshold_incision, double threshold_sed_entrainment,
  int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres)
{

  // First some debugging checks about incoming fluxes and receivers states. 
  // May diseappear once stabilised
  // #1
  if(this->fluvialprop_sedflux > 1)
  {
    if(this->fluvialprop_sedflux > 1.01)
    {
      std::cout << "WARNING!!!!! " << fluvialprop_sedflux << std::endl;;
      throw std::runtime_error("Charlie_I started with fluvialprop_sedflux > 1");
    }
    this->fluvialprop_sedflux = 1;
  }
  // #2
  if(this->water_flux <= 0)
  {
    std::cout << "Charlie_I saw Qw <0 and fixed it but you need to take care of that" << std::endl;
    this->water_flux = 0;
      return;
  } 

  // #3
  // ## If no rec, no point running this function
  if(this->receivers.size() == 0)
    return;

  // IMPORTANT in case another process had affected the sed-height bofre, I am applying it
  this_sed_height += this->sediment_creation_flux *dt;

  // First dealing with the erosion

  // # Initialising local fluxes to 0
  double Er_tot = 0;
  double Es_tot = 0;
  double Ds_tot = 0;
  double E_cap_s = 0;

  // Calculating the exponential component first
  double exp_sed_height_roughness =  std::exp(- this_sed_height / dimless_roughness);

  // Calculation current erosion fluxes for each receivers
  for(size_t i=0; i<this->receivers.size(); i++)
  {

    // Local water flux
    double this_Qw = this->water_flux * this->weigth_water_fluxes[i];

    // Local Stream Power
    double current_stream_power = std::pow(this_Qw,m) * std::pow(this->slope_to_rec[i],n);

    // Calculating the flux E = K s^n A^m - threshold

    // # Thresholds:
    // ## Bedrock
    double threshholder_bedrock = 0.;
    if(threshold_incision > 0)
      threshholder_bedrock = current_stream_power * (1 - std::exp(- current_stream_power/threshold_incision) );
    
    // ## Sediments
    double threshholder_sed = 0.;
    if(threshold_sed_entrainment > 0)
      threshholder_sed = current_stream_power * (1 - std::exp(- current_stream_power/threshold_sed_entrainment) );


    // Calculating erosion
    // # Bedrock erosion
    double Er = (current_stream_power * K_r 
        - threshholder_bedrock )
        * exp_sed_height_roughness;

    // # If neg erosion iz 0
    if(Er<0)
      Er = 0;

    // # Sediment erosion (entrainment)
    double Es = (current_stream_power * K_s
        - threshholder_sed)
        * (1 - exp_sed_height_roughness);

    // # If neg erosion iz 0
    if(Es<0)
      Es = 0;

    // Ecap will be needed for later 
    double datcaps = current_stream_power * K_s - threshholder_sed;
    // I keep it if negative (see H(t) calculation)
    E_cap_s += datcaps;

    // Adding to global values
    Er_tot += Er;
    Es_tot += Es;
  }

  // Adding the eroded bedrock to the sediment flux
  
  //# label proportion for current bedrock
  std::vector<double> buluf(this->label_tracker.size(), 0.);
  buluf[zone_label] = 1.;

  //# Depodivider is the analytical solution from sediment divergence (See equation 31 -> SPACE paper)
  double depodivider = (1 + V_param * d_star * Xres * Yres / (this->water_flux ) );

  //# temp variables
  double tot_add = 0;
  //# Bedrock erosion 
  double tadd = Er_tot * Xres * Yres * dt;
  tot_add += tadd;
  this->add_to_sediment_flux(tadd, buluf, 1.);

  // Adding the sediment entrained into the sedimetn flux
  tadd = Es_tot * Xres * Yres * dt;  
  tot_add += tadd;
  this->add_to_sediment_flux(tadd, sed_label_prop, 1.);

  // # Getting the sediment flux available for fluvial deposition
  double Qs = this->sediment_flux * this->fluvialprop_sedflux;  
  // COrrecting analytically THE DEPOSITION (see SPACE gmd paper equation 31)
  tadd = Qs/depodivider - Qs;
  this->add_to_sediment_flux(tadd, this->label_tracker, 1.);
  Qs += tadd;

  // # Deposition
  Ds_tot += V_param * d_star * (Qs/ (this->water_flux * dt));
  double removersed = -1 * Ds_tot * dt * Xres * Yres;

  // Applying to the global fluxes
  this->erosion_flux_only_bedrock += Er_tot;
  this->erosion_flux_only_sediments += Es_tot;
  this->deposition_flux += Ds_tot;

  // Now calculating the new sediment heugh analytically
  double phi = 0; // TEMPORARY MEASURE, PHI WILL BE TO BE ADDED AFTER
  double new_sed_height = 0.;
  double Dsphi = Ds_tot / (1 - phi);

  // IMPORTANT to avoid numerical overflow, recasting H to 400 max for the exponential calculation
  double H_eq = this_sed_height;
  if(H_eq > 100)
    H_eq = 100;

  double tempsedheight = H_eq;

  // Case 1 (32)
  if(Dsphi != E_cap_s && E_cap_s > 0)
  {
    new_sed_height = dimless_roughness * std::log( 1/ ((Dsphi / E_cap_s) - 1) *  \
      ( std::exp( (Dsphi - E_cap_s) * dt/dimless_roughness ) * ( ( Dsphi/E_cap_s - 1 ) \
       * std::exp(H_eq/dimless_roughness) + 1 ) - 1 )  );
  }
  else if (Dsphi == E_cap_s)
  {
    new_sed_height = dimless_roughness * std::log(E_cap_s/dimless_roughness * dt + std::exp(H_eq/dimless_roughness));
  }
  else
  {
    new_sed_height = this_sed_height + Dsphi * dt;
  }

  double new_sedcrea = (new_sed_height - tempsedheight) / dt;

  // if(new_sedcrea > 0.01)
  // {
  //   std::cout << "DEBUG::Sedcrea in Charlie_I very high:";
  //   std::cout << new_sed_height << "|" << this_sed_height << "|" << Ds_tot << "|" << Es_tot << "|" << Er_tot << "|" \
  //    << this->fluvialprop_sedflux << "|" << Qs << "|" << this->water_flux << "|"  << std::endl;
  // }

  this->add_sediment_creation_flux(new_sedcrea);

  // Finally need to deal with flux partition
  // let's assume here that the fluxes are partitioned by water
  auto HS_fluxes = this->get_preexisting_sediment_flux_by_receivers_hillslopes();
  bool HS_already_there = true;

  if(sum_vector(HS_fluxes) == 0) {HS_already_there = false;};


  for(size_t i=0; i<this->receivers.size(); i++)
  {
    if(HS_already_there)
      this->weigth_sediment_fluxes[i] = (this->weigth_water_fluxes[i] * Qs + HS_fluxes[i])/this->sediment_flux;
    else
      this->weigth_sediment_fluxes[i] = this->weigth_water_fluxes[i];
  }   



  return;
}
// Before test
// void chonk::charlie_I(double n, double m, double K_r, double K_s,
//   double dimless_roughness, double this_sed_height, double V_param, 
//   double d_star, double threshold_incision, double threshold_sed_entrainment,
//   int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres)
// {
//    // I am recording the current sediment fluxes in the model distributed for each receivers
//   std::vector<double> pre_sedfluxes = std::vector<double>(this->receivers.size(), 0. );
//   // Initialising the fluxes bwith the water ones
//   // std::vector<double> charlie_I_weights4sed = this->weigth_water_fluxes;
//   double total_fluvial_sedflux = this->sediment_flux * this->fluvialprop_sedflux;
//   double save_entry = total_fluvial_sedflux;


//   // I first assume my sediment fluxes are following my water 
//   for(size_t i=0; i < pre_sedfluxes.size(); i++)
//     pre_sedfluxes[i] = this->weigth_water_fluxes[i] * total_fluvial_sedflux;


//   if(this->fluvialprop_sedflux > 1)
//   {
//     if(this->fluvialprop_sedflux > 1.01)
//     {
//       std::cout << "WARNING!!!!! " << fluvialprop_sedflux << std::endl;;
//       throw std::runtime_error("Charlie_I started with fluvialprop_sedflux > 1");
//     }
    
//     this->fluvialprop_sedflux = 1;
//   }

//   // IMPORTANT in case another process had affected the sed-height bofre, I am applying it
//   this_sed_height += this->sediment_creation_flux *dt;


//   double Er_tot = 0;
//   double Es_tot = 0;
//   double Ds_tot = 0;

//   double E_cap_s = 0;

//   if(this->water_flux <= 0)
//   {
//     std::cout << "Charlie_I saw Qw <0 and fixed it but you need to take care of that" << std::endl;
//     this->water_flux = 0;
//       return;
//   }

//   if(this->receivers.size() == 0)
//     return;

//   double depodivider = 1 + (V_param * d_star * Xres * Yres / this->water_flux);
//   // double

//   // for(auto& flub:pre_sedfluxes)
//   //   flub = flub/depodivider;

//   double exp_sed_height_roughness =  std::exp(- this_sed_height / dimless_roughness);
//   // Calculation current fluxes
//   for(size_t i=0; i<this->receivers.size(); i++)
//   {

//     double this_Qw = this->water_flux * this->weigth_water_fluxes[i];

//     double current_stream_power = std::pow(this_Qw,m) * std::pow(this->slope_to_rec[i],n);
//     // std::cout << "SLOPE IS " << this->slope_to_rec[i] << " CURRENT OMEGA = " << current_stream_power;
//     // calculating the flux E = K s^n A^m
//     double threshholder_bedrock = 0.;
//     if(threshold_incision > 0)
//       threshholder_bedrock = current_stream_power * (1 - std::exp(- current_stream_power/threshold_incision) );

//     double threshholder_sed = 0.;
//     if(threshold_sed_entrainment > 0)
//       threshholder_sed = current_stream_power * (1 - std::exp(- current_stream_power/threshold_sed_entrainment) );

//     double Er = (current_stream_power * K_r 
//         - threshholder_bedrock )
//         * exp_sed_height_roughness;

//     if(Er<0)
//       Er = 0;

//     double Es = (current_stream_power * K_s
//         - threshholder_sed)
//         * (1 - exp_sed_height_roughness);

//     if(Es<0)
//       Es = 0;


//     E_cap_s += (current_stream_power * K_s - threshholder_sed);
    
//     Er_tot += Er;
//     Es_tot += Es;
 
//     // // recording the current flux 
//     pre_sedfluxes[i] += (Er + Es) * Xres * Yres * dt;

//   }

//   // Adding the eroded bedrock to the sediment flux
//   std::vector<double> buluf(this->label_tracker.size(), 0.);


//   buluf[zone_label] = 1.;
//   double tadd = Er_tot * Xres * Yres * dt;
//   this->add_to_sediment_flux(tadd, buluf, 1.);
//   total_fluvial_sedflux += tadd;

//   tadd = Es_tot * Xres * Yres * dt;
  
//   // Adding the sediment entrained into the sedimetn flux
//   this->add_to_sediment_flux(tadd, sed_label_prop, 1.);
//   total_fluvial_sedflux += tadd;

//   // COrrecting analytically (see SPACE gmd paper equation 31)
//   // tadd = total_fluvial_sedflux -  total_fluvial_sedflux/depodivider;
//   // this->add_to_sediment_flux( -tadd, this->label_tracker, 1.);

//   total_fluvial_sedflux = total_fluvial_sedflux/depodivider;
  

//   Ds_tot += V_param * d_star * (total_fluvial_sedflux/ (this->water_flux * dt));

//   double removersed = -1 * Ds_tot * dt * Xres * Yres;
//   double ratio_removersed = abs(removersed) / sum_vector(pre_sedfluxes);
//   double delta_truc_1 = abs(removersed) - total_fluvial_sedflux;
//   if(delta_truc_1 > 0 )
//   {
//     if(delta_truc_1 > 1e2)
//       std::cout << "DEBUG::REcalibrating sediment fluxes::" << removersed << " vs " << total_fluvial_sedflux << std::endl;
//     removersed = -total_fluvial_sedflux;
//     Ds_tot = - removersed / (dt * Xres * Yres);
//     pre_sedfluxes = std::vector<double>(pre_sedfluxes.size(),0);
//   }

//   // std::cout << ratio_removersed << "||" << sum_vector(pre_sedfluxes)/depodivider << "||" << total_fluvial_sedflux << std::endl;
//   this->add_to_sediment_flux(removersed, this->label_tracker, 1.);



//   // if(this->sediment_flux<0)
//   //   std::


//   // std::cout << "Gub:" << total_fluvial_sedflux - this->sediment_flux * this->fluvialprop_sedflux << std::endl;

//   double sumweights = 0;
//   std::vector<double> HS_fluxes = this->get_preexisting_sediment_flux_by_receivers_hillslopes();
//   double tsum = sum_vector(HS_fluxes);
//   bool has_been_calc = true;
//   if(tsum == 0)
//     has_been_calc = false;

//   if(this->receivers.size()>0)
//   {
//     for(size_t i=0; i<this->receivers.size(); i++)
//     {
//       if(this->sediment_flux>0)
//       {
//         if(has_been_calc)
//           this->weigth_sediment_fluxes[i] = (pre_sedfluxes[i]/depodivider + HS_fluxes[i])/this->sediment_flux;
//         else
//           this->weigth_sediment_fluxes[i] = (pre_sedfluxes[i]/depodivider)/(this->sediment_flux * this->fluvialprop_sedflux);
//       }

//       sumweights += this->weigth_sediment_fluxes[i];
//       if(this->weigth_sediment_fluxes[i] < 0)
//       {
//         std::cout << has_been_calc << "|" << ratio_removersed << "|" << HS_fluxes[i] << "|" << sediment_flux << std::endl;
//         throw std::runtime_error("fluxweight<0 sed charlie_I :: " + std::to_string(this->weigth_sediment_fluxes[i]) + " -> " + std::to_string(pre_sedfluxes[i]) );
//       }
    
//     }
//   }
//   // std::cout << this->fluvialprop_sedflux << "|";
//   if(double_equals(sumweights,0,1e-7) ==  true)
//     this->weigth_sediment_fluxes = std::vector<double>(this->weigth_water_fluxes);

//   sumweights = 0;
//   for (auto s:this->weigth_sediment_fluxes)
//     sumweights += s;

//   if(double_equals(sumweights,1,1e-6) ==  false)
//   {
//     // throw std::runtime_error("Sedweightserrors::" + std::to_string(sumweights));
//     std::cout << "Sedweightapproxwarning::" + std::to_string(sumweights) << std::endl;
//     for (auto& s:this->weigth_sediment_fluxes)
//       s/sumweights;
//   }


//   // removing the deposition from sediment flux

//   // Applying to the global fluxes
//   this->erosion_flux_only_bedrock += Er_tot;
//   this->erosion_flux_only_sediments += Es_tot;
//   this->deposition_flux += Ds_tot;

//   double phi = 0; // TEMPORARY MEASURE, PHI WILL BE TO BE ADDED AFTER
  
//   double new_sed_height = 0.;
//   double Dsphi = Ds_tot / (1 - phi);

//   double H_eq = this_sed_height;
//   if(H_eq > 400)
//     H_eq = 400;

//   if(E_cap_s <= 0 || Es_tot * dt < H_eq)
//   {
//     new_sed_height = H_eq + Dsphi * dt;
//     // std::cout << "No ER but In this case Er is " << Er_tot << std::endl;

//   }
//   else if( (Ds_tot/(1-phi))/E_cap_s == 1 )
//   {
//     new_sed_height = dimless_roughness * std::log( E_cap_s/ dimless_roughness * dt + std::exp(H_eq/dimless_roughness) );
//   }
//   else
//   {
//     double A = 1 / ((Dsphi / E_cap_s) - 1);    
//     double B = Dsphi - E_cap_s;
//     B = B * dt / dimless_roughness;
//     B = std::exp(B);
//     double C = ( ( Dsphi / E_cap_s ) - 1 ) * std::exp(H_eq/dimless_roughness);
//     new_sed_height = dimless_roughness * std::log( A * (B * (C + 1) - 1));
//     if(std::isfinite(new_sed_height) == false)
//       std::cout << dimless_roughness << "|*|" << A << "|*|" << B << "|*|"  << C << "|*|"  << Dsphi  << "|*|" << E_cap_s <<  "|*|"  << H_eq << "|*|" << std::exp(H_eq/dimless_roughness) << std::endl;
//   }

//   double new_sedcrea = (new_sed_height - this_sed_height) / dt;

//   if(new_sedcrea > 0.01)
//   {
//     std::cout << "DEBUG::Sedcrea in Charlie_I very high:";
//     std::cout << new_sed_height << "|" << this_sed_height << "|" << Ds_tot << "|" << Es_tot << "|" << Er_tot << "|" \
//      << this->fluvialprop_sedflux << "|" << total_fluvial_sedflux << "|" << this->water_flux << "|" << save_entry  << std::endl;
//   }



//   this->add_sediment_creation_flux(new_sedcrea);



//   if(std::isfinite(this->sediment_creation_flux) == false)
//   {
//     std::cout << new_sed_height << "||" << this_sed_height << "||" << Es_tot << "||" << Ds_tot << "||" << this->sediment_flux << "||" << this->water_flux<< "||" << this->chonkID <<  std::endl;
//     throw std::runtime_error("Sedcrea getting nan value in CHARLIE_I");
//   }

//   // Done
//   return;
// }

// OLD CHARLIE I WITH DEPO DIVIDER TO KEEP
// void chonk::charlie_I(double n, double m, double K_r, double K_s,
//   double dimless_roughness, double this_sed_height, double V_param, 
//   double d_star, double threshold_incision, double threshold_sed_entrainment,
//   int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres)
// {
//    // I am recording the current sediment fluxes in the model distributed for each receivers
//   std::vector<double> pre_sedfluxes = std::vector<double>(this->receivers.size(), 0. );
//   // Initialising the fluxes bwith the water ones
//   // std::vector<double> charlie_I_weights4sed = this->weigth_water_fluxes;
//   double total_fluvial_sedflux = this->sediment_flux * this->fluvialprop_sedflux;

//   // I first assume my sediment fluxes are following my water 
//   for(size_t i=0; i < pre_sedfluxes.size(); i++)
//     pre_sedfluxes[i] = this->weigth_water_fluxes[i] * total_fluvial_sedflux;


//   if(this->fluvialprop_sedflux > 1.01)
//   {
//     std::cout << "WARNING!!!!! " << fluvialprop_sedflux << std::endl;;
//     throw std::runtime_error("Charlie_I started with fluvialprop_sedflux > 1");
//     this->fluvialprop_sedflux = 1;
//   }

//   // IMPORTANT in case another process had affected the sed-height bofre, I am applying it
//   this_sed_height += this->sediment_creation_flux *dt;


//   double Er_tot = 0;
//   double Es_tot = 0;
//   double Ds_tot = 0;

//   double E_cap_s = 0;

//   if(this->water_flux <= 0)
//   {
//     std::cout << "Charlie_I saw Qw <0 and fixed it but you need to take care of that" << std::endl;
//     this->water_flux = 0;
//       return;
//   }

//   if(this->receivers.size() == 0)
//     return;

//   double depodivider = 1 + (V_param * d_star * Xres * Yres / this->water_flux);
//   // double

//   for(auto& flub:pre_sedfluxes)
//     flub = flub/depodivider;

//   double exp_sed_height_roughness =  std::exp(- this_sed_height / dimless_roughness);
//   // Calculation current fluxes
//   for(size_t i=0; i<this->receivers.size(); i++)
//   {

//     double this_Qw = this->water_flux * this->weigth_water_fluxes[i];

//     double current_stream_power = std::pow(this_Qw,m) * std::pow(this->slope_to_rec[i],n);
//     // std::cout << "SLOPE IS " << this->slope_to_rec[i] << " CURRENT OMEGA = " << current_stream_power;
//     // calculating the flux E = K s^n A^m
//     double threshholder_bedrock = 0.;
//     if(threshold_incision > 0)
//       threshholder_bedrock = current_stream_power * (1 - std::exp(- current_stream_power/threshold_incision) );

//     double threshholder_sed = 0.;
//     if(threshold_sed_entrainment > 0)
//       threshholder_sed = current_stream_power * (1 - std::exp(- current_stream_power/threshold_sed_entrainment) );

//     double Er = (current_stream_power * K_r 
//         - threshholder_bedrock )
//         * exp_sed_height_roughness;

//     double Es = (current_stream_power * K_s
//         - threshholder_sed)
//         * (1 - exp_sed_height_roughness);


//     E_cap_s += (current_stream_power * K_s - threshholder_sed);
    
//     Er_tot += Er;
//     Es_tot += Es;
 
//     // // recording the current flux 
//     pre_sedfluxes[i] += (Er + Es) * Xres * Yres * dt / depodivider;

//   }

//   // Adding the eroded bedrock to the sediment flux
//   std::vector<double> buluf(this->label_tracker.size(), 0.);


//   buluf[zone_label] = 1.;
//   double tadd = Er_tot * Xres * Yres * dt;
//   this->add_to_sediment_flux(tadd, buluf, 1.);
//   total_fluvial_sedflux += tadd;

//   tadd = Es_tot * Xres * Yres * dt;
  
//   // Adding the sediment entrained into the sedimetn flux
//   this->add_to_sediment_flux(tadd, sed_label_prop, 1.);
//   total_fluvial_sedflux += tadd;

//   // COrrecting analytically (see SPACE gmd paper equation 31)
//   tadd = total_fluvial_sedflux -  total_fluvial_sedflux/depodivider;
//   this->add_to_sediment_flux( -tadd, this->label_tracker, 1.);
//   total_fluvial_sedflux = total_fluvial_sedflux/depodivider;

//   double sumweights = 0;
//   std::vector<double> HS_fluxes = this->get_preexisting_sediment_flux_by_receivers_hillslopes();
//   if(this->receivers.size()>0)
//   {
//     for(size_t i=0; i<this->receivers.size(); i++)
//     {
//       if(this->sediment_flux>0)
//         this->weigth_sediment_fluxes[i] = (pre_sedfluxes[i] + HS_fluxes[i])/this->sediment_flux;

//       sumweights += this->weigth_sediment_fluxes[i];
    
//     }
//   }
//   // std::cout << this->fluvialprop_sedflux << "|";
//   if(double_equals(sumweights,0,1e-7) ==  true)
//     this->weigth_sediment_fluxes = std::vector<double>(this->weigth_water_fluxes);

//   sumweights = 0;
//   for (auto s:this->weigth_sediment_fluxes)
//     sumweights += s;

//   if(double_equals(sumweights,1,1e-3) ==  false)
//   {
//     throw std::runtime_error("Sedweightserrors::" + std::to_string(sumweights));
//     for (auto& s:this->weigth_sediment_fluxes)
//       s/sumweights;
//   }
  

//   Ds_tot += V_param * d_star * (total_fluvial_sedflux/ (this->water_flux * dt));
//   // this->add_to_sediment_flux(-1 * Ds_tot * dt * Xres * Yres, this->label_tracker, 1.);

//   // removing the deposition from sediment flux

//   // Applying to the global fluxes
//   this->erosion_flux_only_bedrock += Er_tot;
//   this->erosion_flux_only_sediments += Es_tot;
//   this->deposition_flux += Ds_tot;

//   double phi = 0; // TEMPORARY MEASURE, PHI WILL BE TO BE ADDED AFTER
  
//   double new_sed_height = 0.;
//   double Dsphi = Ds_tot / (1 - phi);

//   if(E_cap_s <= 0 || Es_tot * dt < this_sed_height)
//   {
//     new_sed_height = this_sed_height + Dsphi * dt;
//     // std::cout << "No ER but In this case Er is " << Er_tot << std::endl;

//   }
//   else if( (Ds_tot/(1-phi))/E_cap_s == 1 )
//   {
//     new_sed_height = dimless_roughness * std::log( E_cap_s/ dimless_roughness * dt + std::exp(this_sed_height/dimless_roughness) );
//   }
//   else
//   {
//     double A = 1 / ((Dsphi / E_cap_s) - 1);    
//     double B = Dsphi - E_cap_s;
//     B = B * dt / dimless_roughness;
//     B = std::exp(B);
//     double C = ( ( Dsphi / E_cap_s ) - 1 ) * std::exp(this_sed_height/dimless_roughness);
//     new_sed_height = dimless_roughness * std::log( A * (B * (C + 1) - 1));
//     if(std::isfinite(new_sed_height) == false)
//       std::cout << dimless_roughness << "|*|" << A << "|*|" << B << "|*|"  << C << "|*|"  << Dsphi  << "|*|" << E_cap_s <<  "|*|"  << this_sed_height << "|*|" << std::exp(this_sed_height/dimless_roughness) << std::endl;
//   }

//   double new_sedcrea = (new_sed_height - this_sed_height) / dt;

//   if(new_sedcrea > 0.1)
//     std::cout << new_sed_height << "|" << this_sed_height << "|" << Ds_tot << "|" << Es_tot << "|" << Er_tot << "|" << this->fluvialprop_sedflux << "|" << total_fluvial_sedflux << "|" << this->water_flux  << std::endl;

//   this->add_sediment_creation_flux(new_sedcrea);



//   if(std::isfinite(this->sediment_creation_flux) == false)
//   {
//     std::cout << new_sed_height << "||" << this_sed_height << "||" << Es_tot << "||" << Ds_tot << "||" << this->sediment_flux << "||" << this->water_flux<< "||" << this->chonkID <<  std::endl;
//     throw std::runtime_error("Sedcrea getting nan value in CHARLIE_I");
//   }

//   // Done
//   return;
// }

void chonk::CidreHillslopes(double this_sed_height, double kappa_s, double kappa_r, double Sc,
  int zone_label, std::vector<double> sed_label_prop, double dt, double Xres, double Yres, bool bedrock, 
  NodeGraphV2& graph, double tolerance_to_Sc)
{
  if(this->receivers.size() == 0)
    return;
  // std::cout << "Starting here" << std::endl;
   // I am recording the current sediment fluxes in the model distributed for each receivers
  // std::vector<double> pre_sedfluxes = this->get_preexisting_sediment_flux_by_receivers_hillslopes();
  std::vector<double> pre_sedfluxes = std::vector<double>(this->receivers.size(), 0);
  std::vector<double> presedfluv = this->get_preexisting_sediment_flux_by_receivers_fluvial();

  double sum_of_presedf = 0;
  for(auto v:presedfluv)
    sum_of_presedf += v;
  // if(double_equals(sum_of_presedf,this->sediment_flux * this->fluvialprop_sedflux,1e-3)==false)
  //   throw std::runtime_error("wrong start");
  
  double sed_HS_in = this->sediment_flux * (1 - this->fluvialprop_sedflux);

  double sum_added_HS = 0;
  double beef_sedfluv = this->sediment_flux * this->fluvialprop_sedflux;

  // getting the steepest slope and dx
  std::vector<double> dXs =  graph.get_distance_to_receivers_custom(this->chonkID, this->receivers);

  // First calculating the steepest slope driving the gravitational forces 
  double SS = -9999;
  double SS_dx = 0;
  int index_SS = 0;
  double sumslopes = 0;
  int inc=0;
  for(auto slope:this->slope_to_rec)
  {
    if(slope > SS)
    {
      SS = slope;
      SS_dx = dXs[inc];
      index_SS = inc;
    }
    sumslopes += slope;
    inc++; 
  }
  if(SS == -9999)
    throw std::runtime_error("slope exception not handled in cidre hillslope routines");

  // Adapting the current sediment height
  double save_sed_height = this_sed_height;
  double new_sed_height = 0;
  this_sed_height = this_sed_height + this->sediment_creation_flux * dt;

  // Pre_calculations

  // starting with L
  // # Checking the critical slope
  double local_L = 0;
  bool L_is_inf = false;
  double this_nl = 0;
  if(SS >= Sc - tolerance_to_Sc)
  {
    SS = Sc - tolerance_to_Sc; // huge number on purpose
    L_is_inf = true;
  }

  this_nl = SS/Sc;

  if(L_is_inf)
    local_L = 1e36;
  else
    local_L = SS_dx / (1 - std::pow(this_nl,2));

  if(std::isfinite(local_L) == false)
    throw std::runtime_error("Local_L is not finite in cidre hillslope routines");


  // Total E assuming there is enough sediments to be diffused
  double local_es = 0;
  local_es = kappa_s * SS;

  // std::cout << "2:" << local_es << std::endl;

  // Now checking if I have enough sediment to diffuse (+ the minimum sediment I am adding from the deposition)
  // Careful with my weird choice of having Qs as a total volume of sed through time which is why the equations are a bit weird

  double fraction_bedrock_exposed = 0;

  // Calculating the new sediment height: taking into account deposition and erosion, as deposition depends on what can be eroded

  bool iz_alldone = false;
  if(this_sed_height - local_es * dt < 0 && local_es > 0)
  {
    // std::cout << "2:" << local_es << "|" << this_sed_height << std::endl;

    fraction_bedrock_exposed = 1 - abs(this_sed_height)/ (local_es * dt);
    new_sed_height = 0;
    iz_alldone = true;
    local_es = (1 - fraction_bedrock_exposed) * local_es;
    // std::cout << "3:" << fraction_bedrock_exposed << std::endl;

  }
  else if(local_es == 0 || this_sed_height == 0)
  {
    fraction_bedrock_exposed = 1;
  }
// 
  // std::cout << this->erosion_flux_only_sediments << " vs " << local_es << std::endl;

  this->erosion_flux_only_sediments += local_es;
  this->add_to_sediment_flux(local_es* dt * Xres * Yres, sed_label_prop, 0.);
  sum_added_HS += local_es* dt * Xres * Yres;
  new_sed_height = this_sed_height - local_es * dt;

  // Apparentyl does not happen
  if(iz_alldone && double_equals(new_sed_height,0,1e-6) == false)
    throw std::runtime_error(std::to_string(new_sed_height) + " should be 0");


  // std::cout << "3" << std::endl;
  if(fraction_bedrock_exposed >1)
    throw std::runtime_error("fraction_bedrock_exposed>1");

  // routines for bedrock if activated
  if(bedrock && fraction_bedrock_exposed > 0)
  {
    double local_er = kappa_r * SS * fraction_bedrock_exposed;
    std::vector<double> tlabprop = std::vector<double>(this->label_tracker.size(),0);
    tlabprop[zone_label] = 1;
    this->add_to_sediment_flux(local_er * dt * Xres * Yres,tlabprop, 0.);
    sum_added_HS += local_er* dt * Xres * Yres;
    // std::cout << this->erosion_flux_only_bedrock << " vs " << local_er << std::endl;
    this->erosion_flux_only_bedrock += local_er;
  }
  else
  {
    // std::cout << this->erosion_flux_only_bedrock << " vs " << 0 << std::endl;
  }

  // Cannot create sediment, so L has to be > 1 nor delete them to the oblivion
  if(local_L < 1)
    local_L = 1;


  // Calculating the deposition rate
  double this_dep = ((sed_HS_in/dt )/ local_L);
  double checker_sedneg = (this->sediment_flux * (1 - this->fluvialprop_sedflux)) - this_dep * dt * Xres * Yres;
  if(checker_sedneg < 0)
  {
    this_dep = (1 - abs(checker_sedneg)/(this_dep * dt * Xres * Yres)) * this_dep;
    // checker_sedneg = this->sediment_flux - this_dep * dt * Xres * Yres;
  }
  
  // std::cout << this->deposition_flux << " vs " << this_dep << std::endl;

  // And adding it to the thingy
  this->deposition_flux += this_dep;


  // correcting the new sediment height
  new_sed_height += this_dep * dt;
  // And calculating the sediment creation one
  this->sediment_creation_flux += (new_sed_height - this_sed_height)/dt;

  if(this->sediment_creation_flux > 0.01)
    std::cout << "SedHighCidre!!::" << this->sediment_creation_flux << " vs " << (new_sed_height - this_sed_height)/dt << std::endl;

  if(std::isfinite(this->sediment_creation_flux) == false)
  {
    std::cout << new_sed_height << "||" << this_sed_height << std::endl;
    throw std::runtime_error("NAN SEDCREA CIDRE");
  }
  // Removing the deposited sediments from the global fluxes
  // std::cout << "bulf:" << - this_dep * dt * Xres * Yres << "|" << this->sediment_flux << std::endl;
  this->add_to_sediment_flux(- this_dep * dt * Xres * Yres, this->label_tracker,0);
    sum_added_HS -= this_dep* dt * Xres * Yres;
  if(this->sediment_flux < 0 )
  {
    std::cout << "SEDCIDRE<0::" << this->sediment_flux << std::endl;
    this->sediment_flux = 0;
  }
  // std::cout << "4" << std::endl;


    // std::cout << "HighDepWarning::" << this_dep << "||" << local_L << "||" << local_es << "||" << this_nl << "||" << SS_dx << std::endl;


  // Backcalculating the new weight of sediment
  double sumsum = 0;
  double delta_sed = ( this->sediment_flux * (1 - this->fluvialprop_sedflux) );
  // if(delta_sed == 0)
  // std::cout << delta_sed << "||" << sum_added_HS << std::endl;
  double corrector = 0;
  double sumweights_1 = 0;
  if(sumslopes > 0)
  {
    for(size_t i = 0; i < this->receivers.size(); i++ )
    {
      pre_sedfluxes[i] += (this->slope_to_rec[i]/ sumslopes) * delta_sed;
      sumweights_1 += this->slope_to_rec[i]/ sumslopes;


      if(pre_sedfluxes[i] < 0)
      {
        std::cout << "Caluf::Correcting thingied " << abs(pre_sedfluxes[i]) << std::endl;
        corrector += abs(pre_sedfluxes[i]);
        pre_sedfluxes[i] = 0;
      }

      sumsum += pre_sedfluxes[i];
      sumweights_1 += presedfluv[i];
    }
  }

  if(corrector > 0)
  {
    pre_sedfluxes[index_SS] += corrector;
    sumsum += corrector;
  }

  // ALL HERE IS VERIFIED, WTF IS GOING ON ?!
  // if(sumsum != delta_sed || delta_sed + (this->fluvialprop_sedflux * this->sediment_flux) != this->sediment_flux)
  // std::cout << sumweights_1 - (this->fluvialprop_sedflux * this->sediment_flux) << " vs " << delta_sed + (this->fluvialprop_sedflux * this->sediment_flux) << " vs " << this->sediment_flux << " vs " << sumsum << " vs " << delta_sed << std::endl;


  if(sumsum == 0)
  {
    if(this->sediment_flux < 0 || std::isfinite(this->sediment_flux) == false)
    {
      std::cout << "CidreSedProblem::" << this->sediment_flux << std::endl;
      this->sediment_flux = 0;
      // throw std::runtime_error("SedNegCidre");
    }
    return;
  }

  if(this->sediment_flux < 0 || std::isfinite(this->sediment_flux) == false || this->sediment_flux > 1e36)
  {
    std::cout << "CidreSedProblem::" << this->sediment_flux << std::endl;
    this->sediment_flux = 0;
    // throw std::runtime_error("SedNegCidre");
  }

  double sumweights = 0;
  double sumstuff = 0;
  for(size_t i = 0; i < this->receivers.size(); i++ )
  {
    this->weigth_sediment_fluxes[i] = (pre_sedfluxes[i] + presedfluv[i])/(this->sediment_flux);

    sumweights += this->weigth_sediment_fluxes[i];
    sumstuff += pre_sedfluxes[i] + presedfluv[i];


    // this->weigth_sediment_fluxes[i] = pre_sedfluxes[i] /sumsum;
    if(std::isfinite(this->weigth_sediment_fluxes[i]) == false)
    {
      std::cout << "cidreWeightProblem::" <<this->weigth_sediment_fluxes[i] <<  "||" << sumsum << std::endl;
      this->weigth_sediment_fluxes = std::vector<double>(this->receivers.size(),0);
      this->weigth_sediment_fluxes[index_SS] = 1;
      break;
    }
  }
  double afft_sedfluv = this->sediment_flux * this->fluvialprop_sedflux;

  if(double_equals(sumstuff,this->sediment_flux,1e-3) == false)
    std::cout << sumstuff << " vs " << this->sediment_flux<< std::endl;

  if(double_equals(sumweights,1,1e-6) == false)
    std::cout << "BAGLABUF::" << sumweights << std::endl;

  // Done
  return;
}

double chonk::sed_flux_given_to_node(int tnode)
{
  double res = 0;
   // J is the indice of this specific node in the tchonk referential
  int j = -1;
  auto itj = std::find(this->receivers.begin(), this->receivers.end(), tnode);
  if(itj != this->receivers.end() )
  {
    j = std::distance(this->receivers.begin(), itj);
    res = this->weigth_sediment_fluxes[j] * this->sediment_flux;
  }

}


// Mixe two proportions
std::vector<double> mix_two_proportions(double prop1, std::vector<double> labprop1, double prop2, std::vector<double> labprop2)
{ 
  if(labprop1.size() == 0)
    return labprop2;
  if(labprop2.size() == 0)
    return labprop1;

  // The ouput will eb the same size as the inputs
  std::vector<double>output(labprop1.size(),0.);
  
  // Saving the global sum
  double sumall = 0;
  // summing all proportions with their respective weigths
  for(size_t i=0; i< labprop1.size(); i++)
  {
    // Absolute value because one of the two proportions might be negative, if I am revmoving element for example.
    output[i] = std::abs( prop1 * labprop1[i] + prop2 * labprop2[i]);

    sumall += output[i];
  }
  // If all is 0, all is 0
  if(double_equals(sumall,0.,1e-6))
    return output;

  // Normalising the new proportions and checking that the sum is 1 (can take few iterations is cases where there are very small proportions in order to ensure numerical stability)
  do
  {
    double new_sumfin = 0;
    for(auto& gag:output)
    {
      gag = gag/sumall;
      new_sumfin += gag;
    }
    sumall = new_sumfin;
  }while(double_equals(sumall,1) == false);

  // Keeping that check for a bit
  for(auto gag:output)
    if(std::isfinite(gag) == false)
      throw std::runtime_error("There are some nan/inf in the mixing proportions");

  return output;
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
