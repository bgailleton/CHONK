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


// This empty constructor is just there to have a default one.
void chonk::create()
{
  is_empty = true;
}


void chonk::merge(std::vector<chonk> other_chonks)
{
  // The merging function will need to be updated thouroughly!
  // TO SORT
  // for(auto& tchonk:other_chonks)
  // {
  //   this->water_flux += tchonk.get_water_flux();
  // }
}










//####################################################
//####################################################
//############ MOVE FUCTIONs ##########################
//####################################################
//####################################################
//


// Simplest function we can think of: move the thingy to 
void chonk::move_to_steepest_descent(xt::pytensor<double,1>& elevation, NodeGraph& graph)
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
      //lksajfljdfajls;dfjlsjdfl;jaslkdfjasldfjl;sjdfl;sjdglhdfgwerojf,xncvoiuroqpiwu4598275934tglvn
      // get_pits_outlet_at_pit_ID
      int flub = 6;
    }
  }
  // There is a neighbor, let's save it
  receivers.push_back(steepest_rec);
  weigth_water_fluxes.push_back(1.);
  slope_to_rec.push_back(steepest_S); 
}










#endif