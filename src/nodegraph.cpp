#ifndef nodegraph_CPP
#define nodegraph_CPP

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

#include "nodegraph.hpp"



// This empty constructor is just there to have a default one.
void NodeGraph::create()
{
  std::string yo = "I am an empty constructor yo!";

}

// This empty constructor is just there to have a default one.
void NodeGraph::create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_rec, xt::pytensor<int,1>& post_stack,
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec,xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
{

  // I am first correcting the donors using the receivers: the receivers seems alright but somehow my donors are buggy
  // This is simply done by inverting the receiver to the donors
  xt::pytensor<int,1> ndon = xt::zeros<int>({post_stack.size()});
  for (size_t i =0; i< post_stack.size(); i++)
  {
    for(size_t j=0; j<8;j++)
    {
      int this_rec = tMF_rec(i,j);
      if(this_rec<0)
        continue;
      tMF_don(this_rec,ndon[this_rec]) = int(i);
      ndon[this_rec]++;
    }
  }

  // And labelling as no data the remaining donors
  for (size_t i =0; i< post_stack.size(); i++)
  {
    for(int j = ndon[i]; j<8; j++)
    {
      tMF_don(i,j) = -1;
    }
  }
  
  // Inithalising general attributes
  this->NROWS = NROWS;
  this->NCOLS = NCOLS;
  this->XMIN = XMIN;
  this->XMAX = XMAX;
  this->YMIN = YMIN;
  this->YMAX = YMAX;
  this->XRES = XRES;
  this->YRES = YRES;
  this->NODATAVALUE = NODATAVALUE;
  this->MF_stack = tMF_stack;
  this->MF_receivers = tMF_rec;
  this->MF_lengths = tMF_length;
  this->MF_donors = tMF_don;



  // Now I need to post-process the MF stack

   // I need the pre-accumulation vector to calculate the nodes draining to a certain pit
  xt::pytensor<int,1> pre_contributing_pixels = xt::zeros<int>({pre_stack.size()});
  // Initialising my basin array to 0
  basin_label = xt::zeros<int>({pre_stack.size()});
  pit_to_reroute = std::vector<bool>(pre_stack.size(), false);

  // labeling my pits
  int label = 0;
  for(int i=int(pre_stack.size()-1); i>=0; i--)
  {
    // Current node and its receiver
    int this_node = pre_stack[i];
    int this_rec = pre_rec[this_node];
    // Labelling the basin
    basin_label[this_node] = label;
    // If I reach a base level, my node is its own receiver by convention (Braun et Willett, 2013)
    if(this_rec != this_node)
    {
      // Adding the accumulation
      pre_contributing_pixels[this_rec] += pre_contributing_pixels[this_node]+1 ;
    }
    else
    {
      // Incrementing the label
      label++;
      if(pre_rec[this_node] != post_rec[ this_node])
      {
        // labelling this node to be rerouted. I am differentiating pits from model outlet that way
        pit_to_reroute[this_node] = true;
      }
    }
  }

  // Now ordering my pits by how last they appear in the reverse stack
  // The idea is to process the original pits by order of appearance in the reverse stack
  std::vector<int> order_basin(label+1,-1);
  int this_order = -1;
  for(int i= 0; i< int(pre_stack.size()); i++)
  {
    int this_node = post_stack[i];
    int this_label = basin_label[this_node];
    if(order_basin[this_label] == -1)
    {
      this_order++;
      order_basin[this_label] = -1*this_order;
    }
  }
  for(int i= 0; i< label+1; i++)
  {    
    order_basin[i] += this_order;
  }

  // now reordering the multiple-flow stack by order of base-level processing
  // I am assigning a "basin order" to each node by the minimum basin order of all his receiver
  // then storing the node in temporary local stacks linked to each base levels, but still in the order of 
  std::vector<std::vector<int> > new_MF_stack(this_order+1);
  for(size_t i=0; i< new_MF_stack.size(); i++)
    new_MF_stack[i] = {};
  // Iterating through all nodes in the MF stack and applying the change
  for(auto node: this->MF_stack)
  {
    std::vector<int> rec = this->get_MF_receivers_at_node(node);
    // Initialising the minimum order to the maximum +1
    int min_order = this_order + 1;
    for(auto recnode:rec)
    {
      // if receiver is valid
      if(recnode>=0)
      {
        // if this order is lower than the current one, I save it
        if(order_basin[basin_label[recnode]]<min_order)
          min_order = order_basin[basin_label[recnode]];
      }
    }
    // This happens whan I am a base-level, hence my basin order is the one of my own node
    if(min_order == this_order + 1)
      min_order = order_basin[basin_label[node]];
    // Assigning that node to the temporary stack
    new_MF_stack[min_order].push_back(node);
  }

  // now recreating the MFstack with the correct order
  int incr = 0;
  for(int i =0; i < this_order; i++)
  {
    std::vector<int>& this_sub_stack = new_MF_stack[i];
    for (auto node : this_sub_stack)
    {
      this->MF_stack[incr] = node;
      incr++;
    }
  }

  return;

  
// The following part of this function was a first test preprocessing depressions before solving them
// It failed to capture the essences of subdepressions correctly so I changed the method to something more universal
// I am keeping it  in case I come back to it later
//      _                               _           _ 
//     | |                             | |         | |
//   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
//  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
//  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
//            | |                                     
//            |_|                                      


  // labelisation for pit ID, starting at -1 as I increment prior to pushing back
  int this_pit_ID = -1;
  // this vector will contain -1 if the node is not in a pit and pitID if it is
  pits_ID = std::vector<int>(pre_stack.size(),-1);

 

  // First I need the accumulation vector of the prestack,
  // I also labelise the basins as it will be needed for the regroupping
  // To do so I iterate through the stack backward
  // int label = 0;
  for(int i=int(pre_stack.size()-1); i>=0; i--)
  {
    // Current node and its receiver
    int this_node = pre_stack[i];
    int this_rec = pre_rec[this_node];
    // Labelling the basin
    basin_label[this_node] = label;
    // If I reach a base level, my node is its own receiver by convention (Braun et Willett, 2013)
    if(this_rec != this_node)
    {
      // Adding the accumulation
      pre_contributing_pixels[this_rec] += pre_contributing_pixels[this_node]+1 ;
    }
    else
    {
      // Incrementing the label
      label++;
    }
  }
  // Done with the pre labelling

  // Now I need some sort of hiererchy in my basins, with the corrected path.
  std::vector<int> score_basin(label + 1, -1);

  int score_incrementor = 0;
  for(auto node: this->MF_stack)
  {
    int this_label = basin_label[node];
    if(node == pre_rec[node])
    {
      score_basin[this_label] = score_incrementor;
      score_incrementor++;
    }

  }



  // First step is to register the pits before correction by Cordonnier et al., 2019
  // I am therefore detecting where there are internal base levels 
  for(size_t i=0; i< pre_stack.size(); i++)
  {
    // Getting current node and its receiver pre/post correction
    int this_node = pre_stack[i];
    int this_receiver = pre_rec[this_node];
    int tpost_rec = post_rec[this_node];

    // Checking if it is a pit, which is equivalent to checking if:
    // my node is draining to itself pre-correction but not post-correction
    // If still draining to itself post-correction, it is a model base level, i.e. an outlet 
    if(this_node == this_receiver && this_receiver != tpost_rec)
    {
      // Right, I am at the bottom of a depression, a pit
      // Incrementing the pit ID
      this_pit_ID++;

      // Registering it
      this->pits_ID[this_node] = this_pit_ID;
      // I know that I will save the deposition/erosion flux at that node ID to back correct it if necessary
      this->register_deposition_flux[this_node] = 0;
      this->register_erosion_flux[this_node] = 0;

      // The bottom of the pit is this node POTENTIAL OPTIMISATION HERE: PREDEFINE A NUMBER OF NODES IN THE PIT
      pits_bottom.push_back(this_node);
      // initialising the number of pixels to 1
      pits_npix.push_back(1);

      // I need to find its outlet here, so I will follow the receiving correction until I fall in a node in a different basin
      int this_pit_outlet = post_rec[this_node]; // starting at this node 
      int this_basin_label = basin_label[this_node]; // saving the label
      int label_receiver = basin_label[post_rec[this_pit_outlet]]; // and the receiving label
      int elevation_node = this_pit_outlet; // the elevation of the outlet will be the highest of the two nodes at the outlet
      // Iterating until I either find a new basin or an model base-level (depression splilling water out of the model)
      while(this_basin_label == label_receiver && this_pit_outlet != post_rec[this_pit_outlet] && score_basin[this_basin_label] >= score_basin[label_receiver])
      {
        elevation_node = this_pit_outlet;
        this_pit_outlet = post_rec[this_pit_outlet];
        this_basin_label = basin_label[this_pit_outlet];
        label_receiver = basin_label[post_rec[this_pit_outlet]];
      }

      // Checking which of the past two nodes have the same elevation
      if(elevation[elevation_node] < elevation[this_pit_outlet])
        elevation_node = this_pit_outlet;

      // saving the outlet
      this->pits_outlet.push_back(this_pit_outlet);

      // Quack
      //   __
      // <(o )___
      //  ( ._> /
      //   `---'  
      // Quack

      // Saving the elevation iof this local outlet
      double outlet_elevation = elevation[elevation_node];

      // initialising the volume of the pit to zero
      this->pits_volume.push_back(0);
      //Initialising the list ofpixels for each pits
      this->pits_pixels.push_back({});

      // Getting the basin label right
      this->pits_baslab.push_back(this->basin_label[this_node]);

      // Getting all the node draining into that pit and detecting which one are below the elevation
      for(size_t j = i; j <= i+pre_contributing_pixels[this_node]; j++)
      {
        // Waht is this node
        int tested_node = pre_stack[j];
        // If within the pit, I register it and add to the volume
        if(elevation[tested_node]<outlet_elevation)
        {
          pits_ID[tested_node] = this_pit_ID;
          register_deposition_flux[tested_node] = 0;
          register_erosion_flux[tested_node] = 0;
          pits_volume[this_pit_ID] += XRES*YRES*(outlet_elevation-elevation[tested_node]);
          pits_npix[this_pit_ID] += 1;
          pits_pixels[this_pit_ID].push_back(tested_node);
        }
      //Done with labelling that pit
      }
      // Available volume for sediment so far is the pit volume
      this->pits_available_volume_for_sediments.push_back(pits_volume[this_pit_ID]);

    // done with checking that node
    }
  // Done with labelling all pits
  }


//    quack quack  __
//             ___( o)>
//             \ <_. )
//    ~~~~~~~~  `---'  

  // Saving my preacc, I might need it for later (TO DELETE IF NOT)
  this->preacc = pre_contributing_pixels;

  // Now I need to deal with depression hierarchy
  // Something I may have forgot to consider in the v0.01 of this code: some depression are intricated
  // A simple solution is to go through the multiple stack and decide which depression is actually a subset of the new one
  // then when I'll be filling my depressions, I'll jsut have to keep in mind already existing lake depths

  // First initialising the vector to the right size
  this->sub_depressions = std::vector<std::vector<int> >(pits_npix.size());
  for (size_t i=0; i<pits_npix.size(); i++)
    this->sub_depressions[i] = {};

  // keeping track of which pit I processed
  std::vector<bool> is_pit_processed(pits_npix.size(),false);
  for(auto& node:this->MF_stack)
  {
    int tID = this->pits_ID[node];
    // Ignoring non pits
    if(tID <0)
      continue;
    // Ignoring already processed pits
    if(is_pit_processed[tID] == true)
      continue;

    // unprocessed pit
    // Its ID:
    int this_outlet = this->pits_outlet[tID];
    // Its outlet pit ID
    int oID = this->pits_ID[this_outlet];

    // My pit is processed
    is_pit_processed[tID] = true;

    // If my outlet is not a pit, the current pit is not a subset of another large pit
    if (oID < 0)
      continue;

    // else it is a subset of its donor pit
    this->sub_depressions[oID].push_back(tID);
  }

  // Done with the sub-depression routine

  // Initialising the argument for calculating inherited lake waters
  this->has_excess_water_from_lake = std::vector<bool>(pre_stack.size(),false);
  this->pits_inherited_water_volume = std::vector<double>(this_pit_ID+1, 0);

}


xt::pytensor<int,1> NodeGraph::get_all_nodes_in_depression()
{
//      _                               _           _ 
//     | |                             | |         | |
//   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
//  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
//  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
//            | |                                     
//            |_|                                    

  xt::pytensor<int,1> output = xt::zeros<int>({size_t(NROWS*NCOLS)});

  for(auto vec:pits_pixels)
  {
    for(auto node:vec)
      output[node]=1;
  }
  return output;
}


void NodeGraph::calculate_inherited_water_from_previous_lakes(xt::pytensor<double,1>& previous_lake_depth, xt::pytensor<int,1>& post_rec)
{
     // _                               _           _ 
//     | |                             | |         | |
//   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
//  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
//  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
//            | |                                     
//            |_|                                    

  // Iterating through the inverse stack
  std::vector<bool> is_processed(post_rec.size(),false);
  for(size_t i = 0; i<post_rec.size();i++)
  {
    // getting current node ID
    int this_node = this->MF_stack[i];
    double this_lake_depth = previous_lake_depth[this_node];

    // Checking if there is any excess of water 
    if(this_lake_depth == 0)
    {
      is_processed[this_node] = true;
      continue;
    }

    if(is_processed[this_node])
      continue;
    
    is_processed[this_node] = true;
    // alright, I have unprocessed excess of water there
    double excess_volume = this_lake_depth * XRES * YRES;

    // does it fall into an existing pit
    int this_pit_ID = this->pits_ID[this_node];
    int outlet_node = -9999; // triggering segfault if this happens not to be modified. #ExtremeDebugging
    if(this_pit_ID>=0)
    {
      // Yes, I am just adding the excess water to the pit outlet
      this->pits_inherited_water_volume[this_pit_ID] += excess_volume;
      continue;
    }
    else
    {
      outlet_node = post_rec[this_node];
    }
    
    if(this->has_excess_water_from_lake[outlet_node])
      excess_volume += node_to_excess_of_water[outlet_node];

    this->node_to_excess_of_water[outlet_node] = excess_volume;
    this->has_excess_water_from_lake[outlet_node] = true;

  }

}





// This function preprocesses the stack after coorection by Cordonnier et al., 2019. It keeps the order but "unroute" the pits so that the depression solving is not affected
std::vector<xt::pytensor<int,1> > preprocess_stack(xt::pytensor<int,1>& pre_stack, xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_stack, xt::pytensor<int,1>& post_rec)
{ 
// This function was my test #2: I was preprocessng the single flow stack before calculating the multiple stack in the hope to get it right.
// It crashes the fortran code because it enforced some circularity in the MF stack, which would be quite convoluted to get rid of compared to the new version of the code 
//      _                               _           _ 
//     | |                             | |         | |
//   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
//  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
// | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
//  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
//            | |                                     
//            |_|                                    
  // I'll need basin labels
  xt::pytensor<int,1> baslab = xt::zeros<int>({pre_stack.size()});
  // I am keeping tracks of the base levels
  std::vector<int> base_levels;base_levels.reserve(pre_stack.size());
  std::vector<int> base_levels_index;base_levels_index.reserve(pre_stack.size());
  // I will return which pits have been rerouted and needs to be processes
  xt::pytensor<int,1> rerouted_pits = xt::zeros<bool>({pre_stack.size()});

  // First I am labelling the basins
  int this_label = 0;
  for(int i = pre_stack.size() -1 ; i>=0; i--)
  {
    int this_node = pre_stack[i];
    int this_receiver = pre_rec[this_node];
    baslab[this_node] = this_label;
    // if base-level in the prestack (pre correction) then different basin
    if(this_node == this_receiver)
    {
      if(this_node == 2075)
        std::cout << "flub: "<< std::endl;
      base_levels.emplace_back(this_node);
      base_levels_index.emplace_back(i);
      this_label++;
    }
  }
  // cleaning space
  base_levels.shrink_to_fit();
  base_levels_index.shrink_to_fit();

  // Now I am back calculating the corrections on the stack while keeping the right order in the basin calculations
  for(size_t i=0; i<base_levels.size(); i++)
  {   
    int node = base_levels[i];
    int index = base_levels_index[i];
    if(node == 2075)
        std::cout << "bulf: " << std::endl;
    if(pre_rec[node] == post_rec[node])
    {
      // If the receiver has never been corrected -> this is a true base level and fluxes can escape
      rerouted_pits[node] = -1;
      continue;
    }
    if(node == 2075)
        std::cout << "bulf2 " << std::endl;

    // For each correction I gather the node to correct and their receivers
    std::vector<int> nodes_to_correct;
    std::vector<int> rec_to_correct;

    // Dealing with current node first
    int this_node = node;
    int this_rec = post_rec[node];

    // While I am (i) still in the same basin (ii) the node has been corrected and (iii) I am not at a base-level, I keep on gathering nodes
    // Note that if I reach a "true" base level it means my depression is outleting outside of the model
    while(this_rec != pre_rec[this_node] && this_node != this_rec)
    {
      if(node == 2075)
        std::cout << "bulf3 " << std::endl;
      // Above conditions are met: this node will be to be corrected
      nodes_to_correct.push_back(this_node);
      rec_to_correct.push_back(this_rec);
      // preparing the next one
      this_node = this_rec;
      this_rec = post_rec[this_node];
      if(baslab[this_rec] != baslab[this_node])
      {
        nodes_to_correct.push_back(this_node);
        rec_to_correct.push_back(this_rec);
        break;
      }
    }

    // I have supposidely gathered everything. Just checking I have at least one node in the thingy
    if(nodes_to_correct.size()>0)
    {
      // The pit bottom is rerouted to itself -> CORRECTION: to ensure the mstack to keep the basin order, I think i need to make the receiver
      int new_receveir_for_base_level = post_rec[nodes_to_correct[nodes_to_correct.size()-1]];
      while(pre_rec[post_rec[new_receveir_for_base_level]] != pre_rec[new_receveir_for_base_level])
        new_receveir_for_base_level = post_rec[new_receveir_for_base_level];

      post_rec[nodes_to_correct[0]] = new_receveir_for_base_level;


      // For each node, I correct their receiver to the presious one to reroute them to the centre of the pit
      for(size_t i=1; i< nodes_to_correct.size(); i++)
      {
        post_rec[nodes_to_correct[i]] = pre_rec[nodes_to_correct[i]];
      }

      // I need to correct the stack now
      for(int j=0; j<nodes_to_correct.size();j++)
      {
        post_stack[index + j] = nodes_to_correct[nodes_to_correct.size() - 1 - j];
      }

      // I mark this node has to be processed
      rerouted_pits[nodes_to_correct[0]] = 1;
    }
    // Done with this base level
  }

  // For some reason, this function did not modify my arrays in place, idk why... So I return the results instead which could be otpimised in the future to avoid copy
  std::vector<xt::pytensor<int,1> > output;output.reserve(3);
  output.emplace_back(post_rec);
  output.emplace_back(post_stack);
  output.emplace_back(rerouted_pits);

  // returning output
  return output;
}









// // OLDER TESTS!!!


// // This empty constructor is just there to have a default one.
// void cppintail::create()
// {
// 	std::string yo = "I am an empty constructor yo!";

// }

// // Basic constructor
// void cppintail::create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE)
// {
// 	// I think all of these are pretty explicit
// 	XMIN = tXMIN;
// 	XMAX = tXMAX;
// 	YMIN = tYMIN;
// 	YMAX = tYMAX;
// 	XRES = tXRES;
// 	YRES = tYRES;
// 	NROWS = tNROWS;
// 	NCOLS = tNCOLS;
// 	NODATAVALUE = tNODATAVALUE;

  

// }

// //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // This function compute the flow direction
// // -> DEM: numpy array
// //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// void cppintail::compute_neighbors(xt::pytensor<float,2>& DEM)
// {
//   // Initialising flowdir
//   FLOWDIR = xt::zeros<int>({NROWS,NCOLS});

//   // First processing the centre of the DEM
//   // Avoiding the edgae to avoid having to test systematically if I am close to the edge
//   for(size_t i = 1; i < NROWS - 1; i++)
//   for(size_t j = 1; j < NCOLS - 1; j++)
//   {
//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     float this_elevation = DEM(i,j);
//     if(DEM(i-1, j-1) < this_elevation)
//       FLOWDIR(i,j) += 1;
//     if(DEM(i-1, j) < this_elevation)
//       FLOWDIR(i,j) += 10;
//     if(DEM(i-1, j+1) < this_elevation)
//       FLOWDIR(i,j) += 100;
//     if(DEM(i, j+1) < this_elevation)
//       FLOWDIR(i,j) += 1000;
//     if(DEM(i+1, j+1) < this_elevation)
//       FLOWDIR(i,j) += 10000;
//     if(DEM(i+1, j) < this_elevation)
//       FLOWDIR(i,j) += 100000;
//     if(DEM(i+1, j-1) < this_elevation)
//       FLOWDIR(i,j) += 1000000;
//     if(DEM(i, j-1) < this_elevation)
//       FLOWDIR(i,j) += 10000000;
//   }
//   std::cout << "DEBUG::done with core flowdir" << std::endl;

//   // WNow I am looping through the edges, I can add many tests as there are much less nodes
//   // 🦆
//   for(size_t i = 0; i < NROWS ; i++)
//   {
//     // FIRST COLUMN
//     size_t j = 0;
//     float this_elevation = DEM(i,j);

//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     if(i > 0)
//     {
//       if(DEM(i-1, j) < this_elevation)
//         FLOWDIR(i,j) += 10;
//       if(DEM(i-1, j+1) < this_elevation)
//         FLOWDIR(i,j) += 100;
//     }

//     if(i < NROWS - 1)
//     {
//       if(DEM(i+1, j+1) < this_elevation)
//         FLOWDIR(i,j) += 10000;
//       if(DEM(i+1, j) < this_elevation)
//         FLOWDIR(i,j) += 100000;
//     }

//     if(DEM(i, j+1) < this_elevation)
//       FLOWDIR(i,j) += 1000;    


//     // LAST column
//     j = NCOLS - 1;
//     this_elevation = DEM(i,j);

//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     if(i > 0)
//     {
//       if(DEM(i-1, j-1) < this_elevation)
//         FLOWDIR(i,j) += 1;
//       if(DEM(i-1, j) < this_elevation)
//         FLOWDIR(i,j) += 10;
//     }
//     if(i < NROWS - 1)
//     {
//       if(DEM(i+1, j) < this_elevation)
//         FLOWDIR(i,j) += 100000;
//       if(DEM(i+1, j-1) < this_elevation)
//         FLOWDIR(i,j) += 1000000;
//     }

//     if(DEM(i, j-1) < this_elevation)
//       FLOWDIR(i,j) += 10000000;
//   }

//   for(size_t j = 1; j < NCOLS - 1; j++)
//   {

//     size_t i = 0;
//     float this_elevation = DEM(i,j);
    
//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     if(j>0)
//     {
//       if(DEM(i+1, j-1) < this_elevation)
//         FLOWDIR(i,j) += 1000000;
//       if(DEM(i, j-1) < this_elevation)
//         FLOWDIR(i,j) += 10000000;
//     }

//     if(j < NCOLS - 1)
//     {
//       if(DEM(i, j+1) < this_elevation)
//         FLOWDIR(i,j) += 1000;
//       if(DEM(i+1, j+1) < this_elevation)
//         FLOWDIR(i,j) += 10000;
//     }

//     if(DEM(i+1, j) < this_elevation)
//       FLOWDIR(i,j) += 100000;

//     i = NROWS - 1;
//     this_elevation = DEM(i,j);
//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     if(j > 0)
//     { 
//       if(DEM(i-1, j-1) < this_elevation)
//         FLOWDIR(i,j) += 1;
//       if(DEM(i, j-1) < this_elevation)
//         FLOWDIR(i,j) += 10000000;
//     }
//     if(j < NCOLS - 1)
//     {
//       if(DEM(i-1, j+1) < this_elevation)
//         FLOWDIR(i,j) += 100;
//       if(DEM(i, j+1) < this_elevation)
//         FLOWDIR(i,j) += 1000;
//     }
//     if(DEM(i-1, j) < this_elevation)
//       FLOWDIR(i,j) += 10;

//   }
//   std::cout << "DEBUG::done with edge flowdir" << std::endl;

// }

// //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// // This function converts flow direction to list of row-col of receivers
// // -> DEM: numpy array
// //-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// void cppintail::flowdir_to_receiver_indices(int nodeID, std::vector<int>& receiver_nodes)
// {
//   int row,col; this->node_to_row_col(nodeID,row,col);
//   std::vector<int> receiver_rows, receiver_cols;
//   this->flowdir_to_receiver_indices( row,  col,  receiver_rows,  receiver_cols);
//   receiver_nodes = std::vector<int>(receiver_rows.size());
//   for(size_t i =0; i<receiver_rows.size();i++)
//   {
//     int nodeID,row = receiver_rows[i], col = receiver_cols[i];
//     nodeID = this->row_col_to_node(row,col);
//     receiver_nodes[i] = nodeID;
//   }

// }
// void cppintail::flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols)
// {

//   // Making sure the vectors are empty
//   receiver_rows = std::vector<int>(8);
//   receiver_cols = std::vector<int>(8);  
//   int this_code = FLOWDIR(row,col);
//   if(this_code - 10000000 >= 0)
//   {
//     receiver_rows.emplace_back(row - 1);
//     receiver_cols.emplace_back(col - 1);
//     this_code -= 10000000;
//   }

//   if(this_code - 1000000 >= 0)
//   {
//     receiver_rows.emplace_back(row - 1);
//     receiver_cols.emplace_back(col);
//     this_code -= 1000000;
//   }

//   if(this_code - 100000 >= 0)
//   {
//     receiver_rows.emplace_back(row - 1);
//     receiver_cols.emplace_back(col + 1);
//     this_code -= 100000;
//   }

//   if(this_code - 10000 >= 0)
//   {
//     receiver_rows.emplace_back(row );
//     receiver_cols.emplace_back(col + 1);
//     this_code -= 10000;
//   }

//   if(this_code - 1000 >= 0)
//   {
//     receiver_rows.emplace_back(row + 1);
//     receiver_cols.emplace_back(col + 1);
//     this_code -= 1000;
//   }

//   if(this_code - 100 >= 0)
//   {
//     receiver_rows.emplace_back(row + 1);
//     receiver_cols.emplace_back(col);
//     this_code -= 100;
//   }

//   if(this_code - 10 >= 0)
//   {
//     receiver_rows.emplace_back(row + 1);
//     receiver_cols.emplace_back(col - 1);
//     this_code -= 10;
//   }

//   if(this_code - 1 >= 0)
//   {
//     receiver_rows.emplace_back(row + 1);
//     receiver_cols.emplace_back(col);
//     this_code -= 1;
//   }

//   receiver_rows.shrink_to_fit();
//   receiver_cols.shrink_to_fit();

//   std::vector<int> new_vecrow, new_vecol;

//   for (size_t i = 0; i < receiver_rows.size(); i++ )
//   {
//     if(receiver_rows[i] < 0 || receiver_cols[i] < 0 || receiver_rows[i] >= NROWS || receiver_cols[i] >= NCOLS)
//       continue;
//     else
//     {
//       new_vecrow.push_back(receiver_rows[i] );
//       new_vecol.push_back(receiver_cols[i]);
//     }
//   }

//   receiver_rows = new_vecrow;
//   receiver_cols = new_vecol;


// }



// void cppintail::find_nodes_with_no_donors( xt::pytensor<float,2>& DEM)
// {
//   // Attribute containing the no donors nodes
//   no_donor_nodes = std::vector<int>(NROWS * NCOLS);
//   std::vector<int> local_ndonors(NROWS * NCOLS,0);
//   // std::cout << "DEBUG:HERE1" << std::endl;
//   // Looping through the thingy
//   int incrementer = 0;
//   for(size_t i = 0; i < NROWS; i++)
//   for(size_t j = 0; j < NCOLS; j++)
//   {
//     // Ignorign no data
//     if(DEM(i,j) == NODATAVALUE)
//       continue;

//     // Getting the receivers
//     std::vector<int> receiver_rows,receiver_cols;
//     // std::cout << "DEBUG:HERE1.5 || " << i << "||" << j << std::endl;
//     this->flowdir_to_receiver_indices(int(i), int(j), receiver_rows,  receiver_cols);
//     // std::cout << "DEBUG:HERE1.6 || " << i << "||" << j << std::endl;

//     // Incrementing the receivers
//     for(size_t od=0; od<receiver_rows.size(); od++)
//     {
//       // std::cout << "DEBUG:HERE1.7 || " << receiver_rows[od] << "||" << receiver_cols[od] << std::endl;
//       local_ndonors[row_col_to_node(receiver_rows[od],receiver_cols[od])] += 1; 
//     }
//   }
//   // std::cout << "DEBUG:HERE2" << std::endl;


//   // hunting for the no donor nodes
//   for(size_t i = 0; i < local_ndonors.size(); i++)
//   {
//     if(local_ndonors[i] == 0)
//     {
//       no_donor_nodes[incrementer]  = int(i);
//     }
//   }
//   // std::cout << "DEBUG:HERE3" << std::endl;


//   no_donor_nodes.shrink_to_fit();

// }







// struct tempNode
// {
//   /// @brief Elevation data.
//   float Zeta;
//   /// @brief Row index value.
//   int NodeIndex;
// };

// bool operator>( const tempNode& lhs, const tempNode& rhs )
// {
//   return lhs.Zeta > rhs.Zeta;
// }
// bool operator<( const tempNode& lhs, const tempNode& rhs )
// {
//   return lhs.Zeta < rhs.Zeta;
// }


// void cppintail::Initialise_MF_stacks(xt::pytensor<float,2>& DEM)
// {

//   // Initialise my stack to the maximum possible size (data - no data)
//   MF_stack.clear();
//   MF_stack.reserve(NROWS * NCOLS);

//   // Sorting the node 
//   std::priority_queue< tempNode, std::vector<tempNode>, std::greater<tempNode> > PriorityQueue;
//   for(size_t row = 0; row<NROWS; row++)
//   for(size_t col = 0; col<NCOLS; col++)
//   {
//     tempNode this_node;
//     int this_nodeID = this->row_col_to_node(row, col);
//     if(this_nodeID == NODATAVALUE)
//       continue;

//     this_node.Zeta = DEM(row,col);
//     this_node.NodeIndex =  this_nodeID;
//     PriorityQueue.push(this_node);
//   }

//   while(!PriorityQueue.empty())
//     MF_stack.emplace_back(PriorityQueue.top().NodeIndex);

//   MF_stack.shrink_to_fit();
// }

// void cppintail::compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM)
// {

//   Drainage_area = xt::zeros<double>({NROWS,NCOLS});
  
//   for(int i = int(MF_stack.size()); i >= 0; i++)
//   {
//     int this_node = MF_stack[i];
//     int row,col; node_to_row_col(this_node,row,col);
//     // Adding the first thingy
//     Drainage_area(row,col) += XRES*YRES;
//     // Getting the neighbors
//     std::vector<int> neighrow, neighcol;
//     this->flowdir_to_receiver_indices(row, col, neighrow, neighcol);
    
//     if(neighrow.size() == 0)
//       continue;

//     std::vector<double> slope_rep(neighrow.size());
//     double max_slope = std::numeric_limits<double>::min(); 
//     for(size_t tn = 0; tn < neighrow.size(); tn++)
//     {
//       int nrow = neighrow[tn];
//       int ncol = neighcol[tn];
//       double dx = std::sqrt( std::pow( std::abs(nrow - row) * YRES,2) + std::pow(std::abs(ncol - col) * YRES,2) );
//       double dz = DEM(row,col) - DEM(nrow,ncol);

//       double this_slope = std::pow(dz/dx, slexponent);
//       slope_rep[tn] = this_slope;
//       if(max_slope<this_slope)
//         max_slope = this_slope;
//     }

//     for(size_t tn = 0; tn < neighrow.size(); tn++)
//     {
//       slope_rep[tn] = slope_rep[tn]/max_slope;
//       int nrow = neighrow[tn];
//       int ncol = neighcol[tn];
//       Drainage_area(nrow,ncol) = Drainage_area(nrow,ncol) + slope_rep[tn] * Drainage_area(row,col);
//     }
//   }
//   // Done
// }


//  //###############################################  
//  // Duck transition to general functions
//  //           ,-.
//  //       ,--' ~.).
//  //     ,'         `.
//  //    ; (((__   __)))
//  //    ;  ( (#) ( (#)
//  //    |   \_/___\_/|
//  //   ,"  ,-'    `__".
//  //  (   ( ._   ____`.)--._        _
//  //   `._ `-.`-' \(`-'  _  `-. _,-' `-/`.
//  //    ,')   `.`._))  ,' `.   `.  ,','  ;
//  //  .'  .     `--'  /     ).   `.      ;
//  // ;     `-        /     '  )         ;
//  // \                       ')       ,'
//  //  \                     ,'       ;
//  //   \               `~~~'       ,'
//  //    `.                      _,'
//  //      `.                ,--'
//  //        `-._________,--'


// //##################################################
// //############# Stack stuff ########################
// //##################################################
// // Adapted from xarray-topo




#endif