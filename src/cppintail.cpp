#ifndef cppintail_CPP
#define cppintail_CPP

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

#include "cppintail.hpp"



// This empty constructor is just there to have a default one.
void NodeGraph::create()
{
  std::string yo = "I am an empty constructor yo!";

}


// This empty constructor is just there to have a default one.
void NodeGraph::create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, xt::pytensor<int,1>& post_rec,
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec,xt::pytensor<int,2>& tMF_don, xt::pytensor<double,1>& elevation, xt::pytensor<double,2>& tMF_length,
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
{
  
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

  // labelisation for pit ID, starting at -1 as I increment prior to pushing back
  int this_pit_ID = -1;
  // this vector will contain -1 if the node is not in a pit and pitID if it is
  pits_ID = std::vector<int>(pre_stack.size(),-1);

  // I need the pre-accumulation vector to calculate the nodes draining to a certain pit
  xt::pytensor<int,1> pre_contributing_pixels = xt::zeros<int>({pre_stack.size()});
  // Initialising my basin array to 0
  basin_label = xt::zeros<int>({pre_stack.size()});

  // First I need the accumulation vector of the prestack,
  // I also labelise the basins as it will be needed for the regroupping
  // To do so I iterate through the stack backward
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
      // Incrementing the label
      label++;
    }
  }
  // Done with the pre labelling

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
      while(this_basin_label == label_receiver && this_pit_outlet != post_rec[this_pit_outlet])
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
      pits_outlet.push_back(this_pit_outlet);

      // Quack
      //   __
      // <(o )___
      //  ( ._> /
      //   `---'  
      // Quack

      // Saving the elevation iof this local outlet
      double outlet_elevation = elevation[elevation_node];

      // initialising the volume of the pit to zero
      pits_volume.push_back(0);
      //Initialising the list ofpixels for each pits
      pits_pixels.push_back({});

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