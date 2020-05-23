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
  this->NROWS = NROWS;
  this->NCOLS = NCOLS;
  // Initialising a bunch of variable
  int this_pit_ID = -1;
  pits_ID = std::vector<int>(pre_stack.size(),0);
  for (auto& v : pits_ID)
    v -= 1;
  xt::pytensor<int,1> pre_contributing_pixels = xt::zeros<int>({pre_stack.size()});
  // NEED TO CHECK HERE IF THE ARRAYS ARE COPIED OR VIEWS
  this->MF_stack = tMF_stack;
  this->MF_receivers = tMF_rec;
  this->MF_lengths = tMF_length;
  this->MF_donors = tMF_don;

  //First I need the accumulation vector of the prestack
  for(int i=int(pre_stack.size()-1); i>=0; i--)
  {
    int this_node = pre_stack(i);
    int this_rec = pre_rec(this_node);
    if(this_rec != this_node)
    {
      pre_contributing_pixels(this_rec) += pre_contributing_pixels(this_node)+1 ;
    }
    // else
    // {
    //   pre_contributing_pixels[this_node] = -1;
    // }
  }

  // first step is to register the pits before correction
  // basically detecting where the stack is receiving itself
  for(size_t i=0; i< pre_stack.size(); i++)
  {
    // Getting current node and its receiver
    int this_node = pre_stack(i);
    int this_receiver = pre_rec(this_node);
    // checking if it is a pit
    if(this_node == this_receiver)
    {

      // Incrementing the pit ID
      this_pit_ID++;
      // Register it
      pits_ID[this_node] = this_pit_ID;
      this->register_deposition_flux[this_node] = 0;
      this->register_erosion_flux[this_node] = 0;

      // The bottom of the pit is this node
      pits_bottom.push_back(this_node);
      // its outlet is the first in the mstack that does NOT fall back into the pit again
      pits_outlet.push_back(post_rec(this_node));
      // initialising the number of pixels to 1
      pits_npix.push_back(1);

      // Saving the outlet of the elevation temporally
      double outlet_elevation = elevation[pits_outlet[this_pit_ID]];
      // if(pits_bottom[this_pit_ID] != pits_outlet[this_pit_ID])
      //   std::cout<< "ELEVATION BOTTOM:" << elevation[pits_bottom[this_pit_ID]]<< " ELEVATION outel:"   << elevation[pits_outlet[this_pit_ID]] << std::endl ;
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
  this->preacc = pre_contributing_pixels;

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