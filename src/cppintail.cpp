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
void cppintail::create()
{
	std::string yo = "I am an empty constructor yo!";

}

// Basic constructor
void cppintail::create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE)
{
	// I think all of these are pretty explicit
	XMIN = tXMIN;
	XMAX = tXMAX;
	YMIN = tYMIN;
	YMAX = tYMAX;
	XRES = tXRES;
	YRES = tYRES;
	NROWS = tNROWS;
	NCOLS = tNCOLS;
	NODATAVALUE = tNODATAVALUE;

  

}

//-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// This function compute the flow direction
// -> DEM: numpy array
//-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
void cppintail::compute_neighbors(xt::pytensor<float,2>& DEM)
{
  // Initialising flowdir
  FLOWDIR = xt::zeros<int>({NROWS,NCOLS});

  // First processing the centre of the DEM
  // Avoiding the edgae to avoid having to test systematically if I am close to the edge
  for(size_t i = 0; i < NROWS - 1; i++)
  for(size_t j = 0; j < NCOLS - 1; j++)
  {
    if(DEM(i,j) == NODATAVALUE)
      continue;

    float this_elevation = DEM(i,j);
    if(DEM(i-1, j-1) < this_elevation)
      FLOWDIR(i,j) += 1;
    if(DEM(i-1, j) < this_elevation)
      FLOWDIR(i,j) += 10;
    if(DEM(i-1, j+1) < this_elevation)
      FLOWDIR(i,j) += 100;
    if(DEM(i, j+1) < this_elevation)
      FLOWDIR(i,j) += 1000;
    if(DEM(i+1, j+1) < this_elevation)
      FLOWDIR(i,j) += 10000;
    if(DEM(i+1, j) < this_elevation)
      FLOWDIR(i,j) += 100000;
    if(DEM(i+1, j-1) < this_elevation)
      FLOWDIR(i,j) += 1000000;
    if(DEM(i, j-1) < this_elevation)
      FLOWDIR(i,j) += 10000000;
  }

  // WNow I am looping through the edges, I can add many tests as there are much less nodes
  // 🦆
  for(size_t i = 0; i < NROWS - 1; i++)
  {
    // FIRST COLUMN
    size_t j = 0;
    float this_elevation = DEM(i,j);

    if(DEM(i,j) == NODATAVALUE)
      continue;

    if(i > 0)
    {
      if(DEM(i-1, j) < this_elevation)
        FLOWDIR(i,j) += 10;
      if(DEM(i-1, j+1) < this_elevation)
        FLOWDIR(i,j) += 100;
    }

    if(i < NROWS - 1)
    {
      if(DEM(i+1, j+1) < this_elevation)
        FLOWDIR(i,j) += 10000;
      if(DEM(i+1, j) < this_elevation)
        FLOWDIR(i,j) += 100000;
    }

    if(DEM(i, j+1) < this_elevation)
      FLOWDIR(i,j) += 1000;    


    // LAST column
    j = NCOLS - 1;
    this_elevation = DEM(i,j);

    if(DEM(i,j) == NODATAVALUE)
      continue;

    if(i > 0)
    {
      if(DEM(i-1, j-1) < this_elevation)
        FLOWDIR(i,j) += 1;
      if(DEM(i-1, j) < this_elevation)
        FLOWDIR(i,j) += 10;
    }
    if(i < NROWS - 1)
    {
      if(DEM(i+1, j) < this_elevation)
        FLOWDIR(i,j) += 100000;
      if(DEM(i+1, j-1) < this_elevation)
        FLOWDIR(i,j) += 1000000;
    }

    if(DEM(i, j-1) < this_elevation)
      FLOWDIR(i,j) += 10000000;
  }

  for(size_t j = 0; j < NROWS - 1; j++)
  {

    size_t i = 0;
    float this_elevation = DEM(i,j);
    
    if(DEM(i,j) == NODATAVALUE)
      continue;
    if(j>0)
    {
      if(DEM(i+1, j-1) < this_elevation)
        FLOWDIR(i,j) += 1000000;
      if(DEM(i, j-1) < this_elevation)
        FLOWDIR(i,j) += 10000000;
    }
    if(j < NCOLS - 1)
    {
      if(DEM(i, j+1) < this_elevation)
        FLOWDIR(i,j) += 1000;
      if(DEM(i+1, j+1) < this_elevation)
        FLOWDIR(i,j) += 10000;
    }

    if(DEM(i+1, j) < this_elevation)
      FLOWDIR(i,j) += 100000;

    i = NROWS - 1;
    this_elevation = DEM(i,j);
    if(DEM(i,j) == NODATAVALUE)
      continue;

    if(j > 0)
    { 
      if(DEM(i-1, j-1) < this_elevation)
        FLOWDIR(i,j) += 1;
      if(DEM(i, j-1) < this_elevation)
        FLOWDIR(i,j) += 10000000;
    }
    if(j < NCOLS - 1)
    {
      if(DEM(i-1, j+1) < this_elevation)
        FLOWDIR(i,j) += 100;
      if(DEM(i, j+1) < this_elevation)
        FLOWDIR(i,j) += 1000;
    }
    if(DEM(i-1, j) < this_elevation)
      FLOWDIR(i,j) += 10;

  }

}

//-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
// This function converts flow direction to list of row-col of receivers
// -> DEM: numpy array
//-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
void cppintail::flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols)
{

  // Making sure the vectors are empty
  receiver_rows = std::vector<int>(8);
  receiver_cols = std::vector<int>(8);  
  int this_code = FLOWDIR(row,col);
  if(this_code - 10000000 >= 0)
  {
    receiver_rows.emplace_back(row - 1);
    receiver_cols.emplace_back(col - 1);
    this_code -= 10000000;
  }

  if(this_code - 1000000 >= 0)
  {
    receiver_rows.emplace_back(row - 1);
    receiver_cols.emplace_back(col);
    this_code -= 1000000;
  }

  if(this_code - 100000 >= 0)
  {
    receiver_rows.emplace_back(row - 1);
    receiver_cols.emplace_back(col + 1);
    this_code -= 100000;
  }

  if(this_code - 10000 >= 0)
  {
    receiver_rows.emplace_back(row );
    receiver_cols.emplace_back(col + 1);
    this_code -= 10000;
  }

  if(this_code - 1000 >= 0)
  {
    receiver_rows.emplace_back(row + 1);
    receiver_cols.emplace_back(col + 1);
    this_code -= 1000;
  }

  if(this_code - 100 >= 0)
  {
    receiver_rows.emplace_back(row + 1);
    receiver_cols.emplace_back(col);
    this_code -= 100;
  }

  if(this_code - 10 >= 0)
  {
    receiver_rows.emplace_back(row + 1);
    receiver_cols.emplace_back(col - 1);
    this_code -= 10;
  }

  if(this_code - 1 >= 0)
  {
    receiver_rows.emplace_back(row + 1);
    receiver_cols.emplace_back(col);
    this_code -= 1;
  }

  receiver_rows.shrink_to_fit();
  receiver_cols.shrink_to_fit();

}


struct tempNode
{
  /// @brief Elevation data.
  float Zeta;
  /// @brief Row index value.
  int NodeIndex;
};

bool operator>( const tempNode& lhs, const tempNode& rhs )
{
  return lhs.Zeta > rhs.Zeta;
}
bool operator<( const tempNode& lhs, const tempNode& rhs )
{
  return lhs.Zeta < rhs.Zeta;
}


void cppintail::Initialise_MF_stacks(xt::pytensor<float,2>& DEM)
{

  // Initialise my stack to the maximum possible size (data - no data)
  MF_stack.clear();
  MF_stack.reserve(NROWS * NCOLS);

  // Sorting the node 
  std::priority_queue< tempNode, std::vector<tempNode>, std::greater<tempNode> > PriorityQueue;
  for(size_t row = 0; row<NROWS; row++)
  for(size_t col = 0; col<NCOLS; col++)
  {
    tempNode this_node;
    int this_nodeID = this->row_col_to_node(row, col);
    if(this_nodeID == NODATAVALUE)
      continue;

    this_node.Zeta = DEM(row,col);
    this_node.NodeIndex =  this_nodeID;
    PriorityQueue.push(this_node);
  }

  while(!PriorityQueue.empty())
    MF_stack.emplace_back(PriorityQueue.top().NodeIndex);

  MF_stack.shrink_to_fit();
}

void cppintail::compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM)
{

  Drainage_area = xt::zeros<double>({NROWS,NCOLS});
  
  for(int i = int(MF_stack.size()); i >= 0; i++)
  {
    int this_node = MF_stack[i];
    int row,col; node_to_row_col(this_node,row,col);
    // Adding the first thingy
    Drainage_area(row,col) += XRES*YRES;
    // Getting the neighbors
    std::vector<int> neighrow, neighcol;
    this->flowdir_to_receiver_indices(row, col, neighrow, neighcol);
    
    if(neighrow.size() == 0)
      continue;

    std::vector<double> slope_rep(neighrow.size());
    double max_slope = std::numeric_limits<double>::min(); 
    for(size_t tn = 0; tn < neighrow.size(); tn++)
    {
      int nrow = neighrow[tn];
      int ncol = neighcol[tn];
      double dx = std::sqrt( std::pow( std::abs(nrow - row) * YRES,2) + std::pow(std::abs(ncol - col) * YRES,2) );
      double dz = DEM(row,col) - DEM(nrow,ncol);

      double this_slope = std::pow(dz/dx, slexponent);
      slope_rep[tn] = this_slope;
      if(max_slope<this_slope)
        max_slope = this_slope;
    }

    for(size_t tn = 0; tn < neighrow.size(); tn++)
    {
      slope_rep[tn] = slope_rep[tn]/max_slope;
      int nrow = neighrow[tn];
      int ncol = neighcol[tn];
      Drainage_area(nrow,ncol) = Drainage_area(nrow,ncol) + slope_rep[tn] * Drainage_area(row,col);
    }
  }
  // Done
}





#endif