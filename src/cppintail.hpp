//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#ifndef cppintail_HPP
#define cppintail_HPP

// STL imports
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <fstream>
#include <queue>
#include <iostream>
#include <numeric>
#include <cmath>

// All the xtensor requirements
#include "xtensor-python/pyarray.hpp" // manage the I/O of numpy array
#include "xtensor-python/pytensor.hpp" // same
#include "xtensor-python/pyvectorize.hpp" // Contain some algorithm for vectorised calculation (TODO)
#include "xtensor/xadapt.hpp" // the function adapt is nice to convert vectors to numpy arrays
#include "xtensor/xmath.hpp" // Array-wise math functions
#include "xtensor/xarray.hpp"// manages the xtensor array (lower level than the numpy one)
#include "xtensor/xtensor.hpp" // same


class cppintail
{
  public:
  
    cppintail() { create(); }

    void compute_neighbors(xt::pytensor<float,2>& DEM);

    void flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols);

    void Initialise_MF_stacks(xt::pytensor<float,2>& DEM);

    void compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM);


    // This function transform the linearised node indice to row/col
    inline void node_to_row_col(int& node, int& row, int& col)
    {
        col = node % NCOLS;
        row = int((node - col)/NCOLS);
    };

    inline int row_col_to_node(int& row, int& col){return row * NCOLS + col;};
    inline int row_col_to_node(size_t& row, size_t& col){return int(row * NCOLS + col);};
  

  protected:

    // Geometrical/geographical features
    float XMIN;
    float XMAX;
    float YMIN;
    float YMAX;
    float XRES;
    float YRES;
    int NROWS;
    int NCOLS;
    float NODATAVALUE;

    // flow - directions
    // Binary system to detect where the flow goes
    // each number is 0 for block and 1 for goes
    // position is around the pixel:
    // XXXXXXXX:
    // 1,2,3
    // 8,0,4
    // 7,6,5
    // For example:
    // 10010111 means flows in 5 directions
    xt::pytensor<int,2> FLOWDIR;

    // Drainage Area
    xt::pytensor<float,2> Drainage_area;

    // MF stacks: to deal with multiple flow direction
    std::vector<int> MF_stack;





  private:
    void create();
    void create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE);

};




#endif