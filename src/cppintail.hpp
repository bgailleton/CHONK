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


// this class organises the DEM nodes in order to solve all equations in the right order. This is the first step of each iteration of the model 
class NodeGraph
{
  public:
    NodeGraph() { create(); }
    NodeGraph(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, 
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec, xt::pytensor<double,1>& elevation, 
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE)
    {create( pre_stack, pre_rec, tMF_stack, tMF_rec, elevation, XMIN,  XMAX,  YMIN,  YMAX,  XRES, YRES, NROWS, NCOLS, NODATAVALUE);}

    inline int row_col_to_node(int& row, int& col){return row * NCOLS + col;};
    inline int row_col_to_node(size_t& row, size_t& col){return int(row * NCOLS + col);};
    // This function transform the linearised node indice to row/col
    inline void node_to_row_col(int& node, int& row, int& col)
    {
        col = node % NCOLS;
        row = int((node - col)/NCOLS);
    };

  protected:
    // Geometrical/geographical features, their name should be self-explanatory
    float XMIN;
    float XMAX;
    float YMIN;
    float YMAX;
    float XRES;
    float YRES;
    int NROWS;
    int NCOLS;
    float NODATAVALUE;

    // These are the stacks ingested from fastscaplib_fortran
    xt::pytensor<int,1> MF_stack;
    xt::pytensor<int,2> MF_receivers;

    // Number of depressions
    int n_pits;
    // length=N_nodes, -1 if not in a pit, pit_ID otherwise
    std::vector<int> pits_ID;
    // length = n_pits, pit_ID to bottom_nodes
    std::vector<int> pits_bottom;
    // length = n_pits, pit_ID to outlet node. If equal to bottom: fluxes can escape the model
    std::vector<int> pits_outlet; 
    // legth = n_pits, pit_ID to number of pixels in the pit.
    std::vector<int> pits_npix; 
    // list of pixels in each pits
    std::vector<std::vector<int> > pits_pixels; 
    // length = n_pits, pit_ID to colume in L^3
    std::vector<double> pits_volume;

  private:
    void create();
    void create(xt::pytensor<int,1>& pre_stack,xt::pytensor<int,1>& pre_rec, 
  xt::pytensor<int,1>& tMF_stack, xt::pytensor<int,2>& tMF_rec, xt::pytensor<double,1>& elevation, 
  float XMIN, float XMAX, float YMIN, float YMAX, float XRES, float YRES, int NROWS, int NCOLS, float NODATAVALUE);



};





// Older tests


class cppintail
{
  public:
  
    cppintail() { create(); }
    cppintail(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE) { create(tXMIN, tXMAX, tYMIN, tYMAX, tXRES, tYRES, tNROWS, tNCOLS, tNODATAVALUE); }

    void compute_neighbors(xt::pytensor<float,2>& DEM);
    
    void flowdir_to_receiver_indices(int nodeID, std::vector<int>& receiver_nodes);
    void flowdir_to_receiver_indices(int row, int col, std::vector<int>& receiver_rows, std::vector<int>& receiver_cols);

    void Initialise_MF_stacks(xt::pytensor<float,2>& DEM);

    void compute_DA_slope_exp( double slexponent, xt::pytensor<float,2>& DEM);

    void find_nodes_with_no_donors( xt::pytensor<float,2>& DEM);



    // This function transform the linearised node indice to row/col
    inline void node_to_row_col(int& node, int& row, int& col)
    {
        col = node % NCOLS;
        row = int((node - col)/NCOLS);
    };

    inline int row_col_to_node(int& row, int& col){return row * NCOLS + col;};
    inline int row_col_to_node(size_t& row, size_t& col){return int(row * NCOLS + col);};


    //Getter
    std::vector<int> get_label_from_nodonode(){return label_from_nodonodes;};
    xt::pytensor<int,2> get_flowdir(){return FLOWDIR;};
  

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

    // Node with no donor
    std::vector<int> no_donor_nodes, label_from_nodonodes;
    

    // MF stacks: to deal with multiple flow direction
    std::vector<int> MF_stack;





  private:
    void create();
    void create(float tXMIN, float tXMAX, float tYMIN, float tYMAX, float tXRES, float tYRES, int tNROWS, int tNCOLS, float tNODATAVALUE);

};







#endif